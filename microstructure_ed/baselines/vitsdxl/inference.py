"""
Diffusion-path inference: loads a checkpoint and reconstructs one image via the
full SDXL UNet denoising loop conditioned on the image embedding.

Pipeline (v4 — single latent z, AttentionPooler + AdaLN-Zero):
    Image (512x512)
      -> Compressor (ViT-H/14 @ 518 padded + AttentionPooler) -> z (512-D)
      -> SDXLDecoder.generate()
            Channel 1: LatentToTokens(z) (3 AdaLN-Zero blocks) -> (77x2048)
            Channel 2: pooled_proj(z)                          -> (1280-D)
            + EulerDiscrete denoising loop (N steps, frozen UNet)
      -> SDXL VAE decoder -> reconstructed image (512x512)
"""
import os, sys, gc, argparse, logging
from pathlib import Path
from datetime import datetime

import torch
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from microstructure_ed.config import (
    CHECKPOINTS_DIR, INFERENCE_OUTPUT_DIR,
    USE_AMP, NUM_INFERENCE_STEPS,
)
from microstructure_ed.encoder_arch_pretrained import Compressor
from microstructure_ed.baselines.vitsdxl.decoder_arch_pretrained import SDXLDecoder, SDVAE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    t = (tensor.float() / 2 + 0.5).clamp(0, 1)
    return Image.fromarray(
        (t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    )


def compute_psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a.float() - b.float()) ** 2).item()
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(torch.tensor(2.0 / (mse ** 0.5))).item()


def compute_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.mean((a.float() - b.float()) ** 2).item()


def find_latest_checkpoint(checkpoints_dir: str) -> str:
    ckpts = sorted(Path(checkpoints_dir).glob("checkpoint_epoch_*.pth"))
    return str(ckpts[-1]) if ckpts else None


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(
    checkpoint_path: str,
    image_path: str,
    output_dir: str = None,
    guidance_scale: float = 1.0,
    num_steps: int = None,
    seed: int = 42,
):
    """
    Reconstruct a single microstructure image using diffusion conditioned on z.

    Args:
        checkpoint_path  : path to a .pth checkpoint
        image_path       : path to input image (resized to 512x512)
        output_dir       : where to save outputs (default: INFERENCE_OUTPUT_DIR)
        guidance_scale   : CFG scale — 1.0 = conditional only (deterministic)
        num_steps        : denoising steps (default: from config)
        seed             : RNG seed for reproducible outputs
    """
    output_dir = output_dir or INFERENCE_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    if num_steps is None:
        num_steps = NUM_INFERENCE_STEPS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("DIFFUSION-PATH INFERENCE (ViT-H/14 + AttentionPooler + AdaLN-Zero)")
    logger.info(f"  Device          : {device}")
    if device.type == "cuda":
        logger.info(f"  GPU             : {torch.cuda.get_device_name(0)}")
    logger.info(f"  Checkpoint      : {checkpoint_path}")
    logger.info(f"  Image           : {image_path}")
    logger.info(f"  Guidance scale  : {guidance_scale}")
    logger.info(f"  Denoising steps : {num_steps}")
    logger.info(f"  Seed            : {seed}")
    logger.info("=" * 60)

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        sys.exit(1)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    logger.info("Loading checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_epoch = ckpt.get("epoch", 0) + 1
    ckpt_loss  = ckpt.get("loss", float("nan"))
    ckpt_psnr  = ckpt.get("psnr", float("nan"))
    logger.info(
        f"Checkpoint: epoch {ckpt_epoch}  |  "
        f"train_loss {ckpt_loss:.4f}  |  sample_psnr {ckpt_psnr:.2f} dB"
    )

    # ── Build models ──────────────────────────────────────────────────────────
    logger.info("Building models (this loads SDXL UNet)...")

    vae = SDVAE(freeze=True).to(device)

    compressor = Compressor(use_gradient_checkpointing=False).to(device)
    compressor.load_state_dict(ckpt["compressor_state_dict"])
    compressor.eval()

    decoder = SDXLDecoder(p_drop=0.0)
    decoder.token_proj  = decoder.token_proj.to(device)
    decoder.pooled_proj = decoder.pooled_proj.to(device)
    decoder.null_token.data = decoder.null_token.data.to(device)
    decoder.unet        = decoder.unet.to(device)

    decoder.token_proj.load_state_dict(ckpt["token_proj_state_dict"])
    if "pooled_proj_state_dict" in ckpt:
        decoder.pooled_proj.load_state_dict(ckpt["pooled_proj_state_dict"])
    decoder.null_token.data.copy_(ckpt["null_token"])

    decoder.token_proj.eval()
    decoder.pooled_proj.eval()

    del ckpt
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Models ready.")

    # ── Preprocess image ──────────────────────────────────────────────────────
    original_pil = Image.open(image_path).convert("RGB").resize((512, 512))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    image_tensor = transform(original_pil).unsqueeze(0).to(device)
    logger.info(
        f"Input tensor : {image_tensor.shape}  "
        f"range [{image_tensor.min():.2f}, {image_tensor.max():.2f}]"
    )

    # ── Encode: image -> z (ViT-H/14 + AttentionPooler) ───────────────────────
    logger.info("Encoding image via ViT-H/14 + AttentionPooler...")
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if USE_AMP and device.type == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )
    with torch.no_grad(), autocast_ctx:
        embedding = compressor(image_tensor)  # (1, 768)

    logger.info(
        f"Embedding z : {embedding.shape}  "
        f"norm={embedding.norm().item():.2f}  "
        f"range [{embedding.min():.2f}, {embedding.max():.2f}]"
    )

    # ── Generate: z -> SDXL denoising -> image ────────────────────────────────
    logger.info(
        f"Running diffusion denoising  "
        f"({num_steps} steps, guidance_scale={guidance_scale})..."
    )
    image_recon = decoder.generate(
        embedding,
        vae=vae,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        return_latent=False,
    )                                              # (1, 3, 512, 512)

    logger.info(
        f"Output    : {image_recon.shape}  "
        f"range [{image_recon.min():.2f}, {image_recon.max():.2f}]"
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    psnr = compute_psnr(image_tensor, image_recon.to(device))
    mse  = compute_mse(image_tensor,  image_recon.to(device))

    logger.info("QUALITY METRICS")
    logger.info(f"  MSE  : {mse:.6f}")
    logger.info(f"  PSNR : {psnr:.2f} dB")

    # ── Save outputs ──────────────────────────────────────────────────────────
    stem = Path(image_path).stem
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag  = f"{stem}_epoch{ckpt_epoch:03d}_gs{guidance_scale:.1f}_{ts}"

    orig_path  = os.path.join(output_dir, f"{tag}_original.png")
    recon_path = os.path.join(output_dir, f"{tag}_reconstructed.png")
    cmp_path   = os.path.join(output_dir, f"{tag}_comparison.png")

    original_pil.save(orig_path)
    tensor_to_pil(image_recon).save(recon_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_pil)
    axes[0].set_title(f"Original\n{Path(image_path).name}", fontsize=11)
    axes[0].axis("off")
    axes[1].imshow(tensor_to_pil(image_recon))
    axes[1].set_title(
        f"Reconstructed | Epoch {ckpt_epoch} | GS {guidance_scale:.1f}\n"
        f"PSNR: {psnr:.2f} dB  |  MSE: {mse:.5f}",
        fontsize=11,
    )
    axes[1].axis("off")
    plt.suptitle(
        "Microstructure Reconstruction — ViT-H/14 + AttentionPooler + AdaLN-Zero",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(cmp_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved -> {orig_path}")
    logger.info(f"Saved -> {recon_path}")
    logger.info(f"Saved -> {cmp_path}")
    logger.info("Inference complete.")

    return {"psnr": psnr, "mse": mse}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct a microstructure image via SDXL diffusion conditioned on z."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to .pth checkpoint. Defaults to latest in checkpoints/.",
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to input image.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: inference_output_images/).",
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=1.0,
        help="CFG guidance scale. 1.0 = no CFG (default, deterministic).",
    )
    parser.add_argument(
        "--num_steps", type=int, default=None,
        help="Denoising steps (default from config).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for reproducibility.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = find_latest_checkpoint(CHECKPOINTS_DIR)
        if checkpoint is None:
            logger.error(f"No checkpoints found in {CHECKPOINTS_DIR}")
            sys.exit(1)
        logger.info(f"Auto-selected latest checkpoint: {checkpoint}")

    run_inference(
        checkpoint_path=checkpoint,
        image_path=args.image,
        output_dir=args.output_dir,
        guidance_scale=args.guidance_scale,
        num_steps=args.num_steps,
        seed=args.seed,
    )
