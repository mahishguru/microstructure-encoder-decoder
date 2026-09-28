"""
Parallel test-set reconstruction (DDP).

Loads the trained encoder + decoder + VAE for one FM-DiT variant
(`vitfmdit`, `vitfmdit_768`, `vitfmdit_1024`), iterates over the test
dataset with a DistributedSampler, runs the full encode → flow-match
sample → VAE decode pipeline in batches, and writes:

  eval_outputs/originals/<stem>.png                # written once (any rank)
  eval_outputs/<variant>/reconstructions/<stem>.png
  eval_outputs/<variant>/per_image_recon.csv       # filename,psnr,mse  (rank-merged)

Originals are written into a shared folder keyed by stem so the metrics
script can pair them with reconstructions from any variant.

Launch:
    torchrun --nproc_per_node=4 -m microstructure_ed.eval.run_reconstruction \
        --variant vitfmdit \
        --checkpoint checkpoints/fmdit/checkpoint_epoch_029.pth \
        --batch_size 8 --num_steps 50

If --checkpoint is omitted, the latest checkpoint in the variant's default
directory is used.
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib
import math
import os
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

# ── Repo on path ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(os.environ.get("MSED_ROOT", os.getcwd())).resolve()

from microstructure_ed.config import TEST_DIR  # noqa: E402
from microstructure_ed.encoder_arch_pretrained import Compressor  # noqa: E402


# ── Variant registry ──────────────────────────────────────────────────────────

VARIANT_INFO = {
    "vitfmdit":      {"module": "microstructure_ed.fmdit.decoder_arch_pretrained", "target_dim": 512,      "ckpt_dir": "checkpoints/fmdit_512_v4scratch",
                       "decoder_cls": "FlowMatchingDiTDecoder", "vae_cls": "SD35VAE",
                       "steps_attr": "FMDIT_NUM_INFERENCE_STEPS", "vae_uses_token": True,
                       "mode": "flow"},
    "vitfmdit_768":  {"module": "microstructure_ed.fmdit.decoder_arch_pretrained", "target_dim": 768,  "ckpt_dir": "checkpoints/fmdit_768_v4scratch",
                       "decoder_cls": "FlowMatchingDiTDecoder", "vae_cls": "SD35VAE",
                       "steps_attr": "FMDIT_NUM_INFERENCE_STEPS", "vae_uses_token": True,
                       "mode": "flow"},
    "vitfmdit_1024": {"module": "microstructure_ed.fmdit.decoder_arch_pretrained", "target_dim": 1024, "ckpt_dir": "checkpoints/fmdit_1024_v4scratch",
                       "decoder_cls": "FlowMatchingDiTDecoder", "vae_cls": "SD35VAE",
                       "steps_attr": "FMDIT_NUM_INFERENCE_STEPS", "vae_uses_token": True,
                       "mode": "flow"},
    "vitfmdit_1280": {"module": "microstructure_ed.fmdit.decoder_arch_pretrained", "target_dim": 1280, "ckpt_dir": "checkpoints/fmdit_1280_v4scratch",
                       "decoder_cls": "FlowMatchingDiTDecoder", "vae_cls": "SD35VAE",
                       "steps_attr": "FMDIT_NUM_INFERENCE_STEPS", "vae_uses_token": True,
                       "mode": "flow"},
    "vitdit":        {"module": "microstructure_ed.baselines.vitdit.decoder_arch_pretrained",        "ckpt_dir": "checkpoints/vitdit_final",
                       "decoder_cls": "ViTDiTDecoder", "vae_cls": "SDVAE",
                       "steps_attr": "VITDIT_NUM_INFERENCE_STEPS", "vae_uses_token": False,
                       "mode": "flow"},
    "vitsdxl":       {"module": "microstructure_ed.baselines.vitsdxl.decoder_arch_pretrained",       "ckpt_dir": "checkpoints/vitsdxl_final",
                       "decoder_cls": "SDXLDecoder", "vae_cls": "SDVAE",
                       "steps_attr": "NUM_INFERENCE_STEPS", "vae_uses_token": False,
                       "mode": "sdxl"},
}

# Final compare resolution — originals are 300x300 native; reconstructions are
# saved at this size with NEAREST so crystal-orientation RGB codes are not blended.
FINAL_SIZE = 300


def load_variant_modules(variant: str):
    """Returns (decoder_cls, vae_cls, target_dim, default_num_steps, vae_uses_token, mode, spatial_tokens)."""
    if variant not in VARIANT_INFO:
        raise ValueError(f"Unknown variant '{variant}'. Choose one of {list(VARIANT_INFO)}.")
    info = VARIANT_INFO[variant]
    mod = importlib.import_module(info["module"])
    cfg = importlib.import_module("microstructure_ed.config")
    default_steps = int(getattr(mod, info["steps_attr"], getattr(cfg, info["steps_attr"])))
    # Spatial-token latent: mirror the encoder layout the variant was trained with.
    # spatial_tokens > 0 selects the spatial path (num_queries = spatial_tokens,
    # d_z = target_dim // spatial_tokens); 0 keeps the legacy global path.
    spatial_tokens = (int(getattr(mod, "SPATIAL_NUM_TOKENS", 0))
                      if bool(getattr(mod, "SPATIAL_LATENT", False)) else 0)
    return (
        getattr(mod, info["decoder_cls"]),
        getattr(mod, info["vae_cls"]),
        int(info.get("target_dim", getattr(mod, "TARGET_DIM"))),
        default_steps,
        bool(info.get("vae_uses_token", False)),
        str(info.get("mode", "flow")),
        spatial_tokens,
    )


# ── DDP helpers ───────────────────────────────────────────────────────────────

def setup_ddp() -> Tuple[int, int, torch.device]:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world, torch.device(f"cuda:{local_rank}")


def is_main() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


# ── Dataset ───────────────────────────────────────────────────────────────────

class TestImageDataset(Dataset):
    VALID = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(self, folder: str):
        folder_p = Path(folder)
        self.files: List[str] = sorted(
            f.name for f in folder_p.iterdir() if f.suffix.lower() in self.VALID
        )
        self.folder = folder
        # Encoder is trained at 512x512; use NEAREST so orientation RGB codes
        # in the input are not blended.
        self.tf = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        name = self.files[idx]
        img = Image.open(os.path.join(self.folder, name)).convert("RGB")
        return self.tf(img), name


# ── IO ────────────────────────────────────────────────────────────────────────

def tensor_to_pil_uint8(t: torch.Tensor) -> Image.Image:
    """t: (3, H, W) in [-1, 1]."""
    arr = (t.float() / 2 + 0.5).clamp(0, 1)
    arr = (arr.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    return Image.fromarray(arr)


def compute_psnr_mse(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    """a, b: (3, H, W) in [-1, 1]."""
    diff = (a.float() - b.float())
    mse = float((diff ** 2).mean().item())
    if mse <= 0:
        psnr = float("inf")
    else:
        psnr = 20.0 * math.log10(2.0 / math.sqrt(mse))
    return psnr, mse


def find_latest_ckpt(ckpt_dir: str) -> str:
    cands = sorted(Path(ckpt_dir).glob("checkpoint_epoch_*.pth"))
    if not cands:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    # prefer *_best.pth if present
    best = [c for c in cands if c.stem.endswith("_best")]
    return str((best or cands)[-1])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=list(VARIANT_INFO))
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pth. Defaults to latest in the variant's checkpoint dir.")
    parser.add_argument("--test_dir", type=str, default=None,
                        help="Override test image directory (default: config.TEST_DIR).")
    parser.add_argument("--output_root", type=str,
                        default=str(REPO_ROOT / "eval_outputs"),
                        help="Root directory for originals/reconstructions/CSVs.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=None,
                        help="Flow-matching ODE steps (default: variant's FMDIT_NUM_INFERENCE_STEPS).")
    parser.add_argument("--noise_temp", type=float, default=1.0)
    parser.add_argument("--shift", type=float, default=None)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on number of images (for smoke tests).")
    parser.add_argument("--shard_idx", type=int, default=0,
                        help="This process's shard index (0-based).")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total independent shards (for multi-process-per-GPU runs).")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip reconstructions whose PNG already exists on disk.")
    args = parser.parse_args()

    rank, world, device = setup_ddp()

    decoder_cls, vae_cls, target_dim, default_steps, vae_uses_token, mode, spatial_tokens = load_variant_modules(args.variant)
    num_steps = args.num_steps or default_steps
    test_dir = args.test_dir or TEST_DIR
    ckpt_path = args.checkpoint or find_latest_ckpt(VARIANT_INFO[args.variant]["ckpt_dir"])

    out_root = Path(args.output_root)
    orig_dir = out_root / "originals"
    recon_dir = out_root / args.variant / "reconstructions"
    csv_dir = out_root / args.variant
    if is_main():
        orig_dir.mkdir(parents=True, exist_ok=True)
        recon_dir.mkdir(parents=True, exist_ok=True)
        csv_dir.mkdir(parents=True, exist_ok=True)
        print(f"[eval/{args.variant}] world_size={world}  device={device}", flush=True)
        print(f"[eval/{args.variant}] ckpt={ckpt_path}  D={target_dim}  steps={num_steps}", flush=True)
        print(f"[eval/{args.variant}] test_dir={test_dir}", flush=True)
        print(f"[eval/{args.variant}] originals → {orig_dir}", flush=True)
        print(f"[eval/{args.variant}] recons    → {recon_dir}", flush=True)
    dist.barrier()

    # ── Build models ─────────────────────────────────────────────────────────
    encoder = Compressor(use_gradient_checkpointing=False,
                         trainable_blocks=0,
                         target_dim=target_dim,
                         spatial_tokens=spatial_tokens).to(device)
    decoder = (decoder_cls(target_dim=target_dim) if "target_dim" in VARIANT_INFO[args.variant]
               else decoder_cls()).to(device)
    if vae_uses_token:
        vae = vae_cls(token=os.environ.get("HF_TOKEN")).to(device)
    else:
        vae = vae_cls().to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if mode == "sdxl":
        # SDXL trainer saves only the trainable bits of the decoder.
        encoder.load_state_dict(ckpt["compressor_state_dict"])
        decoder.token_proj.load_state_dict(ckpt["token_proj_state_dict"])
        decoder.pooled_proj.load_state_dict(ckpt["pooled_proj_state_dict"])
        decoder.null_token.data.copy_(ckpt["null_token"])
    else:
        # Stripped FM-DiT release checkpoints omit the frozen SD3.5 backbone
        # (loaded from the Hub); everything else must match exactly.
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"],
                                strict=not ckpt.get("stripped_frozen_backbone", False))
    del ckpt
    gc.collect()
    torch.cuda.empty_cache()

    encoder.eval(); decoder.eval()
    if is_main():
        print(f"[eval/{args.variant}] models ready", flush=True)

    # ── Dataset / loader ─────────────────────────────────────────────────────
    dataset = TestImageDataset(test_dir)
    if args.limit is not None:
        dataset.files = dataset.files[: args.limit]
    if args.num_shards > 1:
        dataset.files = dataset.files[args.shard_idx::args.num_shards]
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank,
                                 shuffle=False, drop_last=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0)

    if is_main():
        print(f"[eval/{args.variant}] {len(dataset)} images, "
              f"{len(loader)} batches per rank (B={args.batch_size})", flush=True)

    # ── Per-rank CSV ─────────────────────────────────────────────────────────
    rank_csv = csv_dir / f"per_image_recon.rank{rank:02d}.s{args.shard_idx}.csv"
    fp = open(rank_csv, "w", newline="")
    writer = csv.writer(fp)
    writer.writerow(["filename", "psnr", "mse"])

    autocast = torch.amp.autocast("cuda", dtype=torch.bfloat16)
    n_done = 0
    n_skipped = 0

    for batch_idx, (imgs, names) in enumerate(loader):
        # Optional skip if all reconstructions exist
        if args.skip_existing:
            keep = []
            for i, name in enumerate(names):
                stem = Path(name).stem
                if not (recon_dir / f"{stem}.png").exists():
                    keep.append(i)
            if not keep:
                n_skipped += len(names)
                continue
            if len(keep) != len(names):
                imgs = imgs[keep]
                names = [names[i] for i in keep]

        imgs = imgs.to(device, non_blocking=True)

        with torch.no_grad(), autocast:
            z = encoder(imgs)                                  # (B, D)
            if mode == "sdxl":
                # SDXL decoder owns its full denoise+VAE pipeline.
                recon = decoder.generate(
                    z, vae=vae, num_inference_steps=num_steps,
                    guidance_scale=1.0, return_latent=False,
                )                                              # (B, 3, 512, 512) in [-1, 1]
            else:
                latent = decoder.sample(z, num_steps=num_steps,
                                        noise_temp=args.noise_temp,
                                        shift=args.shift,
                                        guidance_scale=args.guidance)
                recon = vae.decode(latent)                     # (B, 3, 512, 512) in [-1, 1]

        recon = recon.float().clamp(-1.0, 1.0).cpu()

        for i, name in enumerate(names):
            stem = Path(name).stem

            # Save reconstruction at FINAL_SIZE NEAREST (preserve orientation RGB codes).
            recon_pil = tensor_to_pil_uint8(recon[i])
            if recon_pil.size != (FINAL_SIZE, FINAL_SIZE):
                recon_pil = recon_pil.resize((FINAL_SIZE, FINAL_SIZE), Image.NEAREST)
            recon_pil.save(recon_dir / f"{stem}.png")

            # Save the original (each rank saves its own shard) at 300x300 NEAREST copy.
            op = orig_dir / f"{stem}.png"
            if not op.exists():
                src_pil = Image.open(os.path.join(test_dir, name)).convert("RGB")
                if src_pil.size != (FINAL_SIZE, FINAL_SIZE):
                    src_pil = src_pil.resize((FINAL_SIZE, FINAL_SIZE), Image.NEAREST)
                src_pil.save(op)

            # PSNR/MSE at FINAL_SIZE — compare what the user actually sees.
            src_pil = Image.open(os.path.join(test_dir, name)).convert("RGB")
            if src_pil.size != (FINAL_SIZE, FINAL_SIZE):
                src_pil = src_pil.resize((FINAL_SIZE, FINAL_SIZE), Image.NEAREST)
            src_t = transforms.functional.to_tensor(src_pil) * 2.0 - 1.0
            rec_t = transforms.functional.to_tensor(recon_pil) * 2.0 - 1.0
            psnr, mse = compute_psnr_mse(src_t, rec_t)
            writer.writerow([name, f"{psnr:.6f}", f"{mse:.8f}"])

            n_done += 1

        if is_main() and (batch_idx + 1) % 10 == 0:
            print(f"[eval/{args.variant}] rank0 batch {batch_idx + 1}/{len(loader)} "
                  f"({n_done} written, {n_skipped} skipped so far)", flush=True)

    fp.close()
    print(f"[eval/{args.variant}] rank {rank}: wrote {n_done} (skipped {n_skipped}) → {rank_csv.name}", flush=True)

    # ── Merge per-rank CSVs on rank 0 ────────────────────────────────────────
    dist.barrier()
    if is_main():
        rows = []
        for r in range(world):
            rcsv = csv_dir / f"per_image_recon.rank{r:02d}.s{args.shard_idx}.csv"
            if not rcsv.exists():
                continue
            with open(rcsv, newline="") as f:
                rdr = csv.reader(f)
                next(rdr, None)  # header
                rows.extend(rdr)
        rows.sort(key=lambda r: r[0])
        merged = csv_dir / (f"per_image_recon.s{args.shard_idx}.csv"
                            if args.num_shards > 1 else "per_image_recon.csv")
        with open(merged, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "psnr", "mse"])
            w.writerows(rows)
        # cleanup per-rank shards
        for r in range(world):
            (csv_dir / f"per_image_recon.rank{r:02d}.s{args.shard_idx}.csv").unlink(missing_ok=True)
        print(f"[eval/{args.variant}] merged {len(rows)} rows → {merged}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
