"""
trainer_vitsdxl.py  —  DDP training script for the ViTSDXL (SDXL UNet) decoder.

Launch with:
    python -m torch.distributed.run --nproc_per_node=4 -m microstructure_ed.baselines.vitsdxl.trainer
"""
import os, sys, re, csv, gc, random, logging
from pathlib import Path
from datetime import datetime

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from microstructure_ed.config import (
    INPUT_IMAGES_DIR, CHECKPOINTS_DIR, LOGS_DIR, EPOCH_SAMPLES_DIR,
    LOSS_LOG_FILE,
    NUM_EPOCHS, BATCH_SIZE, NUM_WORKERS, MAX_IMAGES,
    COMPRESSOR_LR, WARMUP_STEPS, ACCUMULATION_STEPS,
    P_DROP, USE_AMP, MAX_GRAD_NORM,
    SAVE_EVERY_EPOCH, RESUME_FROM,
    LAMBDA_DIFF_START, LAMBDA_DIFF_END,
    LAMBDA_CONTRASTIVE_START, LAMBDA_CONTRASTIVE_END,
    LAMBDA_RECON_START, LAMBDA_RECON_END, WARMUP_EPOCHS_LOSS,
    LAMBDA_VICREG_START, LAMBDA_VICREG_END,
    LAMBDA_LAB_START, LAMBDA_LAB_END,
    LAMBDA_FREQ_START, LAMBDA_FREQ_END,
    CONTRASTIVE_TEMPERATURE, PIXEL_NOISE_SCALE,
    VICREG_VAR_WEIGHT, VICREG_COV_WEIGHT, VICREG_VAR_TARGET,
    PIXEL_LOSS_ALPHA_T_THRESHOLD,
)
from microstructure_ed.encoder_arch_pretrained import Compressor
from microstructure_ed.baselines.vitsdxl.decoder_arch_pretrained import (
    SDXLDecoder, SDVAE,
    TARGET_DIM as VARIANT_TARGET_DIM,  # 512
)
from microstructure_ed.baselines.vitsdxl.embedding_utils import (
    ContrastiveLoss, VICRegLoss, LabColorLoss, FrequencyLoss,
    EmbeddingAnalyzer, LossWeightScheduler,
)


# ── DDP helpers ───────────────────────────────────────────────────────────────

def setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINTS_DIR = "./checkpoints/vitsdxl"

_RUN_TAG              = datetime.now().strftime('%Y%m%d_%H%M%S')
LOGS_DIR_RUN          = f"./logs/vitsdxl/run_{_RUN_TAG}"
EPOCH_SAMPLES_DIR_RUN = f"./samples/vitsdxl/run_{_RUN_TAG}"
LOSS_LOG_FILE_RUN     = os.path.join(LOGS_DIR_RUN, "loss_log.csv")

logger = logging.getLogger(__name__)


def setup_logging():
    if not is_main():
        logging.basicConfig(level=logging.WARNING)
        return
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR_RUN, exist_ok=True)
    os.makedirs(EPOCH_SAMPLES_DIR_RUN, exist_ok=True)
    log_file = os.path.join(LOGS_DIR_RUN, f"train_{_RUN_TAG}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )


# ── Dataset ───────────────────────────────────────────────────────────────────

class ImageDataset(Dataset):
    def __init__(self, folder_path: str, max_images: int = None):
        self.folder_path = folder_path
        exts = {".png", ".jpg", ".jpeg"}
        files = sorted(f for f in os.listdir(folder_path) if Path(f).suffix.lower() in exts)
        if max_images:
            files = files[:max_images]
        self.image_files = files
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        if is_main():
            logger.info(f"Dataset: {len(self.image_files)} images from {folder_path}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = os.path.join(self.folder_path, self.image_files[idx])
        img = Image.open(path).convert("RGB")
        return self.transform(img)


# ── Utilities ─────────────────────────────────────────────────────────────────

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    t = (tensor / 2 + 0.5).clamp(0, 1)
    return Image.fromarray(
        (t.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255).astype("uint8")
    )

def compute_psnr(img_a: torch.Tensor, img_b: torch.Tensor) -> float:
    mse = torch.mean((img_a.float() - img_b.float()) ** 2).item()
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(torch.tensor(2.0 / (mse ** 0.5))).item()

def _unwrap(model):
    return model.module if isinstance(model, DDP) else model

def generate_epoch_samples(epoch_num, compressor, decoder, vae, device, full_dataset, train_dataset, n_samples=10):
    if not is_main():
        return 0.0
    psnr_values = []
    epoch_dir = os.path.join(EPOCH_SAMPLES_DIR_RUN, f"epoch_{epoch_num:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    comp_raw = _unwrap(compressor)
    comp_raw.eval()
    decoder.token_proj.eval()
    decoder.pooled_proj.eval()

    vae.vae.disable_tiling()
    vae.vae.enable_slicing()
    decoder.unet.to(device)

    indices = random.sample(list(train_dataset.indices), min(n_samples, len(train_dataset)))

    orig_pils  = []
    recon_pils = []
    names      = []

    for k, train_idx in enumerate(indices):
        try:
            sample_path  = os.path.join(full_dataset.folder_path, full_dataset.image_files[train_idx])
            orig_pil     = Image.open(sample_path).convert("RGB").resize((512, 512))
            sample_tensor = full_dataset.transform(orig_pil).unsqueeze(0).to(device)
            sample_name  = full_dataset.image_files[train_idx]

            with torch.no_grad():
                with (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                      if USE_AMP and device.type == "cuda"
                      else torch.amp.autocast("cpu", enabled=False)):
                    emb = comp_raw(sample_tensor)

                img_recon = decoder.generate(
                    emb, vae=vae,
                    num_inference_steps=20,
                    guidance_scale=3.0, seed=42 + k, return_latent=False,
                    use_sdxl_null=(epoch_num == 0),
                )

            psnr = compute_psnr(sample_tensor, img_recon)
            psnr_values.append(psnr)
            recon_pil = tensor_to_pil(img_recon.cpu())

            orig_pils.append(orig_pil)
            recon_pils.append(recon_pil)
            names.append(sample_name)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(orig_pil); axes[0].set_title(f"Original: {sample_name}", fontsize=9); axes[0].axis("off")
            axes[1].imshow(recon_pil); axes[1].set_title(f"ViTSDXL | PSNR: {psnr:.2f} dB", fontsize=9); axes[1].axis("off")
            plt.suptitle(f"Epoch {epoch_num}", fontsize=12, fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(epoch_dir, f"sample_{k+1:02d}_{Path(sample_name).stem}.png"), dpi=120, bbox_inches="tight")
            plt.close()

        except Exception as e:
            logger.warning(f"Epoch {epoch_num} sample {k+1}: {e}")

    if orig_pils:
        n = len(orig_pils)
        fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
        if n == 1:
            axes = axes.reshape(2, 1)
        for j in range(n):
            axes[0, j].imshow(orig_pils[j]); axes[0, j].set_title(f"Orig: {Path(names[j]).stem}", fontsize=7); axes[0, j].axis("off")
            axes[1, j].imshow(recon_pils[j]); axes[1, j].set_title(f"PSNR: {psnr_values[j]:.2f}", fontsize=7); axes[1, j].axis("off")
        plt.suptitle(f"Epoch {epoch_num} -- Mean PSNR: {sum(psnr_values)/len(psnr_values):.2f} dB", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(epoch_dir, f"grid_epoch_{epoch_num:03d}.png"), dpi=150)
        plt.close()
        logger.info(f"Epoch {epoch_num} samples saved -> {epoch_dir}/ | Mean PSNR: {sum(psnr_values)/len(psnr_values):.2f} dB")

    decoder.unet.to(device)
    torch.cuda.empty_cache()

    return sum(psnr_values) / max(1, len(psnr_values)) if psnr_values else 0.0


def cleanup_old_checkpoints(checkpoints_dir: str, keep_last_n: int = 3):
    pattern = re.compile(r"checkpoint_epoch_(\d+)")
    ckpts = sorted(
        [p for p in Path(checkpoints_dir).glob("checkpoint_epoch_*.pth")
         if pattern.search(p.name)],
        key=lambda p: int(pattern.search(p.name).group(1)),
    )
    for old in ckpts[:-keep_last_n]:
        old.unlink()
        logger.info(f"Removed old checkpoint: {old.name}")


# ── CSV Logger ────────────────────────────────────────────────────────────────

def init_csv_logger():
    if not is_main():
        return
    os.makedirs(os.path.dirname(LOSS_LOG_FILE_RUN), exist_ok=True)
    if not os.path.exists(LOSS_LOG_FILE_RUN):
        with open(LOSS_LOG_FILE_RUN, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_diff", "train_recon", "train_contrastive",
                 "train_vicreg", "train_lab", "train_freq", "train_total",
                 "val_diff", "val_total", "psnr_sample",
                 "emb_norm_mean", "emb_cosine_sim", "emb_effective_dim"]
            )


def log_to_csv(epoch, train_diff, train_recon, train_contrastive,
               train_vicreg, train_lab, train_freq, train_total,
               val_diff, val_total, psnr, emb_metrics):
    if not is_main():
        return
    os.makedirs(os.path.dirname(LOSS_LOG_FILE_RUN), exist_ok=True)
    with open(LOSS_LOG_FILE_RUN, "a", newline="") as f:
        csv.writer(f).writerow([
            epoch,
            f"{train_diff:.6f}",
            f"{train_recon:.6f}",
            f"{train_contrastive:.6f}",
            f"{train_vicreg:.6f}",
            f"{train_lab:.6f}",
            f"{train_freq:.6f}",
            f"{train_total:.6f}",
            f"{val_diff:.6f}",
            f"{val_total:.6f}",
            f"{psnr:.4f}",
            f"{emb_metrics.get('norm_mean', 0):.4f}",
            f"{emb_metrics.get('cosine_sim_mean', 0):.4f}",
            f"{emb_metrics.get('effective_dimensionality', 0):.1f}",
        ])


def get_trainable_params(module: nn.Module, lr: float) -> dict:
    params = [p for p in module.parameters() if p.requires_grad and p.is_leaf]
    return {"params": params, "lr": lr}


def log_param_counts(compressor, decoder):
    def count(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    if not is_main():
        return
    logger.info("Trainable parameter budget:")
    logger.info(f"  Compressor (ViT-H/14 blocks 0-15 + Pooler)                  : {count(compressor):>12,}")
    logger.info(f"  LatentToTokens (AdaLN-Zero self-attention)                    : {count(decoder.token_proj):>12,}")
    logger.info(f"  pooled_proj                                                   : {count(decoder.pooled_proj):>12,}")
    logger.info(f"  null_token                                                    : {decoder.null_token.numel():>12,}")
    total = (count(compressor) + count(decoder.token_proj)
             + count(decoder.pooled_proj) + decoder.null_token.numel())
    logger.info(f"  -- TOTAL ---------------------------------------------------------------- {total:>12,}")


def create_train_val_split(dataset, val_split=0.1):
    if val_split <= 0 or val_split >= 1:
        return dataset, None
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_split, random_state=42, shuffle=True
    )
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    if is_main():
        logger.info(f"Split: {len(train_dataset)} train, {len(val_dataset)} val")
    return train_dataset, val_dataset


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(compressor, decoder, val_loader, vae, device, embedding_analyzer):
    comp_raw = _unwrap(compressor)
    comp_raw.eval()
    decoder.token_proj.eval()
    decoder.pooled_proj.eval()

    total_diff_loss = 0.0
    all_embeddings = []

    for batch in val_loader:
        images = batch.to(device, non_blocking=True)
        B = images.size(0)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16) if USE_AMP and device.type == "cuda" else torch.amp.autocast("cpu", enabled=False):
            embedding = comp_raw(images)
            all_embeddings.append(embedding.float())

            latent_target = vae.encode(images).detach()

            conditioning = decoder.get_conditioning(embedding)
            added_kwargs = decoder.get_added_cond_kwargs(B, embedding, device=device)

            timesteps = torch.randint(
                0, decoder.scheduler.config.num_train_timesteps,
                (B,), device=device, dtype=torch.long,
            )
            noise = torch.randn_like(latent_target, device=device)
            noisy_latents = decoder.scheduler.add_noise(latent_target, noise, timesteps)

            noise_pred = decoder.unet(
                noisy_latents.to(torch.bfloat16),
                timesteps,
                encoder_hidden_states=conditioning.to(torch.bfloat16),
                added_cond_kwargs={k: v.to(torch.bfloat16) for k, v in added_kwargs.items()},
            ).sample

            loss_diff = nn.functional.mse_loss(noise_pred.float(), noise.float())
            total_diff_loss += loss_diff.item()

    avg_diff_loss = total_diff_loss / max(1, len(val_loader))
    all_embeddings = torch.cat(all_embeddings, dim=0)
    emb_metrics = embedding_analyzer.compute_metrics(all_embeddings)

    return avg_diff_loss, emb_metrics


# ── Training ──────────────────────────────────────────────────────────────────

def train():

    # ── DDP init ──────────────────────────────────────────────────────────────
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()

    setup_logging()

    if is_main():
        logger.info(f"Using device: {device}")
        logger.info(f"GPU: {torch.cuda.get_device_name(local_rank)}")
        logger.info(f"DDP world_size={world_size}")

    torch.backends.cudnn.benchmark = True

    SDXL_NUM_EPOCHS = 30
    UNFREEZE_EPOCH  = 8   # Phase 2: unfreeze last 16 ViT blocks at this epoch

    init_csv_logger()

    # ── Dataset & DataLoader ──────────────────────────────────────────────────
    full_dataset = ImageDataset(INPUT_IMAGES_DIR, max_images=MAX_IMAGES)
    if len(full_dataset) == 0:
        if is_main():
            logger.error(f"No images found in {INPUT_IMAGES_DIR}")
        cleanup_ddp()
        sys.exit(1)

    train_dataset, val_dataset = create_train_val_split(full_dataset, val_split=0.1)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                       rank=dist.get_rank(), shuffle=True, drop_last=True)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"), drop_last=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"), drop_last=False,
    ) if val_dataset else None

    if is_main():
        logger.info(f"Batches per epoch: train={len(train_loader)}, val={len(val_loader) if val_loader else 0}")

    # ── Models ────────────────────────────────────────────────────────────────
    if is_main():
        logger.info("Loading models...")

    vae = SDVAE(freeze=True).to(device)

    # Phase 1: trainable_blocks=0 -> freeze ALL ViT blocks.
    # Only attention_pooler + target_proj train (the bottleneck head).
    # Phase 2 (epoch >= UNFREEZE_EPOCH): unfreeze last 16 blocks.
    compressor = Compressor(use_gradient_checkpointing=True, trainable_blocks=0,
                            target_dim=VARIANT_TARGET_DIM).to(device)

    decoder = SDXLDecoder(p_drop=P_DROP)
    decoder.token_proj  = decoder.token_proj.to(device)
    decoder.pooled_proj = decoder.pooled_proj.to(device)
    decoder.null_token  = nn.Parameter(decoder.null_token.data.to(device))
    decoder.unet        = decoder.unet.to(device)
    decoder.unet.enable_gradient_checkpointing()

    log_param_counts(compressor, decoder)

    # ── Wrap compressor in DDP (decoder parts are wrapped individually) ───────
    compressor_ddp = DDP(compressor, device_ids=[local_rank], find_unused_parameters=True)

    # token_proj and pooled_proj are small trainable modules — wrap them too
    decoder.token_proj  = DDP(decoder.token_proj, device_ids=[local_rank], find_unused_parameters=True)
    decoder.pooled_proj = DDP(decoder.pooled_proj, device_ids=[local_rank], find_unused_parameters=True)
    # UNet is frozen, null_token is a single Parameter — no DDP needed

    # ── Loss Functions ────────────────────────────────────────────────────────
    contrastive_loss_fn = ContrastiveLoss(temperature=CONTRASTIVE_TEMPERATURE).to(device)
    vicreg_loss_fn = VICRegLoss(
        var_weight=VICREG_VAR_WEIGHT, cov_weight=VICREG_COV_WEIGHT, var_target=VICREG_VAR_TARGET,
    ).to(device)
    lab_loss_fn = LabColorLoss().to(device)
    freq_loss_fn = FrequencyLoss().to(device)

    embedding_analyzer = EmbeddingAnalyzer()
    loss_scheduler = LossWeightScheduler(
        num_epochs=SDXL_NUM_EPOCHS,
        lambda_diff_start=LAMBDA_DIFF_START, lambda_diff_end=LAMBDA_DIFF_END,
        lambda_contrastive_start=LAMBDA_CONTRASTIVE_START, lambda_contrastive_end=LAMBDA_CONTRASTIVE_END,
        lambda_recon_start=LAMBDA_RECON_START, lambda_recon_end=LAMBDA_RECON_END,
        lambda_vicreg_start=LAMBDA_VICREG_START, lambda_vicreg_end=LAMBDA_VICREG_END,
        lambda_lab_start=LAMBDA_LAB_START, lambda_lab_end=LAMBDA_LAB_END,
        lambda_freq_start=LAMBDA_FREQ_START, lambda_freq_end=LAMBDA_FREQ_END,
        warmup_epochs=WARMUP_EPOCHS_LOSS,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Uniform LR across all groups — no √N scaling. This matches the stable
    # vitfmdit recipe and the peer's stable single-GPU SDXL recipe.
    DDP_LR = COMPRESSOR_LR

    def _build_optimizer():
        """(Re)build optimizer over current trainable params. Used to swap in
        the newly-unfrozen ViT blocks at Phase 2."""
        groups = [
            get_trainable_params(_unwrap(compressor_ddp),      DDP_LR),
            get_trainable_params(_unwrap(decoder.token_proj),  DDP_LR),
            get_trainable_params(_unwrap(decoder.pooled_proj), DDP_LR),
            {"params": [decoder.null_token], "lr": DDP_LR},
        ]
        opt = optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.01,
                          eps=1e-08, fused=True)
        for g in opt.param_groups:
            g["base_lr"] = g["lr"]
        return opt

    optimizer = _build_optimizer()

    # ── LR Scheduler ──────────────────────────────────────────────────────────
    # NB: `lr_scheduler.step()` is only called once per ACCUMULATION_STEPS
    # micro-batches (inside update_lr at the optimizer-step boundary), so
    # T_max must be in OPTIMIZER STEPS, not micro-batches.
    opt_steps_per_epoch = max(1, len(train_loader) // ACCUMULATION_STEPS)
    total_opt_steps     = opt_steps_per_epoch * SDXL_NUM_EPOCHS
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_opt_steps - WARMUP_STEPS), eta_min=1e-7
    )
    curr_step = 0

    def update_lr():
        nonlocal curr_step
        if curr_step < WARMUP_STEPS:
            lr_factor = curr_step / max(1, WARMUP_STEPS)
            for g in optimizer.param_groups:
                g["lr"] = g.get("base_lr", DDP_LR) * lr_factor
        else:
            lr_scheduler.step()
        curr_step += 1

    for g in optimizer.param_groups:
        g["base_lr"] = g["lr"]

    alphas_cumprod = decoder.scheduler.alphas_cumprod.to(device)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float('inf')

    if RESUME_FROM and os.path.exists(RESUME_FROM):
        if is_main():
            logger.info(f"Resuming from checkpoint: {RESUME_FROM}")
        ckpt = torch.load(RESUME_FROM, map_location=device)
        compressor.load_state_dict(ckpt["compressor_state_dict"])
        _unwrap(decoder.token_proj).load_state_dict(ckpt["token_proj_state_dict"])
        if "pooled_proj_state_dict" in ckpt:
            _unwrap(decoder.pooled_proj).load_state_dict(ckpt["pooled_proj_state_dict"])
        decoder.null_token.data.copy_(ckpt["null_token"])

        # If resuming into Phase 2, unfreeze and rebuild optimizer BEFORE loading
        # its state so param shapes match the checkpointed optimizer state.
        resume_next_epoch = ckpt["epoch"] + 1
        if resume_next_epoch >= UNFREEZE_EPOCH:
            _unwrap(compressor_ddp).unfreeze_vit_blocks(last_n=16)
            optimizer = _build_optimizer()
            opt_steps_per_epoch = max(1, len(train_loader) // ACCUMULATION_STEPS)
            remaining_opt_steps = opt_steps_per_epoch * (SDXL_NUM_EPOCHS - UNFREEZE_EPOCH)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, remaining_opt_steps), eta_min=1e-7,
            )
            if is_main():
                logger.info(f"Resume into Phase 2 - unfroze last 16 ViT blocks before loading optimizer state")

        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "lr_scheduler_state_dict" in ckpt:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        curr_step = ckpt.get("curr_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float('inf'))
        if is_main():
            logger.info(f"Resumed from epoch {ckpt['epoch'] + 1}, loss {ckpt['loss']:.4f}")
        del ckpt
        gc.collect()
        torch.cuda.empty_cache()
    elif RESUME_FROM:
        if is_main():
            logger.warning(f"Checkpoint not found: {RESUME_FROM} -- starting from scratch.")

    if is_main():
        logger.info(
            f"TRAINING  |  epochs {start_epoch + 1} -> {SDXL_NUM_EPOCHS}  |  "
            f"batch_size {BATCH_SIZE}x{world_size}GPUs  |  accum_steps {ACCUMULATION_STEPS}"
        )

    # ── DDP-safe skip helper ──────────────────────────────────────────────────
    # Any rank that wants to skip a microbatch must coordinate with the others,
    # otherwise DDP's gradient all-reduce will hang waiting for the missing rank.
    def _any_rank_skips(local_skip: bool) -> bool:
        flag = torch.tensor([1 if local_skip else 0], device=device, dtype=torch.int32)
        if dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, SDXL_NUM_EPOCHS):
        train_sampler.set_epoch(epoch)

        # Phase 2: unfreeze last 16 ViT blocks and rebuild optimizer + scheduler
        if epoch == UNFREEZE_EPOCH:
            _unwrap(compressor_ddp).unfreeze_vit_blocks(last_n=16)
            optimizer = _build_optimizer()
            opt_steps_per_epoch = max(1, len(train_loader) // ACCUMULATION_STEPS)
            remaining_opt_steps = opt_steps_per_epoch * (SDXL_NUM_EPOCHS - UNFREEZE_EPOCH)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, remaining_opt_steps), eta_min=1e-7,
            )
            if is_main():
                n_trainable = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
                logger.info(f"Phase 2 (epoch {epoch + 1}) - unfroze last 16 ViT blocks | "
                            f"trainable params now {n_trainable:,}")

        loss_weights = loss_scheduler.get_weights(epoch)
        if is_main():
            loss_scheduler.log_weights(epoch, logger)

        compressor_ddp.train()
        decoder.token_proj.train()
        decoder.pooled_proj.train()

        total_diff_loss = 0.0
        total_recon_loss = 0.0
        total_contrastive_loss = 0.0
        total_vicreg_loss = 0.0
        total_lab_loss = 0.0
        total_freq_loss = 0.0
        total_loss = 0.0

        valid_batches = 0
        batch_count   = 0

        num_batches = len(train_loader)
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            images = batch.to(device, non_blocking=True)

            local_skip = bool(torch.isnan(images).any() or torch.isinf(images).any())
            if _any_rank_skips(local_skip):
                if is_main() and local_skip:
                    logger.warning(f"Skipping corrupt input batch at idx {batch_idx}")
                continue

            B = images.size(0)
            try:
                autocast_ctx = (
                    torch.amp.autocast("cuda", dtype=torch.bfloat16)
                    if USE_AMP and device.type == "cuda"
                    else torch.amp.autocast("cpu", enabled=False)
                )

                with autocast_ctx:
                    embedding = compressor_ddp(images)

                    local_skip = bool(torch.isnan(embedding).any())
                    if _any_rank_skips(local_skip):
                        if is_main() and local_skip:
                            logger.warning(f"NaN in embedding at batch {batch_idx}, skipping")
                        optimizer.zero_grad()
                        torch.cuda.empty_cache()
                        continue

                    loss_contrastive = torch.tensor(0.0, device=device)
                    if loss_weights['lambda_contrastive'] > 0:
                        pixel_noise = torch.randn_like(images, device=device) * PIXEL_NOISE_SCALE
                        images_noised = (images + pixel_noise).clamp(-1.0, 1.0)
                        embedding_aug = compressor_ddp(images_noised)
                        embeddings_both = torch.cat([embedding.detach(), embedding_aug], dim=0).float()
                        loss_contrastive = contrastive_loss_fn(embeddings_both)

                    embedding_cf = decoder.apply_cfg_dropout(embedding)

                    conditioning = decoder.get_conditioning(embedding_cf)
                    added_kwargs = decoder.get_added_cond_kwargs(B, embedding_cf, device=device)

                    latent_target = vae.encode(images).detach()

                    timesteps = torch.randint(
                        0, decoder.scheduler.config.num_train_timesteps,
                        (B,), device=device, dtype=torch.long,
                    )
                    noise_diffusion = torch.randn_like(latent_target, device=device)
                    noisy_latents = decoder.scheduler.add_noise(latent_target, noise_diffusion, timesteps)

                    noise_pred = decoder.unet(
                        noisy_latents.to(torch.bfloat16),
                        timesteps,
                        encoder_hidden_states=conditioning.to(torch.bfloat16),
                        added_cond_kwargs={k: v.to(torch.bfloat16) for k, v in added_kwargs.items()},
                    ).sample

                    loss_diff = nn.functional.mse_loss(noise_pred.float(), noise_diffusion.float())

                    alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)

                    need_z0 = (loss_weights['lambda_recon'] > 0 or
                               loss_weights['lambda_lab'] > 0 or
                               loss_weights['lambda_freq'] > 0)
                    z0_pred = None
                    if need_z0:
                        sqrt_alpha_t = torch.sqrt(alpha_t).clamp(min=0.1)
                        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
                        z0_pred = (
                            noisy_latents.float() - sqrt_one_minus_alpha_t * noise_pred.float()
                        ) / sqrt_alpha_t

                    loss_recon = torch.tensor(0.0, device=device)
                    if loss_weights['lambda_recon'] > 0 and z0_pred is not None:
                        alpha_t_1d = alpha_t.view(B)
                        recon_mask = alpha_t_1d > PIXEL_LOSS_ALPHA_T_THRESHOLD
                        if recon_mask.any():
                            idx_r = recon_mask.nonzero(as_tuple=False).view(-1)
                            loss_recon = nn.functional.mse_loss(z0_pred[idx_r], latent_target.float()[idx_r])

                    loss_vicreg = torch.tensor(0.0, device=device)
                    if loss_weights['lambda_vicreg'] > 0:
                        loss_vicreg = vicreg_loss_fn(embedding.float())

                    loss_lab = torch.tensor(0.0, device=device)
                    loss_freq = torch.tensor(0.0, device=device)

                    if (loss_weights['lambda_lab'] > 0 or loss_weights['lambda_freq'] > 0) and z0_pred is not None:
                        alpha_t_flat = alpha_t.squeeze()
                        if alpha_t_flat.dim() == 0:
                            alpha_t_flat = alpha_t_flat.unsqueeze(0)
                        mask = alpha_t_flat > PIXEL_LOSS_ALPHA_T_THRESHOLD

                        if mask.any():
                            idx_sel = mask.nonzero(as_tuple=False).squeeze(1)[:2]
                            sf = vae.vae.config.scaling_factor

                            with torch.amp.autocast('cuda', enabled=False):
                                # The VAE is loaded in bfloat16 (see SDVAE). When autocast
                                # is disabled, the inputs default to float32 and conv2d
                                # rejects mixed dtypes -> cast inputs to the VAE's dtype.
                                vae_dtype = vae.vae.dtype
                                def _vae_decode(z):
                                    return vae.vae.decode(z).sample

                                z_pred_in = (z0_pred[idx_sel].float() / sf).to(vae_dtype)
                                img_pred = grad_checkpoint(
                                    _vae_decode, z_pred_in, use_reentrant=False,
                                ).float().clamp(-1, 1)

                                with torch.no_grad():
                                    z_tgt_in = (latent_target[idx_sel].float() / sf).to(vae_dtype)
                                    img_tgt = vae.vae.decode(z_tgt_in).sample.float().clamp(-1, 1)

                            if loss_weights['lambda_lab'] > 0:
                                loss_lab = lab_loss_fn(img_pred, img_tgt)
                            if loss_weights['lambda_freq'] > 0:
                                loss_freq = freq_loss_fn(img_pred, img_tgt)

                    loss = (
                        loss_weights['lambda_diff'] * loss_diff +
                        loss_weights['lambda_recon'] * loss_recon +
                        loss_weights['lambda_contrastive'] * loss_contrastive +
                        loss_weights['lambda_vicreg'] * loss_vicreg +
                        loss_weights['lambda_lab'] * loss_lab +
                        loss_weights['lambda_freq'] * loss_freq
                    ) / ACCUMULATION_STEPS

                local_skip = bool(torch.isnan(loss) or torch.isinf(loss))
                if _any_rank_skips(local_skip):
                    if is_main() and local_skip:
                        logger.warning(
                            f"Invalid loss at batch {batch_idx}: "
                            f"diff={loss_diff.item():.4f}, recon={loss_recon.item():.4f}, "
                            f"contr={loss_contrastive.item():.4f}, vicreg={loss_vicreg.item():.4f}, "
                            f"lab={loss_lab.item():.4f}, freq={loss_freq.item():.4f}"
                        )
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    continue

                loss.backward()

                if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                    has_nan_grad = any(
                        p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                        for g in optimizer.param_groups for p in g["params"]
                    )
                    if has_nan_grad:
                        if is_main():
                            logger.warning(f"NaN gradient at batch {batch_idx}, skipping step")
                        optimizer.zero_grad()
                        torch.cuda.empty_cache()
                        continue

                    torch.nn.utils.clip_grad_norm_(
                        [p for g in optimizer.param_groups for p in g["params"]],
                        max_norm=MAX_GRAD_NORM,
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                    update_lr()

                    valid_batches += 1

                total_loss             += loss.item() * ACCUMULATION_STEPS
                total_diff_loss        += loss_diff.item()
                total_recon_loss       += loss_recon.item()
                total_contrastive_loss += loss_contrastive.item()
                total_vicreg_loss      += loss_vicreg.item()
                total_lab_loss         += loss_lab.item()
                total_freq_loss        += loss_freq.item()
                batch_count += 1

                if is_main():
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"{ts} | Epoch {epoch+1}/{SDXL_NUM_EPOCHS} [{batch_idx}/{num_batches}] | "
                        f"loss={loss.item()*ACCUMULATION_STEPS:.4f} diff={loss_diff.item():.4f} "
                        f"recon={loss_recon.item():.4f} contr={loss_contrastive.item():.4f} "
                        f"vicreg={loss_vicreg.item():.4f} lab={loss_lab.item():.4f} "
                        f"freq={loss_freq.item():.4f} lr={optimizer.param_groups[0]['lr']:.2e}",
                        flush=True
                    )

                if (batch_idx + 1) % 100 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if is_main():
                        logger.warning(f"OOM at batch {batch_idx}, clearing cache...")
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    gc.collect()
                    # Force all ranks to skip this microbatch together to avoid
                    # an all-reduce deadlock with ranks that did not OOM.
                    _any_rank_skips(True)
                    continue
                raise

        if valid_batches == 0:
            if is_main():
                logger.warning(f"Epoch {epoch + 1}: all batches failed!")
            continue

        avg_total       = total_loss             / max(1, valid_batches)
        avg_diff        = total_diff_loss        / max(1, batch_count)
        avg_recon       = total_recon_loss       / max(1, batch_count)
        avg_contrastive = total_contrastive_loss / max(1, batch_count)
        avg_vicreg      = total_vicreg_loss      / max(1, batch_count)
        avg_lab         = total_lab_loss         / max(1, batch_count)
        avg_freq        = total_freq_loss        / max(1, batch_count)

        if is_main():
            logger.info(
                f"Epoch {epoch + 1}/{SDXL_NUM_EPOCHS} TRAIN | "
                f"Loss: {avg_total:.4f} | Diff: {avg_diff:.4f} | Recon: {avg_recon:.4f} | "
                f"Contr: {avg_contrastive:.4f} | VICReg: {avg_vicreg:.4f} | "
                f"Lab: {avg_lab:.4f} | Freq: {avg_freq:.4f} | "
                f"Batches: {valid_batches}/{len(train_loader)}"
            )

        # ── Validation ────────────────────────────────────────────────────────
        dist.barrier()

        val_diff_loss = 0.0
        val_total_loss = 0.0
        emb_metrics = {}
        sample_psnr = 0.0

        if is_main():
            if val_loader:
                val_diff_loss, emb_metrics = validate(
                    compressor_ddp, decoder, val_loader, vae, device, embedding_analyzer
                )
                val_total_loss = val_diff_loss

                logger.info(f"Epoch {epoch + 1}/{SDXL_NUM_EPOCHS} VAL   | Loss: {val_total_loss:.4f}")
                embedding_analyzer.log_metrics(emb_metrics, logger, prefix="  ")

            # ── Epoch sample ──────────────────────────────────────────────────
            decoder.unet.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()

            n_samp = 3 if epoch < 3 else 10
            sample_psnr = generate_epoch_samples(
                epoch + 1, compressor_ddp, decoder, vae, device, full_dataset, train_dataset, n_samples=n_samp
            )

        log_to_csv(epoch + 1, avg_diff, avg_recon, avg_contrastive,
                   avg_vicreg, avg_lab, avg_freq, avg_total,
                   val_diff_loss, val_total_loss, sample_psnr, emb_metrics)

        # ── Checkpoint (rank 0 only) ──────────────────────────────────────────
        if is_main():
            is_best = val_total_loss < best_val_loss if val_loader else True
            if is_best and val_loader:
                best_val_loss = val_total_loss

            if SAVE_EVERY_EPOCH or is_best:
                ckpt_name = f"checkpoint_epoch_{epoch + 1:03d}"
                if is_best:
                    ckpt_name += "_best"
                ckpt_path = os.path.join(CHECKPOINTS_DIR, f"{ckpt_name}.pth")

                torch.save({
                    "epoch":                   epoch,
                    "compressor_state_dict":   compressor.state_dict(),
                    "token_proj_state_dict":   _unwrap(decoder.token_proj).state_dict(),
                    "pooled_proj_state_dict":  _unwrap(decoder.pooled_proj).state_dict(),
                    "null_token":              decoder.null_token.data,
                    "optimizer_state_dict":    optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "curr_step":               curr_step,
                    "loss":                    avg_total,
                    "val_loss":                val_total_loss,
                    "best_val_loss":           best_val_loss,
                    "psnr":                    sample_psnr,
                    "emb_metrics":             emb_metrics,
                }, ckpt_path)
                logger.info(f"Checkpoint saved: {ckpt_path}")
                cleanup_old_checkpoints(CHECKPOINTS_DIR, keep_last_n=3)

        # All ranks wait for rank-0 to finish val/samples/checkpoint
        dist.barrier()
        torch.cuda.empty_cache()
        gc.collect()

    if is_main():
        logger.info("=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Loss logs   : {LOSS_LOG_FILE_RUN}")
        logger.info(f"Samples     : {EPOCH_SAMPLES_DIR_RUN}")
        logger.info(f"Checkpoints : {CHECKPOINTS_DIR}")
        logger.info("=" * 70)

    cleanup_ddp()


if __name__ == "__main__":
    train()
