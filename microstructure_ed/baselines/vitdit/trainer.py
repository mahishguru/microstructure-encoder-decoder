"""
trainer_vitdit.py  —  DDP training script for the ViTDiT (DiT-XL/2 + DDPM) decoder.

Launch with:
    python -m torch.distributed.run --nproc_per_node=4 -m microstructure_ed.baselines.vitdit.trainer
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
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


from microstructure_ed.config import (
    DEVICE, ENCODER_CKPT, DATA_DIR,
    VITDIT_BATCH_SIZE, VITDIT_LR, VITDIT_WEIGHT_DECAY, VITDIT_USE_AMP, VITDIT_ACCUMULATION_STEPS,
    VITDIT_NUM_INFERENCE_STEPS_TRAIN, VITDIT_SAVE_EVERY_EPOCH, VITDIT_RESUME_FROM,
    NUM_EPOCHS, NUM_WORKERS, MAX_IMAGES, WARMUP_STEPS, MAX_GRAD_NORM, VAL_SPLIT,
)
from microstructure_ed.encoder_arch_pretrained import Compressor
from microstructure_ed.baselines.vitdit.decoder_arch_pretrained import (
    ViTDiTDecoder, SDVAE, ddpm_loss, composite_ddpm_loss,
    TARGET_DIM as VARIANT_TARGET_DIM,  # 512
)
from microstructure_ed.baselines.vitdit.embedding_utils import ContrastiveLoss, EmbeddingAnalyzer, LossWeightScheduler
from microstructure_ed.baselines.vitsdxl.embedding_utils import VICRegLoss, FrequencyLoss


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


# ── Logging (rank-0 only) ────────────────────────────────────────────────────
_RUN_TAG          = datetime.now().strftime('%Y%m%d_%H%M%S')

CHECKPOINTS_DIR   = "./checkpoints/vitdit"
LOGS_DIR          = f"./logs/vitdit/run_{_RUN_TAG}"
EPOCH_SAMPLES_DIR = f"./samples/vitdit/run_{_RUN_TAG}"
LOSS_LOG_FILE     = os.path.join(LOGS_DIR, "loss_log.csv")

logger = logging.getLogger(__name__)


def setup_logging():
    if not is_main():
        logging.basicConfig(level=logging.WARNING)
        return
    log_file = os.path.join(LOGS_DIR, f"train_{_RUN_TAG}.log")
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(EPOCH_SAMPLES_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )


# ── Dataset ───────────────────────────────────────────────────────────────────
class ImageDataset(Dataset):
    def __init__(self, folder_path: str, max_images: int = None):
        exts  = {".png", ".jpg", ".jpeg"}
        files = sorted(f for f in os.listdir(folder_path) if Path(f).suffix.lower() in exts)
        if max_images:
            files = files[:max_images]
        self.files       = files
        self.folder_path = folder_path
        self.transform   = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        if is_main():
            logger.info(f"Dataset: {len(self.files)} images from {folder_path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.folder_path, self.files[idx])
        img  = Image.open(path).convert("RGB")
        return self.transform(img)


def create_train_val_split(dataset: Dataset, val_split: float = 0.1):
    if val_split <= 0 or val_split >= 1:
        return dataset, None
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=val_split, random_state=42, shuffle=True)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


# ── Utilities ─────────────────────────────────────────────────────────────────
def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    t = (tensor / 2 + 0.5).clamp(0, 1)
    arr = (t.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(arr)

def compute_psnr(img_a: torch.Tensor, img_b: torch.Tensor) -> float:
    mse = torch.mean((img_a.float() - img_b.float()) ** 2).item()
    if mse <= 0:
        return float("inf")
    return 20 * torch.log10(torch.tensor(2.0 / (mse ** 0.5))).item()

def _unwrap(model):
    return model.module if isinstance(model, DDP) else model

def generate_epoch_samples(epoch, encoder, decoder, vae, full_dataset, train_subset, device, n_samples=10):
    if not is_main():
        return 0.0
    psnr_values = []
    epoch_dir = os.path.join(EPOCH_SAMPLES_DIR, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    enc_raw = _unwrap(encoder)
    dec_raw = _unwrap(decoder)
    enc_raw.eval(); dec_raw.eval()
    indices = random.sample(range(len(train_subset)), min(n_samples, len(train_subset)))

    orig_pils  = []
    recon_pils = []
    names      = []

    for k, pos in enumerate(indices):
        try:
            dataset_idx   = train_subset.indices[pos]
            sample_tensor = full_dataset[dataset_idx].unsqueeze(0).to(device)
            orig_pil      = tensor_to_pil(sample_tensor[0])
            sample_name   = full_dataset.files[dataset_idx]

            with torch.no_grad():
                with autocast('cuda', dtype=torch.bfloat16):
                    z_512        = enc_raw(sample_tensor)
                    latent_recon = dec_raw.sample(z_512, num_steps=VITDIT_NUM_INFERENCE_STEPS_TRAIN)
                    img_recon    = vae.decode(latent_recon)

            psnr = compute_psnr(sample_tensor, img_recon)
            psnr_values.append(psnr)
            recon_pil = tensor_to_pil(img_recon.cpu())

            orig_pils.append(orig_pil)
            recon_pils.append(recon_pil)
            names.append(sample_name)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(orig_pil); axes[0].set_title(f"Original: {sample_name}", fontsize=9); axes[0].axis("off")
            axes[1].imshow(recon_pil); axes[1].set_title(f"ViTDiT | PSNR: {psnr:.2f} dB", fontsize=9); axes[1].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(epoch_dir, f"sample_{k+1:02d}_{Path(sample_name).stem}.png"), dpi=120)
            plt.close()

        except Exception as e:
            logger.warning(f"Epoch {epoch} sample {k+1}: {e}")

    if orig_pils:
        n = len(orig_pils)
        fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
        if n == 1:
            axes = axes.reshape(2, 1)
        for j in range(n):
            axes[0, j].imshow(orig_pils[j]); axes[0, j].set_title(f"Orig: {Path(names[j]).stem}", fontsize=7); axes[0, j].axis("off")
            axes[1, j].imshow(recon_pils[j]); axes[1, j].set_title(f"PSNR: {psnr_values[j]:.2f}", fontsize=7); axes[1, j].axis("off")
        plt.suptitle(f"Epoch {epoch} — Mean PSNR: {sum(psnr_values)/len(psnr_values):.2f} dB", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(epoch_dir, f"grid_epoch_{epoch:03d}.png"), dpi=150)
        plt.close()
        logger.info(f"Epoch {epoch} samples saved -> {epoch_dir}/ | Mean PSNR: {sum(psnr_values)/len(psnr_values):.2f} dB")

    return sum(psnr_values) / max(1, len(psnr_values)) if psnr_values else 0.0


@torch.no_grad()
def validate(encoder, decoder, val_loader, vae, device):
    enc_raw = _unwrap(encoder)
    dec_raw = _unwrap(decoder)
    enc_raw.eval(); dec_raw.eval()
    total_loss = 0.0
    for batch in val_loader:
        images = batch.to(device)
        with autocast('cuda', dtype=torch.bfloat16):
            z_512 = enc_raw(images)
            x_0   = vae.encode(images)
            loss, _ = ddpm_loss(dec_raw, x_0, z_512)
        total_loss += loss.item()
    return total_loss / max(1, len(val_loader))


# ── CSV Logger ────────────────────────────────────────────────────────────────
def init_csv_logger():
    if not is_main():
        return
    if not os.path.exists(LOSS_LOG_FILE):
        with open(LOSS_LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "train_eps", "train_recon", "train_contrastive",
                "train_vicreg", "train_freq", "val_loss", "psnr_sample"
            ])

def log_to_csv(epoch, train_loss, train_eps, train_recon, train_contr,
               train_vicreg, train_freq, val_loss, psnr):
    if not is_main():
        return
    with open(LOSS_LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            epoch,
            f"{train_loss:.6f}",
            f"{train_eps:.6f}",
            f"{train_recon:.6f}",
            f"{train_contr:.6f}",
            f"{train_vicreg:.6f}",
            f"{train_freq:.6f}",
            f"{val_loss:.6f}",
            f"{psnr:.4f}"
        ])

def cleanup_old_checkpoints(keep_last_n: int = 3):
    pattern = re.compile(r"checkpoint_epoch_(\d+)")
    ckpts = sorted(
        [p for p in Path(CHECKPOINTS_DIR).glob("*.pth") if pattern.search(p.name)],
        key=lambda p: int(pattern.search(p.name).group(1)),
    )
    for old in ckpts[:-keep_last_n]:
        old.unlink()


# ── Training Loop ─────────────────────────────────────────────────────────────
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

    BATCH_SIZE         = VITDIT_BATCH_SIZE
    LR                 = VITDIT_LR  # no aggressive scaling; config LR is the peak LR
    ENCODER_LR         = LR                              # uniform LR (matches old working setup)
    BACKBONE_LR        = LR                              # uniform LR (matches old working setup)
    ACCUMULATION_STEPS = VITDIT_ACCUMULATION_STEPS
    RESUME_FROM        = VITDIT_RESUME_FROM
    PIXEL_NOISE_SCALE  = 0.05
    VITDIT_NUM_EPOCHS  = 30
    UNFREEZE_EPOCH     = 9999  # Disabled — encoder already trainable from epoch 1

    init_csv_logger()
    full_dataset  = ImageDataset(DATA_DIR, MAX_IMAGES)
    train_dataset, val_dataset = create_train_val_split(full_dataset, VAL_SPLIT)

    # ── DDP sampler ───────────────────────────────────────────────────────────
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                       rank=dist.get_rank(), shuffle=True, drop_last=True)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        persistent_workers=True,
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                   num_workers=NUM_WORKERS, pin_memory=True)
        if val_dataset else None
    )

    # ── Models ────────────────────────────────────────────────────────────────
    if is_main():
        logger.info("Loading models ...")

    # Phase 1: trainable_blocks=0 → freeze ALL ViT blocks.
    # Only attention_pooler + target_proj train (the bottleneck head).
    # Phase 2 (epoch >= UNFREEZE_EPOCH): unfreeze last 16 blocks.
    encoder = Compressor(use_gradient_checkpointing=True, trainable_blocks=16,
                         target_dim=VARIANT_TARGET_DIM).to(device)
    encoder.train()

    vae     = SDVAE().to(device)
    decoder = ViTDiTDecoder().to(device)

    # ── Wrap trainable models in DDP ──────────────────────────────────────────
    encoder_ddp = DDP(encoder, device_ids=[local_rank], find_unused_parameters=True)
    decoder_ddp = DDP(decoder, device_ids=[local_rank], find_unused_parameters=True)

    # ── Helper: build / rebuild optimizer for current trainable params ────────
    def _build_optimizer(enc, dec, enc_lr, backbone_lr, adapter_lr):
        enc_p = [p for p in enc.parameters() if p.requires_grad]
        # Split decoder into backbone vs adapter params
        backbone_names = {"backbone."}
        backbone_p, adapter_p = [], []
        for n, p in dec.named_parameters():
            if not p.requires_grad:
                continue
            if any(bk in n for bk in backbone_names):
                backbone_p.append(p)
            else:
                adapter_p.append(p)
        all_p = enc_p + backbone_p + adapter_p
        opt = optim.AdamW([
            {"params": enc_p,      "lr": enc_lr},
            {"params": backbone_p, "lr": backbone_lr},
            {"params": adapter_p,  "lr": adapter_lr},
        ], weight_decay=VITDIT_WEIGHT_DECAY, fused=True)
        return opt, all_p, enc_p, backbone_p, adapter_p

    optimizer, trainable_params, encoder_params, backbone_params, adapter_params = \
        _build_optimizer(encoder_ddp, decoder_ddp, ENCODER_LR, BACKBONE_LR, LR)

    if is_main():
        logger.info(f"Phase 1 (frozen ViT) — encoder {sum(p.numel() for p in encoder_params):,}  "
                     f"backbone {sum(p.numel() for p in backbone_params):,}  "
                     f"adapters {sum(p.numel() for p in adapter_params):,}")

    contrastive_fn     = ContrastiveLoss(temperature=0.07).to(device)
    vicreg_fn          = VICRegLoss(var_weight=25.0, cov_weight=1.0, var_target=1.0).to(device)
    freq_fn            = FrequencyLoss().to(device)
    embedding_analyzer = EmbeddingAnalyzer()
    loss_scheduler     = LossWeightScheduler(
        num_epochs=VITDIT_NUM_EPOCHS,
        lambda_diff_start=0.5,        lambda_diff_end=1.0,
        lambda_recon_start=0.3,       lambda_recon_end=0.1,
        lambda_contrastive_start=0.0, lambda_contrastive_end=0.05,
        lambda_vicreg=0.01,
        lambda_freq=0.1,
        warmup_epochs=5,
        warmup_epochs_vicreg=5,                          # delay: silent epochs 0-4
        warmup_epochs_freq=6,                            # delay: silent epochs 0-5
        ramp_epochs_vicreg=3,                            # ramp epochs 5-7, full from 8
        ramp_epochs_freq=3,                              # ramp epochs 6-8, full from 9
    )

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, len(train_loader) * VITDIT_NUM_EPOCHS - WARMUP_STEPS), eta_min=1e-7,
    )
    curr_step    = 0
    start_epoch  = 0
    best_val_loss= float('inf')

    if RESUME_FROM and os.path.exists(RESUME_FROM):
        ckpt = torch.load(RESUME_FROM, map_location="cpu")
        if "encoder" in ckpt:
            encoder.load_state_dict(ckpt["encoder"])
            if is_main():
                logger.info("Loaded encoder weights from checkpoint.")
        decoder.load_state_dict(ckpt["decoder"])

        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float('inf'))

        # If resuming into Phase 2, unfreeze blocks & rebuild optimizer first
        if start_epoch >= UNFREEZE_EPOCH:
            encoder.unfreeze_vit_blocks(last_n=16)
            optimizer, trainable_params, encoder_params, backbone_params, adapter_params = \
                _build_optimizer(encoder_ddp, decoder_ddp, ENCODER_LR, BACKBONE_LR, LR)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, len(train_loader) * (VITDIT_NUM_EPOCHS - UNFREEZE_EPOCH)),
                eta_min=1e-7,
            )
            if is_main():
                logger.info(f"Resume into Phase 2 — unfroze last 16 ViT blocks | "
                             f"encoder {sum(p.numel() for p in encoder_params):,}  "
                             f"backbone {sum(p.numel() for p in backbone_params):,}  "
                             f"adapters {sum(p.numel() for p in adapter_params):,}")

        optimizer.load_state_dict(ckpt["optimizer"])
        if is_main():
            logger.info(f"Resumed from epoch {start_epoch}")

    use_amp = VITDIT_USE_AMP
    scaler  = None  # bf16 does not need GradScaler (only fp16 does)
    if is_main():
        logger.info(f"AMP enabled: {use_amp}")
        logger.info(
            f"TRAINING  |  epochs {start_epoch + 1} -> {VITDIT_NUM_EPOCHS}"
            f"  |  batch_size {BATCH_SIZE}x{world_size}GPUs  |  accum {ACCUMULATION_STEPS}"
        )

    # ── Main Loop ─────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, VITDIT_NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        # ── Phase 2: unfreeze late ViT blocks ──────────────────────────────────
        if epoch == UNFREEZE_EPOCH:
            encoder.unfreeze_vit_blocks(last_n=16)
            # Reduce batch size to fit extra 315M trainable encoder params
            PHASE2_BS = max(16, BATCH_SIZE // 2)
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=PHASE2_BS, sampler=train_sampler,
                num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                persistent_workers=True,
            )
            num_batches = len(train_loader)
            if is_main():
                logger.info(f"Phase 2 batch_size reduced: {BATCH_SIZE} -> {PHASE2_BS}")
            optimizer, trainable_params, encoder_params, backbone_params, adapter_params = \
                _build_optimizer(encoder_ddp, decoder_ddp, ENCODER_LR, BACKBONE_LR, LR)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, len(train_loader) * (VITDIT_NUM_EPOCHS - UNFREEZE_EPOCH)),
                eta_min=1e-7,
            )
            if is_main():
                logger.info(f"Phase 2 — unfroze last 16 ViT blocks | "
                             f"encoder {sum(p.numel() for p in encoder_params):,}  "
                             f"backbone {sum(p.numel() for p in backbone_params):,}  "
                             f"adapters {sum(p.numel() for p in adapter_params):,}")
        loss_weights = loss_scheduler.get_weights(epoch)
        if is_main():
            loss_scheduler.log_weights(epoch, logger)
        decoder_ddp.train()
        encoder_ddp.train()

        total_loss = 0.0
        total_eps = 0.0
        total_recon = 0.0
        total_contr = 0.0
        total_vicreg = 0.0
        total_freq = 0.0
        batch_count = 0

        num_batches = len(train_loader)
        optimizer.zero_grad()

        for batch_idx, images in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            try:
                z_512 = encoder_ddp(images)

                if loss_weights["lambda_contrastive"] > 0:
                    pixel_noise = torch.randn_like(images) * PIXEL_NOISE_SCALE
                    images_aug  = (images + pixel_noise).clamp(-1.0, 1.0)
                    z_512_aug   = encoder_ddp(images_aug)
                else:
                    z_512_aug = z_512

                with torch.no_grad():
                    x_0 = vae.encode(images)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    loss, info = composite_ddpm_loss(
                        decoder_ddp, x_0, z_512, z_512_aug,
                        contrastive_fn=contrastive_fn,
                        lambda_diff=loss_weights["lambda_diff"],
                        lambda_recon=loss_weights["lambda_recon"],
                        lambda_contrastive=loss_weights["lambda_contrastive"],
                        vicreg_fn=vicreg_fn,
                        freq_fn=freq_fn,
                        vae=vae,
                        lambda_vicreg=loss_weights["lambda_vicreg"],
                        lambda_freq=loss_weights["lambda_freq"],
                    )
                    loss = loss / ACCUMULATION_STEPS

                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad()
                    if scaler:
                        scaler.update()
                    torch.cuda.empty_cache()
                    continue

                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    if curr_step < WARMUP_STEPS:
                        frac = curr_step / max(1, WARMUP_STEPS)
                        optimizer.param_groups[0]["lr"] = ENCODER_LR * frac   # encoder
                        optimizer.param_groups[1]["lr"] = BACKBONE_LR * frac  # backbone
                        optimizer.param_groups[2]["lr"] = LR * frac           # adapters
                    else:
                        lr_scheduler.step()
                    curr_step += 1

                total_loss   += loss.item() * ACCUMULATION_STEPS
                total_eps    += info["loss_epsilon"]
                total_recon  += info["loss_recon"]
                total_contr  += info["loss_contrastive"]
                total_vicreg += info["loss_vicreg"]
                total_freq   += info["loss_freq"]
                batch_count += 1

                if is_main():
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"{ts} | Epoch {epoch+1}/{VITDIT_NUM_EPOCHS} [{batch_idx}/{num_batches}] | "
                        f"loss={total_loss/max(1,batch_count):.4f} eps={info['loss_epsilon']:.4f} "
                        f"recon={info['loss_recon']:.4f} contr={info['loss_contrastive']:.4f} "
                        f"vicreg={info['loss_vicreg']:.4f} freq={info['loss_freq']:.4f} "
                        f"lr_enc={optimizer.param_groups[0]['lr']:.2e} "
                        f"lr_bb={optimizer.param_groups[1]['lr']:.2e} "
                        f"lr_adp={optimizer.param_groups[2]['lr']:.2e}",
                        flush=True
                    )

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if is_main():
                        logger.warning(f"OOM at batch {batch_idx}, clearing cache ...")
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                raise

        avg_loss   = total_loss   / max(1, batch_count)
        avg_eps    = total_eps    / max(1, batch_count)
        avg_recon  = total_recon  / max(1, batch_count)
        avg_contr  = total_contr  / max(1, batch_count)
        avg_vicreg = total_vicreg / max(1, batch_count)
        avg_freq   = total_freq   / max(1, batch_count)

        if is_main():
            logger.info(
                f"Epoch {epoch + 1}/{VITDIT_NUM_EPOCHS} TRAIN | "
                f"Loss: {avg_loss:.4f} | Eps: {avg_eps:.4f} | Recon: {avg_recon:.4f} | "
                f"Contr: {avg_contr:.4f} | VICReg: {avg_vicreg:.4f} | Freq: {avg_freq:.4f}"
            )

        dist.barrier()

        val_loss = 0.0
        sample_psnr = 0.0
        if is_main():
            if val_loader:
                val_loss = validate(encoder_ddp, decoder_ddp, val_loader, vae, device)
                logger.info(f"Epoch {epoch + 1}/{VITDIT_NUM_EPOCHS} VAL   | Loss: {val_loss:.4f}")

            torch.cuda.empty_cache()
            n_samp = 3 if epoch < 3 else 10
            sample_psnr = generate_epoch_samples(
                epoch + 1, encoder_ddp, decoder_ddp, vae, full_dataset, train_dataset, device, n_samples=n_samp
            )

        log_to_csv(epoch + 1, avg_loss, avg_eps, avg_recon, avg_contr,
                   avg_vicreg, avg_freq, val_loss, sample_psnr)

        # ── Checkpointing (rank 0 only) ──────────────────────────────────────
        if is_main():
            is_best = val_loss < best_val_loss if val_loader else True
            if is_best and val_loader:
                best_val_loss = val_loss

            if VITDIT_SAVE_EVERY_EPOCH or is_best:
                suffix   = "_best" if is_best else ""
                ckpt_path = os.path.join(
                    CHECKPOINTS_DIR,
                    f"checkpoint_epoch_{epoch + 1:03d}{suffix}.pth",
                )
                torch.save({
                    "epoch":         epoch,
                    "encoder":       encoder.state_dict(),
                    "decoder":       decoder.state_dict(),
                    "optimizer":     optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                }, ckpt_path)
                logger.info(f"Checkpoint saved: {ckpt_path}")
                cleanup_old_checkpoints()

        # All ranks wait for rank-0 to finish val/samples/checkpoint
        dist.barrier()

    cleanup_ddp()


if __name__ == "__main__":
    train()
