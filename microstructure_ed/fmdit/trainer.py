"""
trainer_vitfmdit.py  —  DDP training script for the FM-DiT (SD3.5 hijack) decoder.

Launch with:
    FMDIT_TARGET_DIM=512 torchrun --nproc_per_node=4 -m microstructure_ed.fmdit.trainer
"""

import os, sys, re, csv, gc, random, logging
from pathlib import Path
from datetime import datetime

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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
    FMDIT_MODEL_ID,
    FMDIT_BATCH_SIZE, FMDIT_LR, FMDIT_USE_AMP, FMDIT_ACCUMULATION_STEPS,
    FMDIT_NUM_INFERENCE_STEPS_TRAIN, FMDIT_SAVE_EVERY_EPOCH, FMDIT_RESUME_FROM,
    NUM_EPOCHS, NUM_WORKERS, MAX_IMAGES, WARMUP_STEPS, MAX_GRAD_NORM, VAL_SPLIT,
    FMDIT_LAMBDA_ORIENT, FMDIT_ORIENT_WARMUP_EPOCHS, FMDIT_ORIENT_RAMP_EPOCHS,
    FMDIT_ORIENT_PIX_WEIGHT, FMDIT_ORIENT_POOL_WEIGHT, FMDIT_ORIENT_POOL_SIZE,
    FMDIT_ORIENT_EPS, FMDIT_ORIENT_V2,
    FMDIT_PIXEL_MAX_SAMPLES, FMDIT_PIXEL_T_THRESHOLD,
)
from microstructure_ed.encoder_arch_pretrained import Compressor
from microstructure_ed.fmdit.decoder_arch_pretrained import (
    FlowMatchingDiTDecoder, SD35VAE, flow_matching_loss, composite_fm_loss,
    TARGET_DIM as VARIANT_TARGET_DIM,  # FMDIT_TARGET_DIM
    SPATIAL_NUM_TOKENS as VARIANT_SPATIAL_TOKENS,  # 16 (spatial-token latent)
)
from microstructure_ed.fmdit.embedding_utils import (
    ContrastiveLoss, EmbeddingAnalyzer, LossWeightScheduler,
)
# Differentiable, HCP-symmetry-aware orientation loss (variant-agnostic).
from microstructure_ed.orientation_loss import OrientationDistributionLoss
# VICReg + FrequencyLoss are decoder-agnostic — reuse the SDXL implementations
# as the single source of truth.
from microstructure_ed.baselines.vitsdxl.embedding_utils import VICRegLoss, FrequencyLoss

# ── DDP helpers ───────────────────────────────────────────────────────────────

def setup_ddp():
    """Initialise the process group. Works with torchrun launcher."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


# ── Logging (rank-0 only) ────────────────────────────────────────────────────

_RUN_TAG         = datetime.now().strftime('%Y%m%d_%H%M%S')

CHECKPOINTS_DIR  = os.environ.get("FMDIT_CKPT_DIR", f"./checkpoints/fmdit_{VARIANT_TARGET_DIM}")
LOGS_DIR         = f"./logs/fmdit_{VARIANT_TARGET_DIM}/run_{_RUN_TAG}"
EPOCH_SAMPLES_DIR= f"./samples/fmdit_{VARIANT_TARGET_DIM}/run_{_RUN_TAG}"
LOSS_LOG_FILE    = os.path.join(LOGS_DIR, "loss_log.csv")

# A/B reproducibility: identical init + data order across arms when FMDIT_SEED set
if os.environ.get('FMDIT_SEED'):
    import random as _rnd, numpy as _np
    _seed = int(os.environ['FMDIT_SEED'])
    _rnd.seed(_seed); _np.random.seed(_seed)
    torch.manual_seed(_seed); torch.cuda.manual_seed_all(_seed)

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
            # NEAREST: IPF maps are piecewise-constant (one colour/grain);
            # BILINEAR invents fake boundary orientations and fragments grains.
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
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
    indices  = list(range(len(dataset)))
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
    """Return the underlying model from DDP wrapper (or the model itself)."""
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
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    z_512      = enc_raw(sample_tensor)
                    latent     = dec_raw.sample(z_512, num_steps=FMDIT_NUM_INFERENCE_STEPS_TRAIN)
                    img_recon  = vae.decode(latent)

            psnr = compute_psnr(sample_tensor, img_recon)
            psnr_values.append(psnr)
            recon_pil = tensor_to_pil(img_recon.cpu())

            orig_pils.append(orig_pil)
            recon_pils.append(recon_pil)
            names.append(sample_name)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(orig_pil); axes[0].set_title(f"Original: {sample_name}", fontsize=9); axes[0].axis("off")
            axes[1].imshow(recon_pil); axes[1].set_title(f"ViTFMDiT | PSNR: {psnr:.2f} dB", fontsize=9); axes[1].axis("off")
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
        logger.info(f"Epoch {epoch} samples saved → {epoch_dir}/ | Mean PSNR: {sum(psnr_values)/len(psnr_values):.2f} dB")

    mean_psnr = sum(psnr_values) / max(1, len(psnr_values)) if psnr_values else 0.0
    return mean_psnr

@torch.no_grad()
def validate(encoder, decoder, val_loader, vae, device):
    enc_raw = _unwrap(encoder)
    dec_raw = _unwrap(decoder)
    enc_raw.eval(); dec_raw.eval()
    total_loss = 0.0
    for batch in val_loader:
        images = batch.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            z_512 = enc_raw(images)
            x_0   = vae.encode(images)
            loss, _ = flow_matching_loss(dec_raw, x_0, z_512)
        total_loss += loss.item()
    return total_loss / max(1, len(val_loader))


# ── CSV Logger ────────────────────────────────────────────────────────────────

def init_csv_logger():
    if not is_main():
        return
    os.makedirs(os.path.dirname(LOSS_LOG_FILE), exist_ok=True)
    if not os.path.exists(LOSS_LOG_FILE):
        with open(LOSS_LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "train_velocity", "train_recon", "train_contrastive",
                "train_vicreg", "train_freq", "train_orient",
                "train_orient_odf", "train_orient_misori", "train_orient_pf",
                "train_orient_scatter", "train_latstat",
                "train_orient_pix", "train_orient_pfsharp",
                "val_loss", "psnr_sample"
            ])

def log_to_csv(epoch, train_loss, train_vel, train_recon, train_contr,
               train_vicreg, train_freq, train_orient, val_loss, psnr,
               orient_sub=None):
    if not is_main():
        return
    os.makedirs(os.path.dirname(LOSS_LOG_FILE), exist_ok=True)
    with open(LOSS_LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            epoch,
            f"{train_loss:.6f}",
            f"{train_vel:.6f}",
            f"{train_recon:.6f}",
            f"{train_contr:.6f}",
            f"{train_vicreg:.6f}",
            f"{train_freq:.6f}",
            f"{train_orient:.6f}",
            *[f"{(orient_sub or {}).get(k, 0.0):.6f}" for k in
              ("odf", "misori", "pf", "scatter", "latstat", "pix", "pfsharp")],
            f"{val_loss:.6f}",
            f"{psnr:.4f}"
        ])

def cleanup_old_checkpoints(keep_last_n: int = 3):
    """Retain the best checkpoint (newest '_best') plus the `keep_last_n` most-recent ones."""
    pattern = re.compile(r"checkpoint_epoch_(\d+)")
    ckpts = sorted(
        [p for p in Path(CHECKPOINTS_DIR).glob("*.pth") if pattern.search(p.name)],
        key=lambda p: int(pattern.search(p.name).group(1)),
    )
    keep = set(ckpts[-keep_last_n:])                       # N most-recent epochs
    best = [p for p in ckpts if "_best" in p.name]
    if best:
        keep.add(best[-1])                                # newest best == global best
    for old in ckpts:
        if old not in keep:
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

    # ── Speed: enable TF32 on H100 for ~2x matmul throughput ─────────────────
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    BATCH_SIZE         = int(os.environ.get("FMDIT_BATCH_SIZE_OVERRIDE", FMDIT_BATCH_SIZE))
    LR                 = FMDIT_LR * (world_size ** 0.5)  # √N scaling: 4 GPUs → 2× LR
    LR                 = float(os.environ.get("FMDIT_LR_SCALED_OVERRIDE", LR))
    ENCODER_LR         = LR * 0.05                       # 20× lower for pretrained ViT blocks
    ACCUMULATION_STEPS = int(os.environ.get("FMDIT_ACCUM_OVERRIDE", FMDIT_ACCUMULATION_STEPS))
    # Env override so an A/B arm can resume WITHOUT touching the shared
    # config.json (other variants read the same fmdit_resume_from key).
    RESUME_FROM        = os.environ.get("FMDIT_RESUME_FROM_OVERRIDE", FMDIT_RESUME_FROM)
    PIXEL_NOISE_SCALE  = 0.05
    FMDIT_NUM_EPOCHS   = int(os.environ.get("FMDIT_NUM_EPOCHS_OVERRIDE", "30"))
    UNFREEZE_EPOCH     = 8   # Phase 2: unfreeze last 16 ViT blocks at this epoch
    # v3: env-overridable so the orient_v3 arm can enable LoRA earlier (more
    # epochs of decoder plasticity under orientation supervision).
    LORA_START_EPOCH   = int(os.environ.get("FMDIT_LORA_START_EPOCH", "8"))
    # C1/C3 fine-tune knobs (defaults OFF -> legacy behaviour)
    CFG_DROPOUT        = float(os.environ.get("FMDIT_CFG_DROPOUT", "0"))
    TEXHEAD_LAMBDA     = float(os.environ.get("FMDIT_TEXHEAD_LAMBDA", "0"))

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
        logger.info("Loading models …")

    # Phase 1: trainable_blocks=0 → freeze ALL ViT blocks.
    # Only attention_pooler + target_proj train (the bottleneck head).
    # Phase 2 (epoch >= UNFREEZE_EPOCH): unfreeze last 16 blocks.
    encoder = Compressor(use_gradient_checkpointing=True, trainable_blocks=0,
                         target_dim=VARIANT_TARGET_DIM,
                         spatial_tokens=VARIANT_SPATIAL_TOKENS).to(device)
    encoder.train()

    _hf_token = os.environ.get("HF_TOKEN")
    vae     = SD35VAE(token=_hf_token).to(device)
    decoder = FlowMatchingDiTDecoder(token=_hf_token).to(device)

    def _set_lora_trainable(dec_module, flag):
        """Toggle requires_grad on the MMDiT LoRA params (decoder backbone lever)."""
        n = 0
        for _name, _p in dec_module.named_parameters():
            if 'lora_' in _name:
                _p.requires_grad = flag
                n += _p.numel()
        return n

    def _wrap_ddp(mod):
        return DDP(mod, device_ids=[local_rank], find_unused_parameters=True)

    # Phase the decoder backbone (LoRA) in like the encoder ViT: hold it frozen
    # during head-warmup so the spatial-latent conditioning pathway establishes
    # first, then enable it at LORA_START_EPOCH. DDP is re-wrapped at the
    # transition so the reducer registers the newly-trainable LoRA params
    # (toggling requires_grad after DDP init alone leaves their grads un-synced).
    if decoder.lora_enabled and LORA_START_EPOCH > 0:
        _nlora = _set_lora_trainable(decoder, False)
        if is_main():
            logger.info(f'LoRA on MMDiT held FROZEN until epoch {LORA_START_EPOCH} ({_nlora:,} params)')

    # ── Wrap trainable models in DDP ──────────────────────────────────────────
    encoder_ddp = DDP(encoder, device_ids=[local_rank], find_unused_parameters=True)
    decoder_ddp = DDP(decoder, device_ids=[local_rank], find_unused_parameters=True)

    # ── Helper: build / rebuild optimizer for current trainable params ────────
    def _build_optimizer(enc, dec, enc_lr, dec_lr):
        enc_p = [p for p in enc.parameters() if p.requires_grad]
        dec_p = [p for p in dec.parameters() if p.requires_grad]
        all_p = enc_p + dec_p
        opt = optim.AdamW([
            {"params": enc_p, "lr": enc_lr},
            {"params": dec_p, "lr": dec_lr},
        ], weight_decay=0.01, fused=True)
        return opt, all_p, enc_p, dec_p

    optimizer, trainable_params, encoder_params, decoder_params = \
        _build_optimizer(encoder_ddp, decoder_ddp, ENCODER_LR, LR)

    if is_main():
        logger.info(f"Phase 1 (frozen ViT) — encoder {sum(p.numel() for p in encoder_params):,}  "
                     f"decoder {sum(p.numel() for p in decoder_params):,}")

    contrastive_fn   = ContrastiveLoss(temperature=0.07).to(device)
    vicreg_fn        = VICRegLoss(var_weight=25.0, cov_weight=1.0, var_target=1.0).to(device)
    freq_fn          = FrequencyLoss().to(device)
    # Distribution-matching orientation loss (ODF + two-point misorientation
    # statistics); spatially unaligned -> rewards statistical similarity.
    # Track B (orient_v2): metric-aware terms (Sinkhorn-ODF, pole-figure,
    # scatter anti-shrinkage) + stronger/earlier schedule; falls back to the
    # legacy configuration when config.json has no training.fmdit.orient_v2.
    _V2 = FMDIT_ORIENT_V2
    if _V2:
        orient_fn = OrientationDistributionLoss(
            odf_weight=_V2.get("odf_weight", 0.3),
            misori_weight=_V2.get("misori_weight", 0.2),
            pf_weight=_V2.get("pf_weight", 0.4),
            scatter_weight=_V2.get("scatter_weight", 0.1),
            n_refs=int(_V2.get("n_refs", 128)),
            kappa=float(_V2.get("kappa", 150.0)),
            n_pix=int(_V2.get("n_pix", 4096)),
            odf_mode=_V2.get("odf_mode", "sinkhorn"),
            sinkhorn_eps=float(_V2.get("sinkhorn_eps", 0.05)),
            sinkhorn_iters=int(_V2.get("sinkhorn_iters", 30)),
            pf_grid=int(_V2.get("pf_grid", 256)),
            pf_kappa=float(_V2.get("pf_kappa", 64.0)),
            pf_prismatic=bool(_V2.get("pf_prismatic", True)),
            eps=float(_V2.get("eps", FMDIT_ORIENT_EPS)),
            # v3 terms (0.0 -> off, keeps v2 configs working unchanged)
            pix_weight=float(_V2.get("pix_weight", 0.0)),
            pf_sharp_weight=float(_V2.get("pf_sharp_weight", 0.0)),
            mmd_kappa=float(_V2.get("mmd_kappa", 64.0)),
            mmd_n=int(_V2.get("mmd_n", 512)),
        ).to(device)
        ORIENT_LAMBDA        = float(_V2.get("lambda_orient", 1.0))
        ORIENT_WARMUP        = int(_V2.get("warmup_epochs", 2))
        ORIENT_RAMP          = int(_V2.get("ramp_epochs", 3))
        ORIENT_WEIGHT_POWER  = float(_V2.get("orient_weight_power", 1.0))
        LAMBDA_LATSTAT       = float(_V2.get("lambda_latstat", 0.05))
        ORIENT_T_THRESHOLD   = _V2.get("orient_t_threshold", None)
        if ORIENT_T_THRESHOLD is not None:
            ORIENT_T_THRESHOLD = float(ORIENT_T_THRESHOLD)
        if is_main():
            logger.info(f"[orient_v2] ACTIVE: lambda={ORIENT_LAMBDA} warmup={ORIENT_WARMUP} "
                        f"ramp={ORIENT_RAMP} weights(pf/odf/mis/scatter)="
                        f"{_V2.get('pf_weight',0.4)}/{_V2.get('odf_weight',0.3)}/"
                        f"{_V2.get('misori_weight',0.2)}/{_V2.get('scatter_weight',0.1)} "
                        f"odf_mode={_V2.get('odf_mode','sinkhorn')} latstat={LAMBDA_LATSTAT}")
    else:
        orient_fn = OrientationDistributionLoss(
            eps=FMDIT_ORIENT_EPS,
        ).to(device)
        ORIENT_LAMBDA        = FMDIT_LAMBDA_ORIENT
        ORIENT_WARMUP        = FMDIT_ORIENT_WARMUP_EPOCHS
        ORIENT_RAMP          = FMDIT_ORIENT_RAMP_EPOCHS
        ORIENT_WEIGHT_POWER  = 2.0
        LAMBDA_LATSTAT       = 0.0
        ORIENT_T_THRESHOLD   = None
    embedding_analyzer = EmbeddingAnalyzer()
    loss_scheduler   = LossWeightScheduler(
        num_epochs=FMDIT_NUM_EPOCHS,
        lambda_diff_start=1.0,        lambda_diff_end=1.0,   # fixed: ramping was cosmetic (vel plateaued anyway)
        lambda_recon_start=0.5,       lambda_recon_end=0.3,
        lambda_contrastive_start=0.3, lambda_contrastive_end=0.05,
        lambda_vicreg=0.05,
        lambda_freq=0.3,
        lambda_orient=ORIENT_LAMBDA,
        warmup_epochs=max(1, FMDIT_NUM_EPOCHS // 20),  # contrastive
        warmup_epochs_vicreg=0,                          # regularizer active from epoch 0 (no reason to delay)
        warmup_epochs_freq=4,                            # delay: silent epochs 0-3, ramp 4-6, full from 7 (display Ep5→Ep8)
        warmup_epochs_orient=ORIENT_WARMUP,              # delay: silent until decoder produces grain-like images
        ramp_epochs_vicreg=0,                            # full weight immediately
        ramp_epochs_freq=3,                              # 3-epoch ramp to full
        ramp_epochs_orient=ORIENT_RAMP,                  # ramp to full orientation weight
        contrastive_cutoff_epoch=UNFREEZE_EPOCH + 4,     # solved (~0.001) by then; saves the extra 350M-param encoder forward
    )

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, len(train_loader) * FMDIT_NUM_EPOCHS - WARMUP_STEPS), eta_min=1e-6,
    )
    curr_step    = 0
    # LR warmup window: linear ramp over [warmup_start_step, warmup_start_step + WARMUP_STEPS).
    # Reset at the Phase-2 optimizer rebuild so fresh Adam moments don't meet
    # full LR head-on (caused the epoch-9 loss spike in run_20260609_160312).
    warmup_start_step = 0
    start_epoch  = 0
    best_val_loss= float('inf')

    if RESUME_FROM and os.path.exists(RESUME_FROM):
        ckpt = torch.load(RESUME_FROM, map_location="cpu")
        if "encoder" in ckpt:
            encoder.load_state_dict(ckpt["encoder"])
            if is_main():
                logger.info("Loaded encoder weights from checkpoint.")
        _missing, _unexpected = decoder.load_state_dict(ckpt["decoder"], strict=False)
        if is_main() and (_missing or _unexpected):
            logger.info(f"decoder resume: missing={_missing} unexpected={_unexpected}")

        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float('inf'))

        # If resuming into Phase 2, unfreeze blocks & rebuild optimizer first
        if start_epoch >= UNFREEZE_EPOCH or start_epoch >= LORA_START_EPOCH:
            if start_epoch >= UNFREEZE_EPOCH:
                encoder.unfreeze_vit_blocks(last_n=16)
            if start_epoch >= LORA_START_EPOCH and decoder.lora_enabled:
                _set_lora_trainable(decoder, True)
            encoder_ddp = _wrap_ddp(encoder)
            decoder_ddp = _wrap_ddp(decoder)
            optimizer, trainable_params, encoder_params, decoder_params = \
                _build_optimizer(encoder_ddp, decoder_ddp, ENCODER_LR, LR)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, len(train_loader) * (FMDIT_NUM_EPOCHS - min(UNFREEZE_EPOCH, LORA_START_EPOCH))),
                eta_min=1e-6,
            )
            if is_main():
                logger.info(f"Resume into Phase 2 — unfroze last 16 ViT blocks | "
                             f"encoder {sum(p.numel() for p in encoder_params):,}  "
                             f"decoder {sum(p.numel() for p in decoder_params):,}")

        if os.environ.get("FMDIT_RESUME_FRESH_OPT", "0") == "1":
            # C1/C3 fine-tune: param set changed (tex_head) and objective
            # changed (cond dropout) -> start a fresh optimizer/schedule.
            if is_main():
                logger.info("FMDIT_RESUME_FRESH_OPT=1: skipping optimizer/scheduler state restore")
        else:
            optimizer.load_state_dict(ckpt["optimizer"])
            if "lr_scheduler" in ckpt:
                lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            if "curr_step" in ckpt:
                curr_step = ckpt["curr_step"]
        if is_main():
            logger.info(f"Resumed from epoch {start_epoch}")

    if is_main():
        logger.info(
            f"TRAINING  |  epochs {start_epoch + 1} → {FMDIT_NUM_EPOCHS}"
            f"  |  batch_size {BATCH_SIZE}×{world_size}GPUs  |  accum {ACCUMULATION_STEPS}"
        )

    for epoch in range(start_epoch, FMDIT_NUM_EPOCHS):
        train_sampler.set_epoch(epoch)  # reshuffle each epoch for DDP

        # ── Phase 2: unfreeze late ViT blocks ────────────────────────────
        if epoch == UNFREEZE_EPOCH or epoch == LORA_START_EPOCH:
            if epoch == UNFREEZE_EPOCH:
                encoder.unfreeze_vit_blocks(last_n=16)
            if epoch == LORA_START_EPOCH and decoder.lora_enabled:
                _nl = _set_lora_trainable(decoder, True)
                if is_main():
                    logger.info(f'Phase 2 - enabled MMDiT LoRA ({_nl:,} params)')
            encoder_ddp = _wrap_ddp(encoder)
            decoder_ddp = _wrap_ddp(decoder)
            optimizer, trainable_params, encoder_params, decoder_params = \
                _build_optimizer(encoder_ddp, decoder_ddp, ENCODER_LR, LR)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, len(train_loader) * (FMDIT_NUM_EPOCHS - min(UNFREEZE_EPOCH, LORA_START_EPOCH))),
                eta_min=1e-6,
            )
            warmup_start_step = curr_step   # re-warmup LR for the fresh optimizer
            if is_main():
                logger.info(f"Phase 2 — unfroze last 16 ViT blocks | "
                             f"encoder {sum(p.numel() for p in encoder_params):,}  "
                             f"decoder {sum(p.numel() for p in decoder_params):,}")
                logger.info(f"Phase 2 — LR re-warmup over {WARMUP_STEPS} steps")

        loss_weights = loss_scheduler.get_weights(epoch)
        if is_main():
            loss_scheduler.log_weights(epoch, logger)
        decoder_ddp.train()
        encoder_ddp.train()

        total_loss = 0.0
        total_vel = 0.0
        total_recon = 0.0
        total_contr = 0.0
        total_vicreg = 0.0
        total_freq = 0.0
        total_orient = 0.0
        total_osub = {"odf": 0.0, "misori": 0.0, "pf": 0.0, "scatter": 0.0, "latstat": 0.0,
                      "pix": 0.0, "pfsharp": 0.0}
        batch_count = 0

        num_batches = len(train_loader)
        optimizer.zero_grad()

        for batch_idx, images in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    z_512 = encoder_ddp(images)

                    if loss_weights["lambda_contrastive"] > 0:
                        pixel_noise   = torch.randn_like(images) * PIXEL_NOISE_SCALE
                        images_aug    = (images + pixel_noise).clamp(-1.0, 1.0)
                        z_512_aug     = encoder_ddp(images_aug)
                    else:
                        z_512_aug = z_512

                    with torch.no_grad():
                        x_0 = vae.encode(images)

                    loss, info = composite_fm_loss(
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
                        freq_t_threshold=FMDIT_PIXEL_T_THRESHOLD,
                        freq_max_samples=int(os.environ.get(
                            "FMDIT_PIXEL_MAX_SAMPLES_OVERRIDE", FMDIT_PIXEL_MAX_SAMPLES)),
                        orient_fn=orient_fn,
                        lambda_orient=loss_weights["lambda_orient"],
                        lambda_latstat=LAMBDA_LATSTAT,
                        orient_weight_power=ORIENT_WEIGHT_POWER,
                        orient_t_threshold=ORIENT_T_THRESHOLD,
                        cond_dropout=CFG_DROPOUT,
                        lambda_texhead=TEXHEAD_LAMBDA,
                    )
                    loss = loss / ACCUMULATION_STEPS

                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    continue

                loss.backward()

                if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
                    optimizer.step()
                    optimizer.zero_grad()
                    if curr_step - warmup_start_step < WARMUP_STEPS:
                        warmup_frac = (curr_step - warmup_start_step) / max(1, WARMUP_STEPS)
                        optimizer.param_groups[0]["lr"] = ENCODER_LR * warmup_frac
                        optimizer.param_groups[1]["lr"] = LR * warmup_frac
                    else:
                        lr_scheduler.step()
                    curr_step += 1

                total_loss   += loss.item() * ACCUMULATION_STEPS
                total_vel    += info["loss_velocity"]
                total_recon  += info["loss_recon"]
                total_contr  += info["loss_contrastive"]
                total_vicreg += info["loss_vicreg"]
                total_freq   += info["loss_freq"]
                total_orient += info["loss_orient"]
                total_osub["odf"]     += info.get("loss_orient_odf", 0.0)
                total_osub["misori"]  += info.get("loss_orient_misori", 0.0)
                total_osub["pf"]      += info.get("loss_orient_pf", 0.0)
                total_osub["scatter"] += info.get("loss_orient_scatter", 0.0)
                total_osub["latstat"] += info.get("loss_latstat", 0.0)
                total_osub["pix"]     += info.get("loss_orient_pix", 0.0)
                total_osub["pfsharp"] += info.get("loss_orient_pfsharp", 0.0)
                batch_count += 1

                if is_main():
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"{ts} | Epoch {epoch+1}/{FMDIT_NUM_EPOCHS} [{batch_idx}/{num_batches}] | "
                        f"loss={total_loss/max(1,batch_count):.4f} vel={info['loss_velocity']:.4f} "
                        f"recon={info['loss_recon']:.4f} contr={info['loss_contrastive']:.4f} "
                        f"vicreg={info['loss_vicreg']:.4f} freq={info['loss_freq']:.4f} "
                        f"orient={info['loss_orient']:.4f} "
                        f"latstat={info.get('loss_latstat', 0.0):.4f} "
                        f"o_pf={info.get('loss_orient_pf', 0.0):.4f} "
                        f"o_odf={info.get('loss_orient_odf', 0.0):.4f} "
                        f"o_sc={info.get('loss_orient_scatter', 0.0):.4f} "
                        f"lr_enc={optimizer.param_groups[0]['lr']:.2e} "
                        f"lr_dec={optimizer.param_groups[1]['lr']:.2e}",
                        flush=True
                    )

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if is_main():
                        logger.warning(f"OOM at batch {batch_idx}, clearing cache …")
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                raise

        avg_loss   = total_loss   / max(1, batch_count)
        avg_vel    = total_vel    / max(1, batch_count)
        avg_recon  = total_recon  / max(1, batch_count)
        avg_contr  = total_contr  / max(1, batch_count)
        avg_vicreg = total_vicreg / max(1, batch_count)
        avg_freq   = total_freq   / max(1, batch_count)
        avg_orient = total_orient / max(1, batch_count)
        avg_osub   = {k: v / max(1, batch_count) for k, v in total_osub.items()}

        if is_main():
            logger.info(
                f"Epoch {epoch + 1}/{FMDIT_NUM_EPOCHS} TRAIN | "
                f"Loss: {avg_loss:.4f} | Vel: {avg_vel:.4f} | Recon: {avg_recon:.4f} | "
                f"Contr: {avg_contr:.4f} | VICReg: {avg_vicreg:.4f} | Freq: {avg_freq:.4f} | "
                f"Orient: {avg_orient:.4f}"
            )

        # Validation on rank 0 only
        # All ranks hit this barrier to sync after training
        dist.barrier()

        val_loss = 0.0
        sample_psnr = 0.0
        if is_main():
            if val_loader:
                val_loss = validate(encoder_ddp, decoder_ddp, val_loader, vae, device)
                logger.info(f"Epoch {epoch + 1}/{FMDIT_NUM_EPOCHS} VAL   | Loss: {val_loss:.4f}")

            torch.cuda.empty_cache()
            # Fewer samples in early epochs (decoder output is noise anyway)
            n_samp = 3 if epoch < 3 else 10
            sample_psnr = generate_epoch_samples(
                epoch + 1, encoder_ddp, decoder_ddp, vae, full_dataset, train_dataset, device, n_samples=n_samp
            )

            # ── Embedding-space health (collapse monitor) ───────────────────
            try:
                encoder.eval()
                emb_src = val_loader if val_loader else train_loader
                with torch.no_grad():
                    emb_imgs = next(iter(emb_src)).to(device, non_blocking=True)
                    z_dbg = encoder(emb_imgs).float()
                emb_metrics = embedding_analyzer.compute_metrics(z_dbg)
                embedding_analyzer.log_metrics(emb_metrics, logger)
                encoder.train()
            except Exception as e:
                logger.warning(f"Embedding analysis failed: {e}")

        log_to_csv(epoch + 1, avg_loss, avg_vel, avg_recon, avg_contr,
                   avg_vicreg, avg_freq, avg_orient, val_loss, sample_psnr,
                   orient_sub=avg_osub)

        # ── Checkpointing (rank 0 only) ──────────────────────────────────────
        if is_main():
            is_best = val_loss < best_val_loss if val_loader else True
            if is_best and val_loader:
                best_val_loss = val_loss

            if FMDIT_SAVE_EVERY_EPOCH or is_best:
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
                    "lr_scheduler":  lr_scheduler.state_dict(),
                    "curr_step":     curr_step,
                    "best_val_loss": best_val_loss,
                }, ckpt_path)
                logger.info(f"Checkpoint saved: {ckpt_path}")
                cleanup_old_checkpoints()

        # All ranks wait for rank-0 to finish val/samples/checkpoint
        dist.barrier()

    cleanup_ddp()


if __name__ == "__main__":
    train()
