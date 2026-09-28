"""
Logging and visualization utilities
"""

import csv
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import pandas as pd


# ============================================================
# IMAGE CONVERSION (FIXED - NO CIRCULAR IMPORTS)
# ============================================================
def tensor_to_pil(x):
    """
    Convert torch tensor [-1, 1] -> PIL Image
    """
    x = x.detach().cpu().clamp(-1, 1)
    x = (x + 1) / 2  # [-1,1] -> [0,1]
    x = (x * 255).byte()
    x = x.permute(1, 2, 0).numpy()
    return Image.fromarray(x)


# ============================================================
# TRAINING LOGGER
# ============================================================
class TrainingLogger:
    """Logger for training metrics"""

    def __init__(self, log_file):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Create CSV with headers
        if not self.log_file.exists():
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch",
                    "recon_loss",
                    "codebook_loss",
                    "perceptual_loss",
                    "disc_loss",
                    "psnr",
                    "fid",
                    "perplexity"
                ])

    def log_epoch(self, epoch, metrics):
        """Log metrics for an epoch"""

        fid = metrics.get("fid", None)

        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{metrics.get('recon_loss', 0):.6f}",
                f"{metrics.get('codebook_loss', 0):.6f}",
                f"{metrics.get('perceptual_loss', 0):.6f}",
                f"{metrics.get('disc_loss', 0):.6f}",
                f"{metrics.get('psnr', 0):.2f}",
                f"{fid:.2f}" if fid is not None else "",
                f"{metrics.get('perplexity', 0):.1f}"
            ])


# ============================================================
# VISUALIZATION
# ============================================================
def save_comparison(original_pil,
                    reconstructed_tensor,
                    epoch,
                    psnr_val,
                    filename,
                    save_dir):
    """
    Save side-by-side comparison of original and reconstruction
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Original
    axes[0].imshow(original_pil)
    axes[0].set_title(f"Original: {filename}", fontsize=10)
    axes[0].axis("off")

    # Reconstructed
    axes[1].imshow(tensor_to_pil(reconstructed_tensor))
    axes[1].set_title(f"Generated | PSNR: {psnr_val:.2f} dB", fontsize=10)
    axes[1].axis("off")

    plt.suptitle(f"Epoch {epoch} - ViT-VQGAN", fontsize=12, fontweight="bold")
    plt.tight_layout()

    save_path = save_dir / f"epoch_{epoch:03d}.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()

    return save_path


# ============================================================
# TRAINING CURVES
# ============================================================
def plot_training_curves(log_file, save_path=None):
    """
    Plot training curves from CSV log
    """

    df = pd.read_csv(log_file)

    # Ensure numeric fid (fixes empty string issue)
    df["fid"] = pd.to_numeric(df["fid"], errors="coerce")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ---------------- Losses ----------------
    axes[0].plot(df["epoch"], df["recon_loss"], lw=2, label="Reconstruction")
    axes[0].plot(df["epoch"], df["codebook_loss"], lw=2, label="Codebook")
    axes[0].plot(df["epoch"], df["perceptual_loss"], lw=2, label="Perceptual")
    axes[0].set_title("Training Losses")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ---------------- PSNR ----------------
    axes[1].plot(df["epoch"], df["psnr"], lw=2, marker="o")
    axes[1].axhline(30, ls="--", alpha=0.5)
    axes[1].axhline(35, ls="--", alpha=0.5)
    axes[1].set_title("PSNR (Higher is Better)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("dB")
    axes[1].grid(True, alpha=0.3)

    # ---------------- FID ----------------
    fid_df = df.dropna(subset=["fid"])

    if len(fid_df) > 0:
        axes[2].plot(fid_df["epoch"], fid_df["fid"], lw=2, marker="s")
        axes[2].set_title("FID (Lower is Better)")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("FID")
        axes[2].invert_yaxis()
        axes[2].grid(True, alpha=0.3)

    plt.suptitle("ViT-VQGAN Training Curves", fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


# ============================================================
# CHECKPOINTING
# ============================================================
def save_checkpoint(model,
                    discriminator,
                    optimizer_model,
                    optimizer_disc,
                    epoch,
                    metrics,
                    checkpoint_path):

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_model_state_dict": optimizer_model.state_dict(),
        "metrics": metrics
    }

    if discriminator is not None:
        checkpoint["discriminator_state_dict"] = discriminator.state_dict()

    if optimizer_disc is not None:
        checkpoint["optimizer_disc_state_dict"] = optimizer_disc.state_dict()

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path,
                    model,
                    discriminator=None,
                    optimizer_model=None,
                    optimizer_disc=None):

    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])

    if discriminator is not None and "discriminator_state_dict" in checkpoint:
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

    if optimizer_model is not None:
        optimizer_model.load_state_dict(checkpoint["optimizer_model_state_dict"])

    if optimizer_disc is not None and "optimizer_disc_state_dict" in checkpoint:
        optimizer_disc.load_state_dict(checkpoint["optimizer_disc_state_dict"])

    return checkpoint["epoch"], checkpoint.get("metrics", {})