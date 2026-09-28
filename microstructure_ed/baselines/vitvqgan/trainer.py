import os
import random
import csv
import gc
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

import lpips
import paintmind as pm

# =========================
# IMPORT YOUR UTILITIES
# =========================
from microstructure_ed.baselines.vitvqgan.logger_utils import TrainingLogger, save_comparison
from microstructure_ed.baselines.vitvqgan.metrics import compute_psnr

# ============================================================================  
# CONFIG  
# ============================================================================  
IMAGE_SIZE = 256
NUM_EPOCHS = 50
BATCH_SIZE = 64
NUM_WORKERS = 2
LR = 1e-5

RECON_WEIGHT = 1.0
PERCEPTUAL_WEIGHT = 0.1
CODEBOOK_WEIGHT = 1.0

USE_DISCRIMINATOR = True
DISC_START_EPOCH = 10
DISC_WEIGHT = 0.3

SAVE_EVERY = 1

INPUT_DIR = os.environ.get("VQGAN_TRAIN_DIR", "dataset_train")
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
SAMPLE_DIR = os.path.join(LOG_DIR, "samples")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================  
# HELPERS  
# ============================================================================  
def tensor_to_pil(x):
    x = x.detach().cpu().clamp(-1, 1)
    x = (x + 1) / 2
    x = (x * 255).byte()
    x = x.permute(1, 2, 0).numpy()
    return Image.fromarray(x)

# ============================================================================  
# DATASET  
# ============================================================================  
class MicroDataset(Dataset):
    def __init__(self, folder, size=256):
        self.files = sorted(
            list(Path(folder).glob("*.png")) +
            list(Path(folder).glob("*.jpg"))
        )

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.transform(
            Image.open(self.files[i]).convert("RGB")
        )

    def get_orig(self, i):
        return Image.open(self.files[i]).convert("RGB"), self.files[i].name

# ============================================================================  
# DISCRIMINATOR  
# ============================================================================  
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 1, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.model(x)

# ============================================================================  
# MAIN  
# ============================================================================  
def main():

    print("🚀 Loading pretrained ViT-VQGAN...")
    model = pm.create_model(
        arch="vqgan",
        version="vit-s-vqgan",
        pretrained=True
    ).to(device)

    disc = Discriminator().to(device) if USE_DISCRIMINATOR else None

    dataset = MicroDataset(INPUT_DIR, IMAGE_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    opt_model = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.9))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=LR, betas=(0.5, 0.9)) if disc else None

    perceptual = lpips.LPIPS(net='vgg').to(device).eval()

    # =========================
    # LOGGER INIT
    # =========================
    logger = TrainingLogger(os.path.join(LOG_DIR, "training.csv"))

    print(f"Dataset size: {len(dataset)} images")

    # ============================================================================  
    # TRAINING LOOP  
    # ============================================================================  
    for epoch in range(NUM_EPOCHS):

        model.train()
        if disc:
            disc.train()

        use_disc = USE_DISCRIMINATOR and (epoch >= DISC_START_EPOCH)

        ep_recon, ep_code, ep_perc, ep_disc = 0, 0, 0, 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for real in pbar:
            real = real.to(device)

            # ---------------- MODEL ----------------
            opt_model.zero_grad()

            z, code_loss, _ = model.encode(real)
            recon = model.decode(z)

            l_recon = F.l1_loss(recon, real)
            l_perc = perceptual(recon, real).mean()

            loss = (
                RECON_WEIGHT * l_recon +
                PERCEPTUAL_WEIGHT * l_perc +
                CODEBOOK_WEIGHT * code_loss
            )

            l_disc = torch.tensor(0.0, device=device)

            if use_disc:
                logits_fake = disc(recon)
                l_disc = -logits_fake.mean()
                loss += DISC_WEIGHT * l_disc

            loss.backward()
            opt_model.step()

            # ---------------- DISCRIMINATOR ----------------
            if use_disc:
                opt_disc.zero_grad()

                logits_real = disc(real.detach())
                logits_fake = disc(recon.detach())

                d_loss = logits_fake.mean() - logits_real.mean()

                alpha = torch.rand(real.size(0), 1, 1, 1).to(device)
                interp = (alpha * real + (1 - alpha) * recon.detach()).requires_grad_(True)

                d_interp = disc(interp)
                grads = torch.autograd.grad(
                    d_interp,
                    interp,
                    torch.ones_like(d_interp),
                    create_graph=True,
                    retain_graph=True
                )[0]

                gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
                d_loss = d_loss + 10 * gp

                d_loss.backward()
                opt_disc.step()

            ep_recon += l_recon.item()
            ep_code += code_loss.item()
            ep_perc += l_perc.item()
            ep_disc += l_disc.item()

            pbar.set_postfix({
                "R": f"{l_recon:.3f}",
                "P": f"{l_perc:.3f}"
            })

        # ========================================================================
        # EVALUATION
        # ========================================================================
        model.eval()

        idx = random.randint(0, len(dataset) - 1)

        orig_pil, fname = dataset.get_orig(idx)
        test = dataset[idx].unsqueeze(0).to(device)

        with torch.no_grad():
            z_test, _, _ = model.encode(test)
            rec = model.decode(z_test)

        sample_psnr = compute_psnr(test, rec)

        # Save comparison image
        save_comparison(
            original_pil=orig_pil,
            reconstructed_tensor=rec[0],
            epoch=epoch + 1,
            psnr_val=sample_psnr,
            filename=fname,
            save_dir=SAMPLE_DIR
        )

        # ========================================================================
        # LOGGING
        # ========================================================================
        metrics = {
            "recon_loss": ep_recon / len(loader),
            "codebook_loss": ep_code / len(loader),
            "perceptual_loss": ep_perc / len(loader),
            "disc_loss": ep_disc / len(loader),
            "psnr": sample_psnr
        }

        logger.log_epoch(epoch + 1, metrics)

        print(f"\nEpoch {epoch+1} | PSNR: {sample_psnr:.2f} dB")

        # ========================================================================
        # CHECKPOINT
        # ========================================================================
        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "disc": disc.state_dict() if disc else None
            }, f"{CHECKPOINT_DIR}/epoch_{epoch+1:03d}.pth")

        torch.cuda.empty_cache()

    print("\n🎉 Training complete!")


if __name__ == "__main__":
    main()