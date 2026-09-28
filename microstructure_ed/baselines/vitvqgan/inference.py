"""
Inference script for PaintMind ViT-VQGAN
Save ONLY reconstructed images (no comparison plots)
"""

import os
from pathlib import Path
from tqdm import tqdm

import torch
from PIL import Image
from torchvision import transforms

import paintmind as pm


# ============================================================
# CONFIG
# ============================================================
IMAGE_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIR = os.environ.get("VQGAN_TEST_DIR", "dataset_test")
OUTPUT_DIR = "reconstruction_results"
CHECKPOINT_PATH = "checkpoints/epoch_040.pth"


# ============================================================
# HELPERS
# ============================================================
def tensor_to_pil(x):
    """
    Convert tensor [-1, 1] -> PIL image
    """
    x = x.detach().cpu().clamp(-1, 1)
    x = (x + 1) / 2  # [-1,1] -> [0,1]
    x = (x * 255).byte()
    x = x.permute(1, 2, 0).numpy()
    return Image.fromarray(x)


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# ============================================================
# MAIN
# ============================================================
def main():
    print("🚀 Loading PaintMind ViT-VQGAN...")

    model = pm.create_model(
        arch="vqgan",
        version="vit-s-vqgan",
        pretrained=True
    ).to(DEVICE)

    print(f"📂 Loading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    # Your training script saves with key: "model"
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")

    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        list(input_dir.glob("*.png")) +
        list(input_dir.glob("*.jpg")) +
        list(input_dir.glob("*.jpeg"))
    )

    print(f"📁 Found {len(files)} images")
    print("🎨 Starting reconstruction...\n")

    with torch.no_grad():
        for file in tqdm(files):
            # Load original image
            img = Image.open(file).convert("RGB")

            # Transform → tensor
            image = transform(img).unsqueeze(0).to(DEVICE)

            # Encode → Decode
            z, _, _ = model.encode(image)
            recon = model.decode(z)

            # Convert reconstructed tensor to image
            recon_pil = tensor_to_pil(recon[0])

            # Save with original filename
            save_path = output_dir / file.name
            recon_pil.save(save_path)

    print("\n✅ Reconstruction complete!")
    print(f"Saved reconstructed images to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()