"""
Metrics for evaluating reconstruction quality
"""

import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from tqdm import tqdm


def compute_psnr(img1, img2):
    """
    Compute Peak Signal-to-Noise Ratio
    
    Args:
        img1, img2: Tensors in range [-1, 1]
        
    Returns:
        PSNR in dB
    """
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float('inf')
    
    # Max pixel value is 2.0 for range [-1, 1]
    psnr = 20 * np.log10(2.0 / np.sqrt(mse))
    return psnr


def compute_fid(model, dataset, num_samples=500, batch_size=50, device='cuda'):
    """
    Compute Fréchet Inception Distance
    
    Args:
        model: Trained ViT-VQGAN model
        dataset: Dataset to sample from
        num_samples: Number of samples to use
        batch_size: Batch size for generation
        device: Device to use
        
    Returns:
        FID score (lower is better)
    """
    try:
        from pytorch_fid import fid_score
        from .dataset import tensor_to_pil
    except ImportError:
        print("⚠️ pytorch-fid not installed. Install with: pip install pytorch-fid")
        return None
    
    model.eval()
    
    # Create temporary directories
    real_dir = tempfile.mkdtemp()
    fake_dir = tempfile.mkdtemp()
    
    try:
        # Save real images
        print(f"📊 Saving {num_samples} real images...")
        indices = np.random.choice(
            len(dataset), 
            min(num_samples, len(dataset)), 
            replace=False
        )
        
        for i, idx in enumerate(tqdm(indices, desc="Real images", leave=False)):
            img, _ = dataset.get_original(int(idx))
            # Resize to 299×299 (Inception input size)
            img = img.resize((299, 299))
            img.save(f"{real_dir}/{i:04d}.png")
        
        # Generate fake images
        print(f"🎨 Generating {num_samples} fake images...")
        num_generated = 0
        
        with torch.no_grad():
            pbar = tqdm(total=num_samples, desc="Generating", leave=False)
            
            while num_generated < num_samples:
                # Sample random batch
                batch_size_curr = min(batch_size, num_samples - num_generated)
                batch_indices = np.random.choice(len(dataset), batch_size_curr)
                batch = torch.stack([dataset[int(i)] for i in batch_indices]).to(device)
                
                # Generate reconstructions
                z_q, _, _, _ = model.encode(batch)
                recons = model.decode(z_q)
                
                # Save
                for recon in recons:
                    img = tensor_to_pil(recon)
                    img = img.resize((299, 299))
                    img.save(f"{fake_dir}/{num_generated:04d}.png")
                    num_generated += 1
                    pbar.update(1)
                    
                    if num_generated >= num_samples:
                        break
            
            pbar.close()
        
        # Calculate FID
        print("🔢 Computing FID score...")
        fid_value = fid_score.calculate_fid_given_paths(
            [real_dir, fake_dir],
            batch_size=50,
            device=device,
            dims=2048
        )
        
        return fid_value
        
    finally:
        # Cleanup
        shutil.rmtree(real_dir)
        shutil.rmtree(fake_dir)


def compute_codebook_usage(model, dataset, num_batches=100, batch_size=8, device='cuda'):
    """
    Compute codebook usage statistics
    
    Args:
        model: ViT-VQGAN model
        dataset: Dataset
        num_batches: Number of batches to process
        batch_size: Batch size
        device: Device
        
    Returns:
        usage_dict: Dictionary with usage statistics
    """
    model.eval()
    
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    code_counts = torch.zeros(model.quantize.n_embed).to(device)
    total_codes = 0
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            
            batch = batch.to(device)
            _, _, indices, _ = model.encode(batch)
            
            # Count code usage
            unique, counts = torch.unique(indices, return_counts=True)
            code_counts[unique] += counts.float()
            total_codes += indices.numel()
    
    # Compute statistics
    used_codes = (code_counts > 0).sum().item()
    usage_percent = 100 * used_codes / model.quantize.n_embed
    avg_usage = code_counts[code_counts > 0].mean().item() if used_codes > 0 else 0
    
    return {
        'total_codes': model.quantize.n_embed,
        'used_codes': used_codes,
        'usage_percent': usage_percent,
        'avg_usage_per_code': avg_usage,
        'total_tokens': total_codes
    }
