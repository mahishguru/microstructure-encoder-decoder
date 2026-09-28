"""
Utilities for learning high-quality embeddings:
- Contrastive learning loss (InfoNCE)
- VICReg loss (variance + covariance regularization)
- Lab color loss (perceptually uniform)
- Frequency loss (FFT magnitude)
- Embedding quality metrics
- Embedding space visualization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List


class ContrastiveLoss(nn.Module):
    """
    InfoNCE / NT-Xent contrastive loss for learning discriminative embeddings.

    Pulls together perturbed views of the same image (positive pairs),
    pushes apart different images (negative pairs).

    Uses latent-space noise injection (not data augmentation) to create
    positive pairs, preserving physical constraints of microstructure images.

    Used to ensure embeddings are not only good at reconstruction,
    but also discriminative (similar images → similar embeddings).
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (2B, D) where first B are original, next B are perturbed
                       Each pair (i, i+B) are from same image (with/without latent noise)

        Returns:
            loss: scalar contrastive loss
        """
        B = embeddings.shape[0] // 2
        device = embeddings.device

        # Normalize embeddings to unit sphere (cosine similarity)
        embeddings = F.normalize(embeddings, dim=-1)  # (2B, D)

        # Compute similarity matrix: (2B, 2B)
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature

        # InfoNCE loss
        # For each sample: log(exp(pos_sim) / sum(exp(all_sims_except_self)))
        exp_sim = torch.exp(similarity)

        # Denominator: sum all exp-similarities, subtract self-similarity (diagonal)
        denom = exp_sim.sum(dim=1, keepdim=True) - torch.exp(torch.diagonal(similarity)).unsqueeze(1)

        # Positive pair index: sample i pairs with i+B, sample i+B pairs with i
        # Use advanced indexing so gradients flow properly through pos_similarities
        N = 2 * B
        row_idx = torch.arange(N, device=device)
        col_idx = torch.cat([torch.arange(B, N, device=device), torch.arange(B, device=device)])
        pos_similarities = similarity[row_idx, col_idx]  # (2B,) — tracks gradient

        # InfoNCE: log(exp(pos) / denom)
        log_prob = pos_similarities - torch.log(denom.squeeze() + 1e-8)

        # Average over all samples
        loss = -log_prob.mean()

        return loss


class VICRegLoss(nn.Module):
    """
    VICReg-style variance + covariance regularization on embeddings.

    Directly addresses dimensional collapse (e.g. effective_dim=37/768).

    Variance term: hinge loss forcing std(z_i) >= var_target per dimension.
        Prevents dimensions from collapsing to zero variance.

    Covariance term: penalizes off-diagonal entries of the covariance matrix.
        Decorrelates dimensions so each carries unique information.

    Reference: Bardes et al., "VICReg: Variance-Invariance-Covariance
    Regularization for Self-Supervised Learning" (ICLR 2022)
    """

    def __init__(
        self,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        var_target: float = 1.0,
    ):
        super().__init__()
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.var_target = var_target

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, D) — batch of embeddings

        Returns:
            scalar loss = var_weight * var_loss + cov_weight * cov_loss
        """
        B, D = z.shape

        # ── Variance: hinge loss per dimension ────────────────────────────────
        # Force std(z_i) >= var_target for each dimension i
        std_z = torch.sqrt(z.var(dim=0) + 1e-4)
        var_loss = torch.relu(self.var_target - std_z).mean()

        # ── Covariance: penalize off-diagonal → decorrelate dimensions ────────
        z_centered = z - z.mean(dim=0)
        cov = (z_centered.T @ z_centered) / (B - 1)   # (D, D)
        cov.fill_diagonal_(0.0)                        # zero out diagonal (variance)
        cov_loss = (cov ** 2).sum() / D

        return self.var_weight * var_loss + self.cov_weight * cov_loss


class LabColorLoss(nn.Module):
    """
    Perceptually uniform color loss in CIELAB space.

    Differentiable conversion: RGB [-1,1] → sRGB [0,1] → linear RGB → XYZ → Lab.
    MSE in Lab space penalizes perceptually significant color differences more
    than raw RGB MSE (which weights all channels equally regardless of human
    perception).

    Useful for microstructures where different phases have distinct colors
    and accurate color reproduction is important for material identification.
    """

    def __init__(self):
        super().__init__()
        # sRGB → XYZ transformation matrix (D65 illuminant)
        self.register_buffer(
            'xyz_from_rgb',
            torch.tensor([
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ]).float()
        )
        # D65 white point
        self.register_buffer(
            'white_point',
            torch.tensor([0.95047, 1.00000, 1.08883]).float()
        )

    def _srgb_to_linear(self, x: torch.Tensor) -> torch.Tensor:
        """sRGB companding → linear RGB (inverse gamma)."""
        return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

    def _rgb_to_lab(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Convert RGB [-1,1] → CIELAB.

        Args:
            rgb: (B, 3, H, W) in [-1, 1]

        Returns:
            lab: (B, 3, H, W) — L in [0,100], a/b in approx [-128,127]
        """
        # Cast to float32 — torch.where with power is not stable in BFloat16
        rgb = rgb.float()

        # [-1, 1] → [0, 1]
        srgb = (rgb + 1.0) / 2.0
        srgb = srgb.clamp(0.0, 1.0)

        # sRGB → linear RGB
        linear = self._srgb_to_linear(srgb)

        # linear RGB → XYZ (matrix multiply per pixel)
        # (B, 3, H, W) → (B, H, W, 3) for matmul, then back
        B, C, H, W = linear.shape
        rgb_flat = linear.permute(0, 2, 3, 1)              # (B, H, W, 3)
        xyz = torch.matmul(rgb_flat, self.xyz_from_rgb.T)   # (B, H, W, 3)

        # Normalize by white point
        xyz = xyz / self.white_point                         # (B, H, W, 3)

        # XYZ → Lab (f function)
        delta = 6.0 / 29.0
        delta_sq = delta ** 2
        delta_cu = delta ** 3

        # clamp(min=0) before pow: torch.where evaluates both branches before
        # selecting; matrix-multiply rounding can produce tiny negative xyz values
        # → (-1e-9).pow(1/3) = NaN in float, and NaN gradients propagate even for
        # elements that take the other branch.
        f_xyz = torch.where(
            xyz > delta_cu,
            xyz.clamp(min=0).pow(1.0 / 3.0),
            xyz / (3.0 * delta_sq) + 4.0 / 29.0,
        )

        fx, fy, fz = f_xyz[..., 0], f_xyz[..., 1], f_xyz[..., 2]

        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b = 200.0 * (fy - fz)

        lab = torch.stack([L, a, b], dim=-1)                # (B, H, W, 3)
        lab = lab.permute(0, 3, 1, 2)                       # (B, 3, H, W)

        return lab

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred:   (B, 3, H, W) in [-1, 1] — predicted image (from VAE decode of z0_pred)
            target: (B, 3, H, W) in [-1, 1] — target image (from VAE decode of z0_target)

        Returns:
            scalar MSE loss in Lab space
        """
        lab_pred = self._rgb_to_lab(pred)
        lab_target = self._rgb_to_lab(target)

        # Normalize L to [0,1] range (L is 0-100) and a,b to approx [-1,1]
        # This prevents L channel from dominating the loss
        lab_pred_norm = lab_pred.clone()
        lab_target_norm = lab_target.clone()
        lab_pred_norm[:, 0] = lab_pred[:, 0] / 100.0
        lab_target_norm[:, 0] = lab_target[:, 0] / 100.0
        lab_pred_norm[:, 1:] = lab_pred[:, 1:] / 128.0
        lab_target_norm[:, 1:] = lab_target[:, 1:] / 128.0

        return F.mse_loss(lab_pred_norm, lab_target_norm)


class FrequencyLoss(nn.Module):
    """
    FFT magnitude L1 loss in frequency domain.

    Computes 2D FFT of both predicted and target images, then L1 on the
    magnitude spectra. Captures high-frequency texture patterns (grain
    boundaries, phase interfaces, porosity edges) that pixel-space losses
    often underweight.

    Uses orthonormal FFT normalization for scale-invariant comparison.
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred:   (B, 3, H, W) in [-1, 1]
            target: (B, 3, H, W) in [-1, 1]

        Returns:
            scalar L1 loss on FFT magnitude spectra
        """
        # 2D FFT on spatial dims (H, W) with orthonormal normalization
        # fft2 does not support BFloat16 — cast to float32 explicitly
        fft_pred = torch.fft.fft2(pred.float(), norm='ortho')
        fft_target = torch.fft.fft2(target.float(), norm='ortho')

        # Magnitude spectra
        mag_pred = fft_pred.abs()
        mag_target = fft_target.abs()

        return F.l1_loss(mag_pred, mag_target)


class EmbeddingAnalyzer:
    """
    Analyzes embedding space quality during training.

    Metrics:
    - Norm statistics: mean, std of embedding magnitudes
    - Cosine similarity distribution: how similar are random pairs?
    - Effective dimensionality: how many dimensions are actually used?
    - Silhouette score: how well separated are classes (if labels available)?
    """

    def __init__(self):
        self.history = []

    def compute_metrics(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive embedding quality metrics.

        Args:
            embeddings: (N, D) embeddings to analyze
            labels: (N,) optional class labels for silhouette score

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # 1. Embedding norm statistics
        norms = torch.norm(embeddings, dim=-1)
        metrics['norm_mean'] = norms.mean().item()
        metrics['norm_std'] = norms.std().item()
        metrics['norm_min'] = norms.min().item()
        metrics['norm_max'] = norms.max().item()

        # 2. Cosine similarity distribution
        normed = F.normalize(embeddings, dim=-1)
        similarity_matrix = torch.matmul(normed, normed.T)

        # Exclude diagonal (self-similarity = 1.0)
        mask = ~torch.eye(len(embeddings), dtype=torch.bool, device=embeddings.device)
        similarities = similarity_matrix[mask]

        metrics['cosine_sim_mean'] = similarities.mean().item()
        metrics['cosine_sim_std'] = similarities.std().item()
        metrics['cosine_sim_min'] = similarities.min().item()
        metrics['cosine_sim_max'] = similarities.max().item()

        # 3. Effective dimensionality via SVD
        # Measures how many dimensions carry significant information
        try:
            # Center embeddings
            centered = embeddings - embeddings.mean(dim=0, keepdim=True)

            # SVD (on CPU for stability with large matrices)
            _, S, _ = torch.linalg.svd(centered.cpu(), full_matrices=False)

            # Compute eigenvalues (variance explained)
            eigenvalues = S ** 2
            eigenvalues = eigenvalues / eigenvalues.sum()

            # Effective rank (participation ratio)
            # If all dimensions equally used: effective_dim ≈ actual_dim
            # If collapsed to few dims: effective_dim << actual_dim
            effective_dim = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
            metrics['effective_dimensionality'] = effective_dim.item()

            # Top-k variance explained
            metrics['variance_explained_top10'] = eigenvalues[:10].sum().item()
            metrics['variance_explained_top50'] = eigenvalues[:50].sum().item()

        except Exception as e:
            metrics['effective_dimensionality'] = -1.0
            metrics['variance_explained_top10'] = -1.0
            metrics['variance_explained_top50'] = -1.0

        # 4. Silhouette score (if labels provided)
        if labels is not None:
            try:
                from sklearn.metrics import silhouette_score
                embeddings_np = embeddings.cpu().numpy()
                labels_np = labels.cpu().numpy()

                # Check if we have multiple classes
                if len(np.unique(labels_np)) > 1:
                    silhouette = silhouette_score(
                        embeddings_np, labels_np, metric='cosine'
                    )
                    metrics['silhouette_score'] = silhouette
                else:
                    metrics['silhouette_score'] = None
            except Exception as e:
                metrics['silhouette_score'] = None

        return metrics

    def log_metrics(self, metrics: Dict[str, float], logger, prefix: str = ""):
        """Pretty print metrics to logger."""
        logger.info(f"{prefix}Embedding Quality Metrics:")
        logger.info(f"  Norm: {metrics['norm_mean']:.3f} ± {metrics['norm_std']:.3f} "
                   f"[{metrics['norm_min']:.3f}, {metrics['norm_max']:.3f}]")
        logger.info(f"  Cosine Similarity: {metrics['cosine_sim_mean']:.3f} ± {metrics['cosine_sim_std']:.3f} "
                   f"[{metrics['cosine_sim_min']:.3f}, {metrics['cosine_sim_max']:.3f}]")

        if metrics['effective_dimensionality'] > 0:
            logger.info(f"  Effective Dimensionality: {metrics['effective_dimensionality']:.1f}")
            logger.info(f"  Variance Explained (top-10): {metrics['variance_explained_top10']*100:.1f}%")
            logger.info(f"  Variance Explained (top-50): {metrics['variance_explained_top50']*100:.1f}%")

        if metrics.get('silhouette_score') is not None:
            logger.info(f"  Silhouette Score: {metrics['silhouette_score']:.3f}")

    def visualize_embedding_space(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor],
        save_path: str,
        method: str = "umap"
    ):
        """
        Create 2D visualization of embedding space using UMAP or t-SNE.

        Args:
            embeddings: (N, D) embeddings
            labels: (N,) optional labels for coloring
            save_path: where to save the plot
            method: "umap" or "tsne"
        """
        try:
            import matplotlib.pyplot as plt

            embeddings_np = embeddings.cpu().numpy()

            # Reduce to 2D
            if method == "umap":
                from umap import UMAP
                reducer = UMAP(n_components=2, random_state=42, n_neighbors=15)
            else:  # tsne
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42, perplexity=30)

            emb_2d = reducer.fit_transform(embeddings_np)

            # Plot
            plt.figure(figsize=(10, 8))

            if labels is not None:
                labels_np = labels.cpu().numpy()
                scatter = plt.scatter(
                    emb_2d[:, 0], emb_2d[:, 1],
                    c=labels_np, cmap='tab10',
                    s=30, alpha=0.7, edgecolors='k', linewidth=0.5
                )
                plt.colorbar(scatter, label='Class')
            else:
                plt.scatter(
                    emb_2d[:, 0], emb_2d[:, 1],
                    s=30, alpha=0.7, c='steelblue',
                    edgecolors='k', linewidth=0.5
                )

            method_name = method.upper()
            plt.title(f'Embedding Space Visualization ({method_name})', fontsize=14, fontweight='bold')
            plt.xlabel(f'{method_name} 1', fontsize=12)
            plt.ylabel(f'{method_name} 2', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            return True

        except ImportError as e:
            print(f"Visualization failed: {e}. Install umap-learn or scikit-learn.")
            return False
        except Exception as e:
            print(f"Visualization error: {e}")
            return False


class LossWeightScheduler:
    """
    Schedules loss weights for all 6 losses during training.

    Strategy:
    - Diffusion + reconstruction: high early, taper slightly
    - Contrastive: ramps up (reduced from before; VICReg shares this role)
    - VICReg: constant from epoch 0 (immediate dimensionality pressure)
    - Lab + Frequency: 0 during warmup, ramp up after (pixel-space losses
      need z0_pred to be meaningful first)

    Warmup phase (epochs 0 to warmup_epochs-1):
        diff/recon/vicreg at start values, contrastive ramps to 30% of range,
        lab/freq stay at 0.

    Main phase (warmup_epochs to num_epochs):
        All losses interpolate linearly from start → end values.
    """

    def __init__(
        self,
        num_epochs: int,
        lambda_diff_start: float = 0.1,
        lambda_diff_end: float = 0.05,
        lambda_contrastive_start: float = 0.05,
        lambda_contrastive_end: float = 0.30,
        lambda_recon_start: float = 0.5,
        lambda_recon_end: float = 0.45,
        lambda_vicreg_start: float = 0.1,
        lambda_vicreg_end: float = 0.1,
        lambda_lab_start: float = 0.0,
        lambda_lab_end: float = 0.15,
        lambda_freq_start: float = 0.0,
        lambda_freq_end: float = 0.10,
        warmup_epochs: int = 10,
    ):
        self.num_epochs = num_epochs
        self.lambda_diff_start = lambda_diff_start
        self.lambda_diff_end = lambda_diff_end
        self.lambda_contrastive_start = lambda_contrastive_start
        self.lambda_contrastive_end = lambda_contrastive_end
        self.lambda_recon_start = lambda_recon_start
        self.lambda_recon_end = lambda_recon_end
        self.lambda_vicreg_start = lambda_vicreg_start
        self.lambda_vicreg_end = lambda_vicreg_end
        self.lambda_lab_start = lambda_lab_start
        self.lambda_lab_end = lambda_lab_end
        self.lambda_freq_start = lambda_freq_start
        self.lambda_freq_end = lambda_freq_end
        self.warmup_epochs = warmup_epochs

    def _lerp(self, start: float, end: float, alpha: float) -> float:
        """Linear interpolation."""
        return start + alpha * (end - start)

    def get_weights(self, epoch: int) -> Dict[str, float]:
        """Get loss weights for current epoch."""

        if epoch < self.warmup_epochs:
            # Warmup: diff/recon at start, contrastive ramps to 30% of range,
            # vicreg at start, lab/freq stay at 0
            alpha = epoch / self.warmup_epochs
            lambda_diff = self.lambda_diff_start
            lambda_recon = self.lambda_recon_start
            lambda_contrastive = self._lerp(
                self.lambda_contrastive_start,
                self.lambda_contrastive_end,
                alpha * 0.3,
            )
            lambda_vicreg = self.lambda_vicreg_start
            lambda_lab = 0.0
            lambda_freq = 0.0
        else:
            # Main training: all losses interpolate linearly
            alpha = (epoch - self.warmup_epochs) / max(self.num_epochs - self.warmup_epochs, 1)
            lambda_diff = self._lerp(self.lambda_diff_start, self.lambda_diff_end, alpha)
            lambda_recon = self._lerp(self.lambda_recon_start, self.lambda_recon_end, alpha)
            lambda_contrastive = self._lerp(self.lambda_contrastive_start, self.lambda_contrastive_end, alpha)
            lambda_vicreg = self._lerp(self.lambda_vicreg_start, self.lambda_vicreg_end, alpha)
            lambda_lab = self._lerp(self.lambda_lab_start, self.lambda_lab_end, alpha)
            lambda_freq = self._lerp(self.lambda_freq_start, self.lambda_freq_end, alpha)

        return {
            'lambda_diff': lambda_diff,
            'lambda_recon': lambda_recon,
            'lambda_contrastive': lambda_contrastive,
            'lambda_vicreg': lambda_vicreg,
            'lambda_lab': lambda_lab,
            'lambda_freq': lambda_freq,
        }

    def log_weights(self, epoch: int, logger):
        """Log current weights."""
        w = self.get_weights(epoch)
        logger.info(
            f"Epoch {epoch+1} Loss Weights: "
            f"λ_diff={w['lambda_diff']:.3f}, "
            f"λ_recon={w['lambda_recon']:.3f}, "
            f"λ_contrastive={w['lambda_contrastive']:.3f}, "
            f"λ_vicreg={w['lambda_vicreg']:.3f}, "
            f"λ_lab={w['lambda_lab']:.3f}, "
            f"λ_freq={w['lambda_freq']:.3f}"
        )
