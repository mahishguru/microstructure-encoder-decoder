"""
Shared utilities for embedding-space training, analysis, and scheduling.
Imported by both trainers and the SDXL trainer.

Classes:
  ContrastiveLoss      — NT-Xent loss for self-supervised encoder training.
  EmbeddingAnalyzer    — Monitors z_512 health (norm, cosine sim, rank, PCA).
  LossWeightScheduler  — Schedules λ_diff / λ_recon / λ_contrastive per epoch.
"""

import math
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 1. Contrastive Loss (NT-Xent) ─────────────────────────────────────────────

class ContrastiveLoss(nn.Module):
    """
    NT-Xent (Normalised Temperature-scaled Cross Entropy) contrastive loss,
    identical to the SimCLR formulation.

    WHY: The encoder produces a 512-D bottleneck z_512. Without an explicit
    similarity constraint, z_512 may encode dataset-level statistics (e.g.
    "this lab uses a specific microscope") rather than microstructure physics.
    The contrastive loss forces z_512 to be similar for the same image under
    small perturbations (sensor noise, slight illumination shift) and dissimilar
    across different microstructures.

    USAGE:
        # Concatenate clean + augmented embeddings along batch axis
        embeddings_both = torch.cat([z_clean, z_aug], dim=0)  # (2B, 512)
        loss = ContrastiveLoss(temperature=0.07)(embeddings_both)

    MATH:
        For a batch of 2N embeddings (N clean + N augmented):
          - positive pairs: (i, i+N) where i is a sample index
          - negative pairs: all other combinations
          - loss = -log( exp(sim(i, i+N)/τ) / Σ_{j≠i} exp(sim(i,j)/τ) )
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        embeddings: (2B, D) — first B rows are clean, last B rows are augmented.
        Returns: scalar NT-Xent loss.
        """
        B_doubled = embeddings.shape[0]
        assert B_doubled % 2 == 0, "embeddings must have even batch size (B clean + B aug)"
        B = B_doubled // 2

        # L2-normalise to unit sphere so cosine similarity is dot product
        z = F.normalize(embeddings, dim=-1)   # (2B, D)

        # Pairwise cosine similarity matrix
        sim = torch.matmul(z, z.T) / self.temperature   # (2B, 2B)

        # Mask: exclude self-similarity from denominator
        mask_self = torch.eye(B_doubled, device=embeddings.device, dtype=torch.bool)
        sim.masked_fill_(mask_self, float('-inf'))

        # Positive pairs: sample i pairs with i+B and i+B pairs with i
        # Indices: for i in [0,B), positive is i+B; for i+B, positive is i
        pos_idx = torch.cat([
            torch.arange(B, B_doubled, device=embeddings.device),
            torch.arange(0, B, device=embeddings.device),
        ])  # (2B,)

        # Cross-entropy: treat each row as a classification problem
        loss = F.cross_entropy(sim, pos_idx)
        return loss


# ── 2. Embedding Analyzer ─────────────────────────────────────────────────────

class EmbeddingAnalyzer:
    """
    Tracks the health of the z_512 embedding space during training.

    Metrics computed:
      norm_mean          — Average L2 norm of z_512 vectors.
                           Healthy range: 1–20. If it explodes → LR too high.
                           If it collapses → encoder is dying.
      norm_std           — Standard deviation of norms across the batch.
      cosine_sim_mean    — Average cosine similarity between *random* pairs.
                           Should be low (< 0.2). If > 0.5 → mode collapse.
      effective_dimensionality — Participation ratio from PCA eigenvalues.
                           Healthy = using most of the 512 dimensions.
                           Collapse = < 10 effective dims.
      rank               — Approximate matrix rank of the embedding batch.
    """

    def compute_metrics(self, embeddings: torch.Tensor) -> dict:
        """
        embeddings: (N, 512) float32 tensor (already gathered from val set).
        Returns a dict of scalar floats.
        """
        with torch.no_grad():
            N, D = embeddings.shape
            if N < 2:
                return {}

            # ── Norm stats ───────────────────────────────────────────────────
            norms     = embeddings.norm(dim=-1)    # (N,)
            norm_mean = norms.mean().item()
            norm_std  = norms.std().item()

            # ── Cosine similarity between random pairs ────────────────────────
            z_norm = F.normalize(embeddings, dim=-1)
            # Sample min(N, 512) pairs to avoid O(N²) cost
            n_pairs = min(N, 512)
            idx_a   = torch.randperm(N, device=embeddings.device)[:n_pairs]
            idx_b   = torch.randperm(N, device=embeddings.device)[:n_pairs]
            # Avoid self-pairs
            mask    = idx_a != idx_b
            if mask.sum() == 0:
                cos_mean = 0.0
            else:
                cos_sim  = (z_norm[idx_a[mask]] * z_norm[idx_b[mask]]).sum(dim=-1)
                cos_mean = cos_sim.mean().item()

            # ── Effective dimensionality (participation ratio) ────────────────
            # PCA on a CPU-friendly subset
            n_sub = min(N, 256)
            sub   = embeddings[:n_sub].float().cpu()
            sub   = sub - sub.mean(dim=0, keepdim=True)
            try:
                _, S, _ = torch.linalg.svd(sub, full_matrices=False)
                eigvals       = (S ** 2) / max(1, n_sub - 1)
                total_var     = eigvals.sum().item()
                if total_var > 0:
                    p             = eigvals / total_var
                    eff_dim       = (1.0 / (p ** 2 + 1e-10).sum()).item()
                else:
                    eff_dim = 0.0
            except Exception:
                eff_dim = 0.0

            # ── Approximate rank ─────────────────────────────────────────────
            try:
                rank = torch.linalg.matrix_rank(sub).item()
            except Exception:
                rank = 0

        return {
            "norm_mean":               norm_mean,
            "norm_std":                norm_std,
            "cosine_sim_mean":         cos_mean,
            "effective_dimensionality": eff_dim,
            "rank":                    rank,
        }

    def log_metrics(
        self,
        metrics: dict,
        logger: logging.Logger,
        prefix: str = "  ",
    ):
        """Print metrics with health interpretation."""
        if not metrics:
            return

        norm     = metrics.get("norm_mean", 0)
        cos_sim  = metrics.get("cosine_sim_mean", 0)
        eff_dim  = metrics.get("effective_dimensionality", 0)

        norm_health  = "✅" if 0.5 < norm < 30 else "⚠️ "
        cos_health   = "✅" if cos_sim < 0.3 else "⚠️  HIGH — possible collapse"
        dim_health   = "✅" if eff_dim > 20 else "⚠️  LOW — possible collapse"

        logger.info(
            f"{prefix}Embedding metrics:"
            f"  norm={norm:.2f}±{metrics.get('norm_std',0):.2f} {norm_health}"
            f"  cos_sim={cos_sim:.3f} {cos_health}"
            f"  eff_dim={eff_dim:.1f} {dim_health}"
            f"  rank={metrics.get('rank', '?')}"
        )

    def visualize_embedding_space(
        self,
        embeddings:  torch.Tensor,
        labels:      Optional[torch.Tensor],
        save_path:   str,
        method:      str = "pca",   # "pca" or "umap"
    ):
        """
        2-D scatter plot of the embedding space.

        Falls back to PCA if UMAP is not installed or if N < 20.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        N = embeddings.shape[0]
        z_cpu = embeddings.float().cpu().detach()

        coords = None

        if method == "umap" and N >= 20:
            try:
                import umap                                         # type: ignore
                reducer = umap.UMAP(n_components=2, random_state=42)
                coords  = reducer.fit_transform(z_cpu.numpy())
            except ImportError:
                pass   # fall through to PCA

        if coords is None:
            # PCA fallback
            z_c   = z_cpu - z_cpu.mean(dim=0)
            _, _, Vt = torch.linalg.svd(z_c, full_matrices=False)
            proj  = z_c @ Vt[:2].T                                  # (N, 2)
            coords = proj.numpy()

        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=labels.cpu().numpy() if labels is not None else "steelblue",
            alpha=0.7, s=20, cmap="tab20" if labels is not None else None,
        )
        if labels is not None:
            plt.colorbar(sc, ax=ax, label="label")
        ax.set_title(f"Embedding space ({method.upper()}, N={N})", fontsize=11)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()


# ── 3. Loss Weight Scheduler ──────────────────────────────────────────────────

class LossWeightScheduler:
    """
    Linearly interpolates λ_diff, λ_recon, λ_contrastive over training epochs.

    Strategy for microstructure reconstruction:
      - Early epochs:  high λ_contrastive → encoder learns robust z_512.
      - Middle epochs: high λ_recon → decoder anchored to clean latent.
      - Late epochs:   high λ_diff → fine-tune the diffusion objective.

    All three weights are active at all times; the schedule just adjusts emphasis.

    Warmup: during the first `warmup_epochs` epochs, λ_contrastive is ramped
    from 0 to its start value to prevent the contrastive loss from dominating
    before the encoder has stabilised.

    VICReg and Frequency are regularizers — they should be at constant target
    weight for most of training.  Instead of ramping start→end over all epochs,
    they use a delayed onset: silent for `delay` epochs, then ramp to full
    weight over `ramp` epochs, and hold at full weight for the remainder.
    """

    def __init__(
        self,
        num_epochs: int,
        lambda_diff_start:        float = 0.5,
        lambda_diff_end:          float = 1.0,
        lambda_recon_start:       float = 0.5,
        lambda_recon_end:         float = 0.1,
        lambda_contrastive_start: float = 0.3,
        lambda_contrastive_end:   float = 0.05,
        lambda_vicreg:            float = 0.05,
        lambda_freq:              float = 0.3,
        warmup_epochs:            int   = 5,
        warmup_epochs_vicreg:     int   = 2,
        warmup_epochs_freq:       int   = 4,
        ramp_epochs_vicreg:       int   = 2,
        ramp_epochs_freq:         int   = 3,
    ):
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_epochs_vicreg = warmup_epochs_vicreg
        self.warmup_epochs_freq   = warmup_epochs_freq
        self.ramp_epochs_vicreg   = ramp_epochs_vicreg
        self.ramp_epochs_freq     = ramp_epochs_freq

        self._diff_s    = lambda_diff_start
        self._diff_e    = lambda_diff_end
        self._recon_s   = lambda_recon_start
        self._recon_e   = lambda_recon_end
        self._contr_s   = lambda_contrastive_start
        self._contr_e   = lambda_contrastive_end
        self._vicreg    = lambda_vicreg
        self._freq      = lambda_freq

    def _lerp(self, start: float, end: float, epoch: int) -> float:
        """Linear interpolation over [0, num_epochs-1]."""
        if self.num_epochs <= 1:
            return end
        frac = min(1.0, epoch / max(1, self.num_epochs - 1))
        return start + (end - start) * frac

    def _warmup(self, epoch: int, warmup_ep: int) -> float:
        """Linear ramp from 0 to 1 over warmup_ep epochs (contrastive)."""
        return min(1.0, epoch / max(1, warmup_ep))

    def _delayed_onset(self, epoch: int, delay: int, ramp: int) -> float:
        """Silent for `delay` epochs, then linear ramp to 1.0 over `ramp` epochs."""
        if epoch < delay:
            return 0.0
        return min(1.0, (epoch - delay) / max(1, ramp))

    def get_weights(self, epoch: int) -> dict:
        """
        Returns {lambda_diff, lambda_recon, lambda_contrastive,
                 lambda_vicreg, lambda_freq} for the epoch.
        """
        return {
            "lambda_diff":        self._lerp(self._diff_s,  self._diff_e,  epoch),
            "lambda_recon":       self._lerp(self._recon_s, self._recon_e, epoch),
            "lambda_contrastive": self._lerp(self._contr_s, self._contr_e, epoch) * self._warmup(epoch, self.warmup_epochs),
            "lambda_vicreg":      self._vicreg * self._delayed_onset(epoch, self.warmup_epochs_vicreg, self.ramp_epochs_vicreg),
            "lambda_freq":        self._freq   * self._delayed_onset(epoch, self.warmup_epochs_freq,   self.ramp_epochs_freq),
        }

    def log_weights(self, epoch: int, logger: logging.Logger):
        w = self.get_weights(epoch)
        logger.info(
            f"  Loss weights epoch {epoch + 1}:"
            f"  λ_diff={w['lambda_diff']:.3f}"
            f"  λ_recon={w['lambda_recon']:.3f}"
            f"  λ_contrastive={w['lambda_contrastive']:.3f}"
            f"  λ_vicreg={w['lambda_vicreg']:.3f}"
            f"  λ_freq={w['lambda_freq']:.3f}"
        )