"""
orientation_loss.py — Differentiable, symmetry-aware orientation loss for
microstructure (IPF/orientation-map) reconstruction.

WHY THIS EXISTS
---------------
The decoders are trained with image-space objectives (latent flow-matching MSE,
pixel frequency loss). The dataset RGB encodes crystal orientation through the
`orientation_codec` map

    quat --stereographic--> R^3 --class-mean shift--> R^3 --[-1,1]->[0,255]--> RGB

which is HIGHLY non-isometric: a unit RGB error can mean anywhere from ~0.1° to
~90° crystallographic misorientation. So an image-space loss is effectively
blind to orientation, and per-grain disorientation saturates near the HCP random
baseline (~50–75°).

This module reconstructs orientation from the decoded RGB *differentiably* and
measures a proper HCP fundamental-zone-folded disorientation, so the decoder can
be supervised in the manifold that actually matters.

KEY SIMPLIFICATION (class-mean cancels)
---------------------------------------
The codec applies a per-alloy class-mean quaternion as a LEFT multiplication:
    q_centered = q_mean ⊗ q_raw.
The disorientation between prediction and target uses
    misor = conj(q_pred) ⊗ q_gt.
Substituting the centered forms, q_mean cancels EXACTLY:
    conj(q_mean ⊗ q_pred) ⊗ (q_mean ⊗ q_gt) = conj(q_pred) ⊗ q_gt.
=> The loss is independent of the class mean. We do NOT need class_means.json,
   per-sample plumbing, or any handling of the `mirror_` augmentation prefix.
   We compare the RAW decoded quaternions of pred vs. target directly.

This mirrors the eval metric in microstructure_ed/eval/material_metrics.py
(`disorientation_deg`: one-sided 12-symmetry fold, wxyz convention).

MATH
----
RGB in [-1,1] already equals the stereographic coordinate S (the codec's
[-1,1]->[0,255] pack cancels the dataset's [0,255]->[-1,1] normalize), so:

    S      = clamp(rgb, -1+eps, 1-eps)
    S2     = Sx^2 + Sy^2 + Sz^2
    qw     = (1 - S2) / (1 + S2)          # rational inverse stereographic
    qxyz   = 2 S / (1 + S2)               # (no square roots -> no domain errors)
    q      = normalize([qw, qx, qy, qz]); flip to w >= 0    (wxyz)

    misor      = conj(q_pred) ⊗ q_gt
    cos(θ/2)   = max_{g in HCP(12)} | w( g ⊗ misor ) |       # FZ-aware argmin
    L_pixel    = mean( 1 - cos(θ/2) )                        # chord loss ∈ [0,1]

`L_pixel` is smooth, bounded, never NaN (no arccos), and monotone in the true
misorientation angle. The `max_g` is the symmetry-correct "argmin over the 12
operators" recommended for HCP — without it two physically-equivalent
orientations look ~90° apart.

A second, optional `L_pool` term average-pools the unit-quaternion fields over
KxK windows (default 16) and applies the same chord loss to the pooled mean
orientation. This is a label-free approximation of the per-grain metric and
tolerates 1–2 px boundary shifts. It is a regularizer (small weight); pooling
across a grain boundary is inherently meaningless and is accepted as such.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── HCP symmetry quaternions (12, 4) in wxyz ─────────────────────────────────
# Built to match orientation_codec.symmetry.get_hcp_symmetries():
#   for i in 0..5: theta = i*60°;  Rz(theta)  and  Rz(theta) ∘ Rx(180°)
# Generated in torch so this module has no scipy / codec import dependency.

def _qmul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of (..., 4) wxyz quaternions (matches scipy R(a)*R(b))."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _build_hcp_syms() -> torch.Tensor:
    """Return the 12 HCP proper-rotation symmetries as (12, 4) wxyz quaternions."""
    syms = []
    # Rx(180°) as wxyz: (cos90, sin90, 0, 0) = (0, 1, 0, 0)
    qx_pi = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    for i in range(6):
        theta = i * (math.pi / 3.0)
        # Rz(theta) as wxyz: (cos(t/2), 0, 0, sin(t/2))
        qz = torch.tensor(
            [math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)],
            dtype=torch.float64,
        )
        syms.append(qz)
        syms.append(_qmul_wxyz(qz, qx_pi))   # Rz ∘ Rx(180°)
    out = torch.stack(syms, dim=0)           # (12, 4)
    # Normalize defensively and place on the w >= 0 hemisphere.
    out = out / out.norm(dim=-1, keepdim=True)
    out = torch.where(out[:, :1] < 0, -out, out)
    return out.to(torch.float32)


def _qconj_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of (..., 4) wxyz quaternions."""
    return q * q.new_tensor([1.0, -1.0, -1.0, -1.0])


def rgb_to_quat_wxyz(rgb_pm1: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Differentiable inverse codec: RGB in [-1,1] -> unit quaternion field.

    Parameters
    ----------
    rgb_pm1 : (B, 3, H, W) float — decoder output in [-1, 1] (== stereographic S).
    eps     : clamp margin to keep S strictly inside (-1, 1) for stable grads.

    Returns
    -------
    q : (B, 4, H, W) float — unit quaternions in wxyz, w >= 0.
    """
    S = rgb_pm1.clamp(-1.0 + eps, 1.0 - eps)
    Sx, Sy, Sz = S[:, 0], S[:, 1], S[:, 2]          # (B, H, W)
    S2 = Sx * Sx + Sy * Sy + Sz * Sz                 # (B, H, W)
    denom = 1.0 + S2
    qw = (1.0 - S2) / denom
    qx = 2.0 * Sx / denom
    qy = 2.0 * Sy / denom
    qz = 2.0 * Sz / denom
    q = torch.stack([qw, qx, qy, qz], dim=1)         # (B, 4, H, W)
    q = q / (q.norm(dim=1, keepdim=True) + 1e-12)
    # Flip to w >= 0 hemisphere (q and -q are the same rotation).
    q = torch.where(q[:, :1] < 0, -q, q)
    return q


def disorientation_cos_half(
    q_a: torch.Tensor, q_b: torch.Tensor, syms: torch.Tensor
) -> torch.Tensor:
    """HCP-folded cos(θ/2) between two wxyz quaternion sets.

    Parameters
    ----------
    q_a, q_b : (..., 4) wxyz unit quaternions.
    syms     : (G, 4) wxyz symmetry quaternions.

    Returns
    -------
    cos_half : (...) tensor in [0, 1]; 1 => identical orientation (θ=0),
               smaller => larger misorientation. The minimal misorientation
               over the G symmetry operators (one-sided), matching the eval
               metric `disorientation_deg`.
    """
    misor = _qmul_wxyz(_qconj_wxyz(q_a), q_b)        # (..., 4)
    mw = misor[..., 0:1]
    mx = misor[..., 1:2]
    my = misor[..., 2:3]
    mz = misor[..., 3:4]                              # (..., 1)
    gw, gx, gy, gz = syms[:, 0], syms[:, 1], syms[:, 2], syms[:, 3]   # (G,)
    # w-component of (g ⊗ misor) for every symmetry g, broadcast over (...).
    w = gw * mw - gx * mx - gy * my - gz * mz         # (..., G)
    cos_half = w.abs().amax(dim=-1)                   # (...)
    return cos_half.clamp(0.0, 1.0)


class OrientationLoss(nn.Module):
    """Symmetry-aware orientation reconstruction loss (HCP).

    Combines a per-pixel FZ-folded quaternion chord loss (primary) with an
    optional pooled mean-orientation chord loss (regularizer / per-grain proxy).

    forward(img_pred, img_tgt) -> (total, info)
        img_pred, img_tgt : (B, 3, H, W) in [-1, 1].
    """

    def __init__(
        self,
        pix_weight: float = 0.7,
        pool_weight: float = 0.3,
        pool_size: int = 16,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.pix_weight = float(pix_weight)
        self.pool_weight = float(pool_weight)
        self.pool_size = int(pool_size)
        self.eps = float(eps)
        self.register_buffer("syms", _build_hcp_syms(), persistent=False)

    def _pixel_term(self, q_pred: torch.Tensor, q_tgt: torch.Tensor,
                    sample_weight: torch.Tensor = None) -> torch.Tensor:
        B = q_pred.shape[0]
        qp = q_pred.permute(0, 2, 3, 1).reshape(B, -1, 4)   # (B, N, 4)
        qt = q_tgt.permute(0, 2, 3, 1).reshape(B, -1, 4)
        cos_half = disorientation_cos_half(qp, qt, self.syms)   # (B, N)
        per_sample = (1.0 - cos_half).mean(dim=1)               # (B,)
        if sample_weight is not None:
            w = sample_weight.to(per_sample.dtype)
            return (per_sample * w).sum() / (w.sum() + 1e-8)
        return per_sample.mean()

    def _pool_term(self, q_pred: torch.Tensor, q_tgt: torch.Tensor,
                   sample_weight: torch.Tensor = None) -> torch.Tensor:
        k = self.pool_size
        # Average-pool the unit-quaternion fields, then renormalize to a mean
        # orientation direction (all quats are on the w >= 0 hemisphere).
        pp = F.avg_pool2d(q_pred, k)
        pt = F.avg_pool2d(q_tgt, k)
        pp = pp / (pp.norm(dim=1, keepdim=True) + 1e-12)
        pt = pt / (pt.norm(dim=1, keepdim=True) + 1e-12)
        B = pp.shape[0]
        pp = pp.permute(0, 2, 3, 1).reshape(B, -1, 4)
        pt = pt.permute(0, 2, 3, 1).reshape(B, -1, 4)
        cos_half = disorientation_cos_half(pp, pt, self.syms)
        per_sample = (1.0 - cos_half).mean(dim=1)               # (B,)
        if sample_weight is not None:
            w = sample_weight.to(per_sample.dtype)
            return (per_sample * w).sum() / (w.sum() + 1e-8)
        return per_sample.mean()

    def forward(
        self, img_pred: torch.Tensor, img_tgt: torch.Tensor,
        sample_weight: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Compute in float32 for numerical stability regardless of autocast.
        q_pred = rgb_to_quat_wxyz(img_pred.float(), self.eps)
        q_tgt = rgb_to_quat_wxyz(img_tgt.float(), self.eps)

        loss_pix = self._pixel_term(q_pred, q_tgt, sample_weight)
        if self.pool_weight > 0.0:
            loss_pool = self._pool_term(q_pred, q_tgt, sample_weight)
        else:
            loss_pool = torch.zeros((), device=img_pred.device)

        total = self.pix_weight * loss_pix + self.pool_weight * loss_pool
        info = {
            "loss_orient_pix": float(loss_pix.detach()),
            "loss_orient_pool": float(loss_pool.detach()),
            "loss_orient": float(total.detach()),
        }
        return total, info



class OrientationDistributionLoss(nn.Module):
    """Distribution-matching orientation loss (HCP), spatially UNALIGNED.

    WHY (replaces the per-pixel OrientationLoss for the vitdit objective)
    --------------------------------------------------------------------
    The goal is a STATISTICALLY similar microstructure ("from somewhere else in
    the same specimen"), not a pixel-perfect copy. A per-pixel disorientation
    penalises the model for placing the right texture at a different spatial
    position, which directly fights that goal. This loss instead matches two
    spatial-arrangement-invariant statistics between pred and target:

      1. ODF term: soft kernel-density estimate of the orientation
         distribution over a fixed set of reference orientations (FZ-folded
         via the 12 HCP operators), compared with an L1 / total-variation
         distance. Invariant to ANY permutation of the pixels.
      2. Misorientation term: distribution of FZ-folded disorientation
         cos(theta/2) between pixels separated by a set of spatial offsets
         (two-point orientation statistics: captures grain size and boundary
         character), compared with L1. Invariant to translations.

    Both terms are smooth (no arccos, no hard binning), bounded in [0, 2],
    and computed on a random pixel subsample for efficiency.

    forward(img_pred, img_tgt, sample_weight) -> (total, info)
        Same signature as OrientationLoss -> drop-in replacement in trainers.
    """

    def __init__(
        self,
        odf_weight: float = 0.6,
        misori_weight: float = 0.4,
        n_refs: int = 64,
        kappa: float = 64.0,
        n_pix: int = 4096,
        pair_offsets: Tuple[int, ...] = (1, 4, 16, 64),
        n_angle_bins: int = 18,
        eps: float = 1e-4,
        seed: int = 0,
        # ── v2 metric-aware extensions (defaults = OFF -> legacy behaviour) ──
        pf_weight: float = 0.0,        # B2b differentiable pole-figure loss
        scatter_weight: float = 0.0,   # B2c Bingham/scatter anti-shrinkage
        odf_mode: str = "l1",          # "l1" (legacy) | "sinkhorn" (B2a)
        sinkhorn_eps: float = 0.05,    # entropic eps, relative to max cost
        sinkhorn_iters: int = 30,
        pf_grid: int = 256,            # Fibonacci hemisphere nodes
        pf_kappa: float = 64.0,        # vMF KDE concentration on S^2
        pf_prismatic: bool = True,     # add {10-10} maps besides (0002)
        # ── v3 extensions (defaults = OFF -> v2/legacy behaviour) ─────────
        pix_weight: float = 0.0,       # matched-pixel FZ chord anchor (small!)
        pf_sharp_weight: float = 0.0,  # texture-index + maxMRD proxy matching
        mmd_kappa: float = 64.0,       # disorientation-kernel bandwidth (MMD)
        mmd_n: int = 512,              # pixels per image for pairwise MMD
    ):
        super().__init__()
        self.odf_weight = float(odf_weight)
        self.misori_weight = float(misori_weight)
        self.pf_weight = float(pf_weight)
        self.scatter_weight = float(scatter_weight)
        assert odf_mode in ("l1", "sinkhorn", "mmd"), odf_mode
        self.odf_mode = odf_mode
        self.pix_weight = float(pix_weight)
        self.pf_sharp_weight = float(pf_sharp_weight)
        self.mmd_kappa = float(mmd_kappa)
        self.mmd_n = int(mmd_n)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.pf_kappa = float(pf_kappa)
        self.pf_prismatic = bool(pf_prismatic)
        self.kappa = float(kappa)          # ~64 -> ~15 deg kernel width
        self.n_pix = int(n_pix)
        self.pair_offsets = tuple(int(o) for o in pair_offsets)
        self.eps = float(eps)
        self.register_buffer("syms", _build_hcp_syms(), persistent=False)
        # Fixed reference orientations: uniform on SO(3) via normalised 4-D
        # Gaussians (FZ folding happens inside disorientation_cos_half).
        g = torch.Generator().manual_seed(seed)
        refs = torch.randn(n_refs, 4, generator=g)
        refs = refs / refs.norm(dim=-1, keepdim=True)
        refs = torch.where(refs[:, :1] < 0, -refs, refs)
        self.register_buffer("refs", refs, persistent=False)
        # Soft-histogram bin centres in cos(theta/2) space.
        # HCP max disorientation ~93.84 deg -> cos_half ~0.683.
        centers = torch.linspace(0.68, 1.0, int(n_angle_bins))
        self.register_buffer("bin_centers", centers, persistent=False)
        self.bin_bw = float(centers[1] - centers[0])

        # B2a: R x R disorientation-angle cost matrix between the refs (rad).
        if self.odf_mode == "sinkhorn":
            ch = disorientation_cos_half(
                refs.unsqueeze(1), refs.unsqueeze(0), _build_hcp_syms()
            ).clamp(-1.0, 1.0)                              # (R, R)
            cost = 2.0 * torch.acos(ch)                     # angle in radians
            cost = 0.5 * (cost + cost.T)                    # symmetrize
            cost.fill_diagonal_(0.0)
            self.register_buffer("ref_cost", cost, persistent=False)
            self.sinkhorn_eps_abs = float(sinkhorn_eps) * float(cost.max())

        # B2b: Fibonacci upper-hemisphere grid + crystal directions.
        if self.pf_weight > 0.0 or self.pf_sharp_weight > 0.0:
            m = int(pf_grid)
            i = torch.arange(m, dtype=torch.float64) + 0.5
            phi = math.pi * (1.0 + 5.0 ** 0.5) * i          # golden angle
            cz = i / m                                       # z in (0,1] upper hemisphere
            sz = torch.sqrt((1.0 - cz * cz).clamp_min(0.0))
            nodes = torch.stack(
                [sz * torch.cos(phi), sz * torch.sin(phi), cz], dim=-1
            ).to(torch.float32)                              # (G, 3)
            self.register_buffer("pf_nodes", nodes, persistent=False)
            dirs = [torch.tensor([0.0, 0.0, 1.0])]           # (0002) basal
            if self.pf_prismatic:
                for k in range(6):                            # {10-10} star
                    a = k * math.pi / 3.0
                    dirs.append(torch.tensor([math.cos(a), math.sin(a), 0.0]))
            self.register_buffer(
                "pf_dirs", torch.stack(dirs).to(torch.float32), persistent=False
            )                                                # (D, 3)

    @staticmethod
    def _subsample(q: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """(B, 4, H, W) -> (B, n, 4) at flat spatial indices idx."""
        B = q.shape[0]
        return q.permute(0, 2, 3, 1).reshape(B, -1, 4)[:, idx]

    def _odf_hist(self, q: torch.Tensor) -> torch.Tensor:
        """(B, n, 4) -> (B, R) soft ODF histogram over the reference set."""
        s = disorientation_cos_half(
            q.unsqueeze(2), self.refs.view(1, 1, -1, 4), self.syms
        )                                                   # (B, n, R)
        a = F.softmax(self.kappa * s, dim=-1)               # soft assignment
        return a.mean(dim=1)                                # (B, R)

    def _angle_hist(self, cos_half: torch.Tensor) -> torch.Tensor:
        """(B, n) -> (B, nbins) soft histogram of cos(theta/2) values."""
        d = cos_half.unsqueeze(-1) - self.bin_centers.view(1, 1, -1)
        w = torch.exp(-(d / self.bin_bw) ** 2)
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-12)
        return w.mean(dim=1)

    # ── B2a: debiased Sinkhorn divergence over the reference-ODF simplex ─────
    def _ot_eps(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Entropic OT cost <P, C> between histograms a, b: (B, R) -> (B,)."""
        C = self.ref_cost                                    # (R, R)
        eps = self.sinkhorn_eps_abs
        log_a = (a + 1e-12).log()
        log_b = (b + 1e-12).log()
        f = torch.zeros_like(a)
        g = torch.zeros_like(b)
        negC = (-C / eps).unsqueeze(0)                       # (1, R, R)
        for _ in range(self.sinkhorn_iters):
            f = -eps * torch.logsumexp(negC + (g / eps + log_b).unsqueeze(1), dim=2)
            g = -eps * torch.logsumexp(
                negC.transpose(1, 2) + (f / eps + log_a).unsqueeze(1), dim=2)
        logP = ((f.unsqueeze(2) + g.unsqueeze(1) - C.unsqueeze(0)) / eps
                + log_a.unsqueeze(2) + log_b.unsqueeze(1))
        P = torch.exp(logP)
        return (P * C.unsqueeze(0)).sum(dim=(1, 2))

    def _sinkhorn_div(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Debiased Sinkhorn divergence (B,) — ~Wasserstein on SO(3)/sym, radians."""
        return (self._ot_eps(a, b)
                - 0.5 * self._ot_eps(a, a)
                - 0.5 * self._ot_eps(b, b)).clamp_min(0.0)

    # ── B2b: differentiable pole-figure histograms ───────────────────────────
    @staticmethod
    def _quat_rotate_dirs(q: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
        """Rotate crystal directions to sample frame. q: (B,n,4) wxyz,
        dirs: (D,3) -> (B,n,D,3).  v' = v + 2 w (u x v) + 2 u x (u x v)."""
        w = q[..., 0:1].unsqueeze(2)                         # (B,n,1,1)
        u = q[..., 1:4].unsqueeze(2)                         # (B,n,1,3)
        v = dirs.view(1, 1, -1, 3)                           # (1,1,D,3)
        uv = torch.cross(u.expand(*u.shape[:2], v.shape[2], 3), 
                         v.expand(u.shape[0], u.shape[1], -1, 3), dim=-1)
        uuv = torch.cross(u.expand_as(uv), uv, dim=-1)
        return v + 2.0 * (w * uv + uuv)

    def _pf_hist(self, q: torch.Tensor) -> torch.Tensor:
        """(B,n,4) -> (B, n_maps*G) concatenated soft pole-figure KDEs.
        Antipodal fold via |dot|; basal map is exactly HCP-invariant, the
        prismatic maps are made invariant by averaging the 6-fold star."""
        v = self._quat_rotate_dirs(q, self.pf_dirs)          # (B,n,D,3)
        dots = torch.einsum("bndk,gk->bndg", v, self.pf_nodes).abs()
        w = torch.exp(self.pf_kappa * (dots - 1.0))          # vMF, antipodal
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-12)        # (B,n,D,G)
        basal = w[:, :, 0].mean(dim=1)                       # (B,G)
        if w.shape[2] > 1:
            pris = w[:, :, 1:].mean(dim=2).mean(dim=1)       # (B,G)
            return torch.cat([basal, pris], dim=-1)
        return basal

    # ── v3a: matched-pixel FZ-folded chord anchor ────────────────────────────
    def _pix_term(self, sp: torch.Tensor, st: torch.Tensor) -> torch.Tensor:
        """Per-pixel chord loss on MATCHED subsampled pixels (B,n,4)->(B,).
        Class-mean cancels exactly (conj(p) @ t). Keep the weight SMALL: with
        imperfect spatial alignment its optimum drifts toward the mean
        orientation, i.e. the very shrinkage we fight. It exists only to anchor
        the image-specific texture so distribution terms cannot be satisfied by
        a generic texture (the v2 ep30 PF-correlation collapse)."""
        m = _qmul_wxyz(_qconj_wxyz(sp), st)                  # (B,n,4)
        prods = _qmul_wxyz(m.unsqueeze(2), self.syms.view(1, 1, -1, 4))
        cos_half = prods[..., 0].abs().amax(dim=-1)          # (B,n)
        return (1.0 - cos_half).mean(dim=1)                  # (B,)

    # ── v3b: grid-free MMD on the disorientation kernel ──────────────────────
    def _mmd_term(self, sp: torch.Tensor, st: torch.Tensor) -> torch.Tensor:
        """Biased-estimator MMD^2 between pred/target orientation samples with
        kernel k(x,y)=exp(kappa*(cos_half(x,y)-1)), cos_half FZ-folded.

        REPLACES the sinkhorn-on-reference-histogram ODF term: with 128 refs
        and a ~10 deg kernel most of the HCP FZ is NOT covered, so histogram
        gradients quantize toward arbitrary ref points (v2 failure). MMD works
        directly on the samples - no reference grid, no quantization.

        Identity used: w(conj(x) (x) (y (x) g)) = <x, y (x) g>, so the pairwise
        FZ-folded cos_half is max_g |X . (Y (x) g)| - one einsum, no B,m,m,4
        intermediates."""
        m = min(self.mmd_n, sp.shape[1])
        x, y = sp[:, :m], st[:, :m]                          # (B,m,4)
        yg = _qmul_wxyz(y.unsqueeze(2), self.syms.view(1, 1, -1, 4))  # (B,m,12,4)
        xg = _qmul_wxyz(x.unsqueeze(2), self.syms.view(1, 1, -1, 4))

        def _k(a, bg):                                        # (B,m,4),(B,m,12,4)
            ch = torch.einsum("bik,bjgk->bijg", a, bg).abs().amax(dim=-1)
            return torch.exp(self.mmd_kappa * (ch - 1.0)).mean(dim=(1, 2))

        return (_k(x, xg) + _k(y, yg) - 2.0 * _k(x, yg)).clamp_min(0.0)  # (B,)

    # ── v3c: pole-figure sharpness (texture-index + maxMRD proxies) ──────────
    def _pf_sharp(self, hp: torch.Tensor, ht: torch.Tensor) -> torch.Tensor:
        """Match per-image PF concentration DIRECTLY: the two failing eval
        metrics are the texture index (integral of density^2) and maxMRD (peak
        density). With per-map histograms p (sum=1 over G nodes) the MRD field
        is G*p, so TI ~ G*sum(p^2) and maxMRD ~ G*max(p). L1-smoothed PF maps
        can be matched by a diffuse prediction (kernel-width blindness of the
        v2 pf term); these scalars cannot."""
        G = self.pf_nodes.shape[0]
        n_maps = hp.shape[-1] // G
        p = hp.view(hp.shape[0], n_maps, G)
        t = ht.view(ht.shape[0], n_maps, G)
        ti_p, ti_t = G * (p ** 2).sum(-1), G * (t ** 2).sum(-1)
        mx_p, mx_t = G * p.amax(-1), G * t.amax(-1)
        rel = ((ti_p - ti_t).abs() / (ti_t + 1e-6)
               + (mx_p - mx_t).abs() / (mx_t + 1e-6))
        return rel.mean(dim=1)                                # (B,)

    # ── B2c: Bingham/scatter concentration term ──────────────────────────────
    def _fz_fold(self, q: torch.Tensor) -> torch.Tensor:
        """Fold (B,n,4) into a consistent FZ representative: q ⊗ sym_k with
        k = argmax |w|, then flip to w >= 0. Piecewise-constant selection;
        gradients flow through the selected branch."""
        prods = _qmul_wxyz(q.unsqueeze(2), self.syms.view(1, 1, -1, 4))  # (B,n,G,4)
        k = prods[..., 0].abs().argmax(dim=-1)               # (B,n)
        idx = k[..., None, None].expand(-1, -1, 1, 4)
        qf = torch.gather(prods, 2, idx).squeeze(2)          # (B,n,4)
        return torch.where(qf[..., 0:1] < 0, -qf, qf)

    def _scatter_term(self, qp: torch.Tensor, qt: torch.Tensor) -> torch.Tensor:
        """|T_p - T_t|_F + |lambda1_p - lambda1_t|, T = E[q q^T] after FZ fold."""
        fp = self._fz_fold(qp)
        ft = self._fz_fold(qt)
        n = fp.shape[1]
        Tp = torch.einsum("bni,bnj->bij", fp, fp) / n        # (B,4,4)
        Tt = torch.einsum("bni,bnj->bij", ft, ft) / n
        frob = (Tp - Tt).flatten(1).norm(dim=1)
        l1p = torch.linalg.eigvalsh(Tp)[..., -1]
        l1t = torch.linalg.eigvalsh(Tt)[..., -1]
        return frob + (l1p - l1t).abs()                      # (B,)

    def forward(
        self, img_pred: torch.Tensor, img_tgt: torch.Tensor,
        sample_weight: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # bf16 autocast breaks the v2 terms: linalg.eigvalsh has no CUDA
        # BFloat16 kernel (scatter term) and the vMF/Sinkhorn exp/logsumexp
        # lose precision. Compute the whole loss in fp32; inputs are cast
        # below and n_pix is small, so the cost is negligible.
        with torch.amp.autocast("cuda", enabled=False):
            return self._forward_fp32(img_pred, img_tgt, sample_weight)

    def _forward_fp32(
        self, img_pred: torch.Tensor, img_tgt: torch.Tensor,
        sample_weight: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        q_pred = rgb_to_quat_wxyz(img_pred.float(), self.eps)   # (B, 4, H, W)
        q_tgt = rgb_to_quat_wxyz(img_tgt.float(), self.eps)
        B, _, H, W = q_pred.shape
        n = min(self.n_pix, H * W)
        idx = torch.randperm(H * W, device=q_pred.device)[:n]
        sp = self._subsample(q_pred, idx)                       # (B, n, 4)
        st = self._subsample(q_tgt, idx)

        # 1. ODF distribution distance (per sample).
        if self.odf_mode == "mmd":
            # v3: grid-free pairwise MMD (no reference-grid quantization).
            odf_per = self._mmd_term(sp, st)
        else:
            ho_p, ho_t = self._odf_hist(sp), self._odf_hist(st)
            if self.odf_mode == "sinkhorn":
                # Debiased Sinkhorn divergence with disorientation ground cost:
                # gradients TRANSPORT mass toward the right orientations instead
                # of only deflating wrong bins (anti-smearing). Units: radians.
                odf_per = self._sinkhorn_div(ho_p, ho_t)
            else:
                odf_per = (ho_p - ho_t).abs().sum(dim=-1)

        # 2. Two-point misorientation distribution distance.
        hp, ht = [], []
        for off in self.pair_offsets:
            rp = self._subsample(torch.roll(q_pred, (off, off), dims=(2, 3)), idx)
            rt = self._subsample(torch.roll(q_tgt, (off, off), dims=(2, 3)), idx)
            hp.append(self._angle_hist(disorientation_cos_half(sp, rp, self.syms)))
            ht.append(self._angle_hist(disorientation_cos_half(st, rt, self.syms)))
        mis_per = (torch.stack(hp) - torch.stack(ht)).abs().sum(dim=-1).mean(dim=0)

        # 3. B2b pole-figure loss (basal + prismatic soft PF maps, L1).
        #    NOTE class-mean caveat: absolute PFs do NOT cancel the codec's
        #    class-mean shift, but pred & target share the same shifted frame,
        #    so the LOSS is valid. External MTEX comparisons must re-apply the
        #    class mean before comparing to these maps.
        if self.pf_weight > 0.0 or self.pf_sharp_weight > 0.0:
            hpf_p, hpf_t = self._pf_hist(sp), self._pf_hist(st)
            pf_per = ((hpf_p - hpf_t).abs().sum(dim=-1)
                      if self.pf_weight > 0.0 else torch.zeros_like(odf_per))
            pfs_per = (self._pf_sharp(hpf_p, hpf_t)
                       if self.pf_sharp_weight > 0.0 else torch.zeros_like(odf_per))
        else:
            pf_per = torch.zeros_like(odf_per)
            pfs_per = torch.zeros_like(odf_per)

        # 4. B2c scatter/Bingham concentration term (anti-shrinkage: punishes
        #    "right fibre axis but too diffuse", i.e. the maxMRD collapse).
        if self.scatter_weight > 0.0:
            sc_per = self._scatter_term(sp, st)
        else:
            sc_per = torch.zeros_like(odf_per)

        # 5. v3a matched-pixel anchor (small weight - see _pix_term docstring).
        if self.pix_weight > 0.0:
            pix_per = self._pix_term(sp, st)
        else:
            pix_per = torch.zeros_like(odf_per)

        per_sample = (self.odf_weight * odf_per
                      + self.misori_weight * mis_per
                      + self.pf_weight * pf_per
                      + self.scatter_weight * sc_per
                      + self.pix_weight * pix_per
                      + self.pf_sharp_weight * pfs_per)
        if sample_weight is not None:
            w = sample_weight.to(per_sample.dtype)
            total = (per_sample * w).sum() / (w.sum() + 1e-8)
        else:
            total = per_sample.mean()
        info = {
            "loss_orient_odf": float(odf_per.mean().detach()),
            "loss_orient_misori": float(mis_per.mean().detach()),
            "loss_orient_pf": float(pf_per.mean().detach()),
            "loss_orient_scatter": float(sc_per.mean().detach()),
            "loss_orient_pix": float(pix_per.mean().detach()),
            "loss_orient_pfsharp": float(pfs_per.mean().detach()),
            "loss_orient": float(total.detach()),
        }
        return total, info


# ── Offline self-test (run: python -m microstructure_ed.orientation_loss) ──────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    loss_fn = OrientationLoss(pix_weight=0.7, pool_weight=0.3, pool_size=16)

    B, H, W = 2, 64, 64
    img = (torch.rand(B, 3, H, W) * 2 - 1)

    # (1) Identical pred == tgt -> ~0 loss.
    same, info_same = loss_fn(img, img.clone())
    print(f"[identical]  total={same.item():.6e}  info={info_same}")
    assert same.item() < 1e-4, "identical inputs should give ~0 loss"

    # (2) Random vs random -> strictly positive, bounded.
    img2 = (torch.rand(B, 3, H, W) * 2 - 1)
    rnd, info_rnd = loss_fn(img, img2)
    print(f"[random]     total={rnd.item():.6e}  info={info_rnd}")
    assert rnd.item() > 1e-3, "random inputs should give positive loss"

    # (3) Monotonicity: a small perturbation gives less loss than a large one.
    small = (img + 0.02 * torch.randn_like(img)).clamp(-1, 1)
    large = (img + 0.50 * torch.randn_like(img)).clamp(-1, 1)
    ls, _ = loss_fn(img, small)
    ll, _ = loss_fn(img, large)
    print(f"[mono]       small={ls.item():.6e}  large={ll.item():.6e}")
    assert ls.item() < ll.item(), "smaller perturbation should give smaller loss"

    # (4) Gradient flows back to the predicted image.
    pred = img.clone().requires_grad_(True)
    g, _ = loss_fn(pred, img2)
    g.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all(), "bad grad"
    print(f"[grad]       grad_norm={pred.grad.norm().item():.6e}")

    # Distributional loss self-tests
    dist_fn = OrientationDistributionLoss(n_pix=4096)

    # (5) Identical -> ~0.
    d_same, _ = dist_fn(img, img.clone())
    print(f"[dist ident] total={d_same.item():.6e}")
    assert d_same.item() < 1e-3, "identical inputs should give ~0 dist loss"

    # (6) KEY PROPERTY: spatially rearranged copy (roll + flip) of the SAME
    #     image -> near-zero loss (statistics preserved), while the
    #     per-pixel loss is large.
    img_roll = torch.roll(img, shifts=(H // 2, W // 3), dims=(2, 3)).flip(-1)
    d_roll, _ = dist_fn(img, img_roll)
    p_roll, _ = loss_fn(img, img_roll)
    print(f"[dist roll ] dist={d_roll.item():.6e}  per-pixel={p_roll.item():.6e}")
    assert d_roll.item() < 0.25 * p_roll.item(), "dist loss must ignore rearrangement"

    # (7) Different texture -> clearly positive, larger than rearranged.
    d_rnd, info_d = dist_fn(img, img2)
    print(f"[dist rand ] total={d_rnd.item():.6e}  info={info_d}")
    assert d_rnd.item() > d_roll.item(), "different stats must cost more than rearrangement"

    # (8) Gradient flows.
    pred2 = img.clone().requires_grad_(True)
    gd, _ = dist_fn(pred2, img2)
    gd.backward()
    assert pred2.grad is not None and torch.isfinite(pred2.grad).all(), "bad dist grad"
    print(f"[dist grad ] grad_norm={pred2.grad.norm().item():.6e}")

    # ── V1: metric-aware v2 terms ────────────────────────────────────────────
    def _rot_field(img_field, axis_angle):
        """Apply a GLOBAL rotation to the orientation field encoded in rgb."""
        q = rgb_to_quat_wxyz(img_field)                       # (B,4,H,W)
        B_, _, H_, W_ = q.shape
        qf = q.permute(0, 2, 3, 1).reshape(-1, 4)
        ax, ang = axis_angle
        ax = torch.tensor(ax, dtype=torch.float32)
        ax = ax / ax.norm()
        qr = torch.cat([torch.tensor([math.cos(ang / 2)]),
                        math.sin(ang / 2) * ax]).view(1, 4)
        qq = _qmul_wxyz(qr.expand_as(qf), qf)
        qq = torch.where(qq[:, :1] < 0, -qq, qq)
        # re-encode via stereographic projection S = q_xyz / (1 + q_w)
        S = qq[:, 1:] / (1.0 + qq[:, 0:1] + 1e-9)
        return S.view(B_, H_, W_, 3).permute(0, 3, 1, 2).clamp(-0.999, 0.999)

    v2 = OrientationDistributionLoss(
        n_pix=4096, odf_mode="sinkhorn", pf_weight=0.4, scatter_weight=0.1,
        odf_weight=0.3, misori_weight=0.2, n_refs=128, kappa=150.0,
    )

    # (9) identical -> ~0 for every v2 sub-term.
    t9, i9 = v2(img, img.clone())
    print(f"[v2 ident  ] total={t9.item():.6e}  info={i9}")
    assert t9.item() < 5e-3, "v2 identical should be ~0"

    # (10) PF + sinkhorn ignore spatial rearrangement.
    t10, i10 = v2(img, img_roll)
    print(f"[v2 roll   ] total={t10.item():.6e}  pf={i10['loss_orient_pf']:.3e}")
    assert i10["loss_orient_pf"] < 0.1, "PF must be arrangement-invariant"

    # (11) Sinkhorn-ODF monotone under global rotation drift 10/20/40 deg.
    prev = -1.0
    for deg in (10.0, 20.0, 40.0):
        rot = _rot_field(img, ([0.3, -0.5, 0.8], math.radians(deg)))
        _, ii = v2(img, rot)
        val = ii["loss_orient_odf"]
        print(f"[v2 rot {deg:4.0f}] sinkhorn_odf={val:.6e}  pf={ii['loss_orient_pf']:.4e}")
        assert val > prev, "sinkhorn ODF must grow with rotation drift"
        prev = val

    # (12) Scatter term separates concentration (kappa) at SAME mean axis:
    #      the anti-shrinkage property (right axis, too diffuse must cost).
    def _vmf_quat_field(kappa_c, B_=2, H_=64, W_=64):
        mu = torch.tensor([1.0, 0.0, 0.0, 0.0])
        qs = mu + torch.randn(B_ * H_ * W_, 4) / math.sqrt(kappa_c)
        qs = qs / qs.norm(dim=-1, keepdim=True)
        qs = torch.where(qs[:, :1] < 0, -qs, qs)
        S = qs[:, 1:] / (1.0 + qs[:, 0:1] + 1e-9)
        return S.view(B_, H_, W_, 3).permute(0, 3, 1, 2).clamp(-0.999, 0.999)

    sharp  = _vmf_quat_field(100.0)
    sharp2 = _vmf_quat_field(100.0)
    diffuse = _vmf_quat_field(20.0)
    _, is_same = v2(sharp, sharp2)
    _, is_diff = v2(diffuse, sharp)
    print(f"[v2 scatter] same-kappa={is_same['loss_orient_scatter']:.4e}  "
          f"diff-kappa={is_diff['loss_orient_scatter']:.4e}")
    assert is_diff["loss_orient_scatter"] > 2.0 * is_same["loss_orient_scatter"], \
        "scatter term must separate concentration levels"

    # (13) gradients finite + bounded terms.
    p3 = img.clone().requires_grad_(True)
    tv2, _ = v2(p3, img2)
    tv2.backward()
    assert p3.grad is not None and torch.isfinite(p3.grad).all(), "bad v2 grad"
    print(f"[v2 grad   ] grad_norm={p3.grad.norm().item():.6e}  total={tv2.item():.4f}")

    print("All orientation-loss self-tests passed.")
