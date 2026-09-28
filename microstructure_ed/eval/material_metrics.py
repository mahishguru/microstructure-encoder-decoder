"""
Materials-science metrics for encoder-decoder reconstruction evaluation.

Computes the following per-image metrics that complement the standard CV metrics
(SSIM, LPIPS, FID) for the ablation table:

    1. avg_grain_size_rel_err      — |<area>_recon - <area>_orig| / <area>_orig
    2. avg_aspect_ratio_abs_err    — |<AR>_recon - <AR>_orig|     (AR ∈ (0, 1])
    3. orientation_emd_deg         — Orientation Earth Mover's Distance (degrees):
                                     the 1-Wasserstein distance between the binned
                                     orientation distributions (ODFs) of the
                                     original and the reconstruction, with HCP
                                     disorientation as the ground metric. A
                                     spatial-correspondence-free texture-fidelity
                                     score (lower = better; identical ODFs give
                                     exactly 0 — a true zero floor).
    4. mean_disorientation_deg
                                   — disorientation (degrees) between the two
                                     volume-weighted MEAN orientations (the
                                     "disorientation of the means"). A cheap
                                     secondary sanity check on the dominant
                                     texture component; ≈ few ° for extruded Mg
                                     (weak discriminator — rank by metric 3).

    Per-grain orientation fidelity (the downstream crystal-plasticity
    simulation consumes ONE orientation per grain, so these assess
    orientation per segmented grain rather than per pixel; each grain's
    orientation is the sign-aligned mean quaternion of its pixels):

    5. grain_orientation_emd_deg     : area-weighted EMD (degrees) between the
                                       grain-orientation distributions (grain
                                       ODF) of orig vs recon, HCP-disorientation
                                       ground cost. Grain analogue of metric 3;
                                       0 = identical grain-orientation ODF.
    6. grain_mean_disorientation_deg : area-weighted mean HCP disorientation
                                       (degrees) over spatially matched grains
                                       (each orig grain paired with its max-
                                       overlap recon grain, kept if
                                       IoU >= GRAIN_IOU_THRESH). Answers "is
                                       each grain reoriented similarly?".
    7. grain_matched_frac            : fraction of original-grain AREA whose
                                       best-overlap match passes the IoU
                                       threshold (coverage of metric 6).
    8. n_grains_matched              : number of matched grain pairs.

Also exposes the underlying raw stats (n_grains, mean area, mean AR per image)
for both originals and reconstructions, so the ablation table can quote either
relative errors or absolute values.

Conventions
-----------
- Originals are 300×300 PNGs; reconstructions are typically 512×512.
  Both are resized to **300×300 NEAREST** (preserves discrete grain colors,
  which the per-pixel orientation decoder relies on) before any analysis.
- Segmentation: the **same** segmenter is applied to both the original
  and the reconstruction so that grain-size / aspect-ratio errors reflect
  reconstruction quality, not a method gap.
    * default ("sam") : SAM AutomaticMaskGenerator (vit_b, pps=32, iou=0.86,
                        min=200), matching the production codec used by the
                        inverse-design pipeline.
    * fallback ("clean") : connected-components on quantised colour
                           (colour_tol=8) — fast, no GPU, but over-fragments
                           noisy reconstructions.
- Orientation distance:
    * Each image is decoded pixel-wise to Bunge Euler via
      ``orientation_codec.decode_pixelwise`` using the class-mean quaternion
      from ``class_means.json`` (key = filename prefix, e.g. "AZ31_extruded").
    * Per-pixel disorientation is folded into the HCP fundamental zone
      using the 12 HCP point-group symmetries.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from microstructure_ed.config import ASSETS_DIR
from microstructure_ed.segmentation import default_sam_checkpoint


# ── Constants ──────────────────────────────────────────────────

DEFAULT_CLASS_MEANS = os.environ.get(
    "CLASS_MEANS_JSON", os.path.join(ASSETS_DIR, "class_means.json"))
DEFAULT_SAM_CHECKPOINT = default_sam_checkpoint()
NATIVE_SIZE = 300  # native resolution of encoded originals

# Orientation EMD (binned-ODF optimal transport). The fundamental zone is
# discretized into a fixed BINS^3 Rodrigues grid (range set per-original from
# the ORIENT_EMD_RANGE_PCT percentile of |Rodrigues|). The EMD is the exact
# 1-Wasserstein distance between the two normalized orientation histograms,
# so identical ODFs map to exactly 0 (true zero floor). BINS=8 gives ~few-deg
# resolution with a tractable transportation LP (~1 s/pair).
ORIENT_EMD_BINS = 8
ORIENT_EMD_RANGE_PCT = 99.5

# Per-grain orientation metrics: grains smaller than GRAIN_MIN_AREA pixels
# are ignored (segmentation noise). An original grain is matched to a
# reconstruction grain when their pixel IoU >= GRAIN_IOU_THRESH.
GRAIN_MIN_AREA = 4
GRAIN_IOU_THRESH = 0.3

# HCP symmetries cached on first use, in (12, 4) [w, x, y, z] form.
_HCP_SYMS_WXYZ: np.ndarray | None = None
_HCP_LOCK = threading.Lock()

# SAM mask generator cache (one per process).
_SAM_AMG = None
_SAM_LOCK = threading.Lock()


# ── Image IO ─────────────────────────────────────────────────────────────────

def load_uint8_at_native(path: Path) -> np.ndarray:
    """Load a PNG as RGB uint8 resized to NATIVE_SIZE × NATIVE_SIZE (NEAREST)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (NATIVE_SIZE, NATIVE_SIZE):
            im = im.resize((NATIVE_SIZE, NATIVE_SIZE), Image.NEAREST)
        return np.asarray(im, dtype=np.uint8)


# ── Class-mean lookup ────────────────────────────────────────────────────────

def load_class_means(path: str | Path) -> Dict[str, np.ndarray]:
    raw = json.loads(Path(path).read_text())
    return {k: np.asarray(v, dtype=np.float64) for k, v in raw.items()}


def class_key_for_stem(stem: str, available: list[str]) -> str:
    """Map "AZ31_extruded_1003" → "AZ31_extruded" by longest-prefix match."""
    for key in sorted(available, key=len, reverse=True):
        if stem.startswith(key):
            return key
    raise KeyError(f"no class_means key matches stem={stem!r} (have {available})")


# ── Segmentation back-ends ───────────────────────────────────────────────────

def segment_clean_uint8(img_u8: np.ndarray, colour_tol: int = 1) -> np.ndarray:
    """Connected-components with per-channel colour tolerance (orig path)."""
    from orientation_codec.dataset import segment_grains
    return segment_grains(img_u8, colour_tol=int(colour_tol))


def segment_sam_uint8(img_u8: np.ndarray) -> np.ndarray:
    """SAM mask generator → 1..N integer label map (no zero region).

    Delegates to ``microstructure_ed.segmentation._sam_segment`` so this script
    uses the exact same SAM configuration and post-processing as the
    inverse-design optimization pipeline (single source of truth).
    """
    from microstructure_ed.segmentation import _sam_segment
    return _sam_segment(img_u8, checkpoint=DEFAULT_SAM_CHECKPOINT)


def segment_sam_cached(
    img_u8: np.ndarray,
    cache_path: Optional[str] = None,
) -> np.ndarray:
    """``segment_sam_uint8`` with an optional on-disk label-map cache.

    SAM is the dominant per-image cost (~1.3 s/segmentation). The *original*
    image is segmented identically for every model variant, so caching its
    label map lets it be computed once and reused across all variants. The
    write is atomic (tmp + os.replace) so concurrent readers never observe a
    partial file; a corrupt/partial cache simply triggers recomputation.
    """
    import os
    if cache_path is not None and os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            pass  # corrupt/partial cache -> recompute below
    lbl = segment_sam_uint8(img_u8)
    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp = f"{cache_path}.tmp.{os.getpid()}"
        try:
            with open(tmp, "wb") as fh:
                np.save(fh, lbl)
            os.replace(tmp, cache_path)
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
    return lbl


# ── Grain shape statistics ───────────────────────────────────────────────────

def grain_stats(label_map: np.ndarray, min_area: int = 4) -> Dict[str, float]:
    """Return n_grains, mean area (px), mean aspect ratio (minor/major in (0,1])."""
    from skimage.measure import regionprops_table
    props = regionprops_table(
        label_map.astype(np.int32),
        properties=("label", "area", "axis_minor_length", "axis_major_length"),
    )
    areas = np.asarray(props["area"], dtype=np.float64)
    minor = np.asarray(props["axis_minor_length"], dtype=np.float64)
    major = np.asarray(props["axis_major_length"], dtype=np.float64)

    keep = areas >= float(min_area)
    if not keep.any():
        return {"n_grains": 0, "mean_area": float("nan"), "mean_aspect_ratio": float("nan")}

    areas = areas[keep]; minor = minor[keep]; major = major[keep]
    # Aspect ratio = minor/major ∈ (0, 1]. Guard against degenerate major=0.
    safe = major > 1e-6
    ar = np.full_like(major, np.nan)
    ar[safe] = minor[safe] / major[safe]
    ar = ar[np.isfinite(ar)]

    return {
        "n_grains": int(len(areas)),
        "mean_area": float(areas.mean()),
        "mean_aspect_ratio": float(ar.mean()) if ar.size else float("nan"),
    }


# ── Orientation field & disorientation ───────────────────────────────────────

def _hcp_syms_wxyz() -> np.ndarray:
    """12 HCP point-group symmetries in (12, 4) [w, x, y, z] convention."""
    global _HCP_SYMS_WXYZ
    if _HCP_SYMS_WXYZ is not None:
        return _HCP_SYMS_WXYZ
    with _HCP_LOCK:
        if _HCP_SYMS_WXYZ is None:
            from orientation_codec.symmetry import HCP_SYMMETRIES  # list of scipy R
            arr = []
            for sym in HCP_SYMMETRIES:
                qxyzw = sym.as_quat()  # scipy = [x, y, z, w]
                arr.append([qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]])
            _HCP_SYMS_WXYZ = np.asarray(arr, dtype=np.float64)
    return _HCP_SYMS_WXYZ


def _quat_mul(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product of (..., 4) [w, x, y, z] quaternions."""
    pw, px, py, pz = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        pw * qw - px * qx - py * qy - pz * qz,
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
    ], axis=-1)


def _quat_conj(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def euler_to_quat_wxyz(eul_zxz: np.ndarray) -> np.ndarray:
    """Bunge ZXZ Euler (radians) → quaternion (w, x, y, z)."""
    p1 = eul_zxz[..., 0] * 0.5
    P  = eul_zxz[..., 1] * 0.5
    p2 = eul_zxz[..., 2] * 0.5
    cP, sP = np.cos(P), np.sin(P)
    cm, sm = np.cos(p1 - p2), np.sin(p1 - p2)
    cp, sp = np.cos(p1 + p2), np.sin(p1 + p2)
    q = np.stack([cP * cp, sP * cm, sP * sm, cP * sp], axis=-1)
    q /= (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
    return q


def disorientation_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Vectorized HCP disorientation between two (N, 4) [w, x, y, z] quat sets.

    Returns the per-row disorientation angle in degrees, ∈ [0, 90°] for HCP.
    """
    syms = _hcp_syms_wxyz()  # (12, 4)
    misor = _quat_mul(_quat_conj(q1), q2)  # (N, 4)
    # |w| of (sym * misor) — maximizing |w| minimizes the rotation angle.
    best_w = np.zeros(misor.shape[0], dtype=np.float64)
    for sym in syms:
        eq = _quat_mul(sym[None, :], misor)  # (N, 4)
        w_abs = np.abs(eq[..., 0])
        best_w = np.maximum(best_w, w_abs)
    angle_rad = 2.0 * np.arccos(np.clip(best_w, 0.0, 1.0))
    return np.degrees(angle_rad)


def orientation_field_quat(img_u8: np.ndarray, mean_q_xyzw: np.ndarray) -> np.ndarray:
    """Decode an RGB image to a (H*W, 4) wxyz quaternion field via the codec."""
    from orientation_codec.dataset import decode_pixelwise
    eul = decode_pixelwise(img_u8, mean_q_xyzw)  # (H, W, 3) Bunge ZXZ in [0, 2π)
    q = euler_to_quat_wxyz(eul.reshape(-1, 3))
    return q  # (N, 4)



# ── Orientation EMD (texture Earth Mover's Distance, in degrees) ──────────────

def disorientation_matrix(qA: np.ndarray, qB: np.ndarray, chunk: int = 256) -> np.ndarray:
    """All-pairs HCP disorientation (degrees) between two quaternion sets.

    Parameters
    ----------
    qA : (nA, 4) wxyz quaternions.
    qB : (nB, 4) wxyz quaternions.
    chunk : rows of ``qA`` processed at once (caps peak memory).

    Returns
    -------
    (nA, nB) float64 matrix; entry [i, j] is the crystallographic
    disorientation angle in degrees (∈ [0, 90°] for HCP) between qA[i]
    and qB[j], minimized over the 12 HCP point-group symmetries.
    """
    syms = _hcp_syms_wxyz()             # (12, 4)
    qA_conj = _quat_conj(np.ascontiguousarray(qA, dtype=np.float64))
    qB = np.ascontiguousarray(qB, dtype=np.float64)
    nA, nB = qA_conj.shape[0], qB.shape[0]
    out = np.empty((nA, nB), dtype=np.float64)
    for i0 in range(0, nA, chunk):
        i1 = min(i0 + chunk, nA)
        # misor[c, nB, 4] = conj(qA_i) ⊗ qB_j
        misor = _quat_mul(qA_conj[i0:i1, None, :], qB[None, :, :])  # (c, nB, 4)
        best_w = np.zeros((i1 - i0, nB), dtype=np.float64)
        for sym in syms:
            eq_w_abs = np.abs(_quat_mul(sym[None, None, :], misor)[..., 0])
            np.maximum(best_w, eq_w_abs, out=best_w)
        out[i0:i1] = np.degrees(2.0 * np.arccos(np.clip(best_w, 0.0, 1.0)))
    return out


def _transport_emd(p: np.ndarray, q: np.ndarray, D: np.ndarray) -> float:
    """Exact 1-Wasserstein distance between two discrete distributions ``p``
    and ``q`` (probability vectors over a shared K-point support) under the
    ground-cost matrix ``D`` (K×K), solved as a balanced transportation LP.

    Returns the optimal transport cost (same units as ``D``). If ``p == q``
    the cost is 0 (mass stays on the zero-cost diagonal), giving a true zero
    self-distance.
    """
    from scipy.optimize import linprog
    from scipy import sparse

    K = len(p)
    var = np.arange(K * K)
    # Marginals: row sums = p (K eqs), col sums = q (K eqs); var index = i*K + j.
    rows = np.concatenate([np.repeat(np.arange(K), K), K + np.tile(np.arange(K), K)])
    A_eq = sparse.coo_matrix(
        (np.ones(2 * K * K), (rows, np.concatenate([var, var]))),
        shape=(2 * K, K * K),
    )
    res = linprog(
        D.ravel(), A_eq=A_eq, b_eq=np.concatenate([p, q]),
        bounds=(0.0, None), method="highs",
    )
    return max(0.0, float(res.fun))


def _fz_rodrigues(q_wxyz: np.ndarray) -> np.ndarray:
    """Rodrigues vector q_xyz / q_w, after sign-flipping each quaternion to the
    w >= 0 hemisphere (q and -q are the same rotation). Without this flip the
    majority of pixels (whose Bunge (phi1+phi2)/2 exceeds pi/2, giving w < 0)
    would map to spurious near-infinite Rodrigues vectors and corrupt the ODF
    grid."""
    q = np.where(q_wxyz[:, :1] < 0.0, -q_wxyz, q_wxyz)
    w = np.clip(q[:, 0], 1e-9, 1.0)
    return q[:, 1:] / w[:, None]


def _orientation_emd_value(
    q_orig: np.ndarray,
    q_recon: np.ndarray,
    *,
    bins: int = ORIENT_EMD_BINS,
    range_pct: float = ORIENT_EMD_RANGE_PCT,
    weights_orig: Optional[np.ndarray] = None,
    weights_recon: Optional[np.ndarray] = None,
) -> float:
    """Orientation Earth Mover's Distance (degrees) between the orientation
    distributions (ODFs) of two images — a spatial-correspondence-free,
    interpretable texture-fidelity metric for comparing reconstruction models.

    Both pixel-wise orientation fields are binned into a fixed ``bins**3``
    Rodrigues grid spanning the fundamental zone (range set from the original's
    ``range_pct`` percentile). The metric is the exact 1-Wasserstein distance
    between the two normalized orientation histograms, using HCP disorientation
    between bin centres as the ground metric — i.e. the mean crystallographic
    rotation (degrees) needed to optimally morph the reconstruction's texture
    into the original's.

    Because the two histograms share a fixed grid, **identical ODFs give
    exactly 0** (true zero floor). The metric is invariant to the per-class
    mean quaternion and to fundamental-zone folding, and — unlike pixel-
    registered disorientation (which saturates near the HCP random-pair
    baseline of ~60°) — compares the orientation *statistics*, not the spatial
    realization.

    Returns
    -------
    dict with ``orientation_emd_deg`` — the EMD in degrees (0 = identical ODF).
    """
    ro = _fz_rodrigues(np.ascontiguousarray(q_orig, dtype=np.float64))
    rr = _fz_rodrigues(np.ascontiguousarray(q_recon, dtype=np.float64))

    # Fixed grid from the original (shared by all recons of this image ⇒ fair
    # cross-model comparison; self-distance is 0 regardless of the range).
    rmax = np.percentile(np.abs(ro), range_pct, axis=0) * 1.05 + 1e-6
    edges = [np.linspace(-rmax[i], rmax[i], bins + 1) for i in range(3)]
    centers = [0.5 * (e[1:] + e[:-1]) for e in edges]

    Ho, _ = np.histogramdd(np.clip(ro, -rmax, rmax), bins=edges,
                           weights=weights_orig)
    Hr, _ = np.histogramdd(np.clip(rr, -rmax, rmax), bins=edges,
                           weights=weights_recon)
    p, q = Ho.ravel(), Hr.ravel()

    supp = (p > 0) | (q > 0)            # occupied bins in either ODF
    p, q = p[supp], q[supp]
    psum, qsum = p.sum(), q.sum()
    if psum <= 0 or qsum <= 0:
        return float("nan")
    p, q = p / psum, q / qsum

    gx, gy, gz = np.meshgrid(centers[0], centers[1], centers[2], indexing="ij")
    R = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)[supp]  # (K, 3)
    # Rodrigues → wxyz quaternion: q = [1, r] / sqrt(1 + |r|²).
    w = 1.0 / np.sqrt(1.0 + (R * R).sum(axis=1))
    Q = np.concatenate([w[:, None], R * w[:, None]], axis=1)          # (K, 4)

    D = disorientation_matrix(Q, Q)     # (K, K) ground cost in degrees
    return _transport_emd(p, q, D)


def orientation_emd(
    q_orig: np.ndarray,
    q_recon: np.ndarray,
    *,
    bins: int = ORIENT_EMD_BINS,
    range_pct: float = ORIENT_EMD_RANGE_PCT,
) -> Dict[str, float]:
    """Pixel-wise orientation EMD (degrees). Thin wrapper around
    :func:`_orientation_emd_value` (unweighted, one count per pixel)."""
    return {"orientation_emd_deg": _orientation_emd_value(
        q_orig, q_recon, bins=bins, range_pct=range_pct)}


def mean_disorientation_deg(q_orig: np.ndarray, q_recon: np.ndarray) -> float:
    """Disorientation (degrees) between the MEAN orientations of the two
    fields — the *disorientation of the means*, not the mean of disorientations.

    Each field is first collapsed to a single volume-weighted mean quaternion
    (its texture centre-of-mass); the metric is the one disorientation (smallest
    symmetrically reduced misorientation angle) between those two means. The
    broad within-image fibre spread cancels in the averaging, so this isolates
    the dominant texture component / mean fibre axis (≈ few ° for extruded Mg).
    It is a cheap *secondary* descriptor: a sanity check that the recon's
    dominant component is correct. (Note it is a weak discriminator — random
    real specimens also sit at ~4° — so rank models by ``orientation_emd_deg``,
    which has a true zero floor.)

    Contrast with the old pixel-registered ``mean disorientation`` (≈ 60°),
    which averages per-location disorientations (mean OF disorientations) and
    saturates at the HCP random-pair baseline because the recon is a different
    spatial realization.
    """
    def _mean_q(q):
        ref = q[np.argmax(q[:, 0])]
        s = np.sign(q @ ref); s[s == 0] = 1.0
        mq = (q * s[:, None]).mean(axis=0)
        return mq / (np.linalg.norm(mq) + 1e-12)
    return float(disorientation_deg(_mean_q(q_orig)[None, :], _mean_q(q_recon)[None, :])[0])


def _grain_mean_quats(label_map, q_field, *, min_area=GRAIN_MIN_AREA):
    """Per-grain mean wxyz quaternion + area from a label map and its aligned
    per-pixel quaternion field.

    label_map : (H, W) int label map (row-major).
    q_field   : (H*W, 4) wxyz quaternions, row i aligned to pixel i of the
                row-major flattened label_map.
    min_area  : grains smaller than this (pixels) are dropped.

    Returns (mean_q (G,4), areas (G,), lab2idx (max_label+2,) mapping a label
    value to its compact grain index or -1 if dropped/absent).
    """
    lab = np.ascontiguousarray(label_map).reshape(-1).astype(np.int64)
    q = np.ascontiguousarray(q_field, dtype=np.float64)
    uniq, inv = np.unique(lab, return_inverse=True)
    G = uniq.size
    areas = np.bincount(inv, minlength=G).astype(np.float64)
    _, first_occ = np.unique(inv, return_index=True)
    ref = q[first_occ]                                   # (G, 4) per-grain ref
    dots = np.einsum("nk,nk->n", q, ref[inv])
    qa = q * np.where(dots < 0.0, -1.0, 1.0)[:, None]    # align hemisphere
    acc = np.empty((G, 4), dtype=np.float64)
    for k in range(4):
        acc[:, k] = np.bincount(inv, weights=qa[:, k], minlength=G)
    mean_q = acc / (np.linalg.norm(acc, axis=1, keepdims=True) + 1e-12)

    keep = areas >= float(min_area)
    mean_q, areas_k, uniq_k = mean_q[keep], areas[keep], uniq[keep]
    lab2idx = np.full(int(lab.max()) + 2, -1, dtype=np.int64)
    if uniq_k.size:
        lab2idx[uniq_k] = np.arange(uniq_k.size)
    return mean_q, areas_k, lab2idx


def grain_orientation_metrics(lbl_orig, q_orig, lbl_recon, q_recon, *,
                              min_area=GRAIN_MIN_AREA,
                              iou_thresh=GRAIN_IOU_THRESH):
    """Per-grain orientation fidelity between an original and its reconstruction.

    The downstream crystal-plasticity simulation consumes one orientation per
    grain, so orientation fidelity is assessed per segmented grain rather than
    per pixel. Each grain's orientation is the sign-aligned mean quaternion of
    its pixels.

    grain_orientation_emd_deg
        Area-weighted EMD (degrees) between the distributions of grain
        orientations (orig vs recon grain-ODF, HCP-disorientation ground cost).
        0 = identical grain-orientation distribution. Grain analogue of
        orientation_emd_deg; no spatial correspondence needed.
    grain_mean_disorientation_deg
        Area-weighted mean HCP disorientation (degrees) over spatially matched
        grains: each original grain is matched to its maximally overlapping
        reconstruction grain (kept if IoU >= iou_thresh) and the disorientation
        between the two grain orientations is taken. Answers "is each grain
        reoriented similarly in the reconstruction?".
    grain_matched_frac
        Fraction of original-grain area whose best-overlap match passes the IoU
        threshold (coverage of the disorientation average).
    n_grains_matched
        Number of matched grain pairs.
    """
    out = {
        "grain_orientation_emd_deg": float("nan"),
        "grain_mean_disorientation_deg": float("nan"),
        "grain_matched_frac": float("nan"),
        "n_grains_matched": 0.0,
    }
    mqo, ao, lut_o = _grain_mean_quats(lbl_orig, q_orig, min_area=min_area)
    mqr, ar, lut_r = _grain_mean_quats(lbl_recon, q_recon, min_area=min_area)
    if mqo.shape[0] == 0 or mqr.shape[0] == 0:
        return out

    out["grain_orientation_emd_deg"] = _orientation_emd_value(
        mqo, mqr, weights_orig=ao, weights_recon=ar)

    lo = np.ascontiguousarray(lbl_orig).reshape(-1).astype(np.int64)
    lr = np.ascontiguousarray(lbl_recon).reshape(-1).astype(np.int64)
    io_ = lut_o[lo]
    ir_ = lut_r[lr]
    valid = (io_ >= 0) & (ir_ >= 0)
    if not np.any(valid):
        out["grain_matched_frac"] = 0.0
        return out
    io_, ir_ = io_[valid], ir_[valid]
    Gr = mqr.shape[0]
    key = io_ * Gr + ir_
    pk, pc = np.unique(key, return_counts=True)
    p_oi = pk // Gr
    p_rj = pk % Gr
    order = np.lexsort((-pc, p_oi))
    p_oi, p_rj, pc = p_oi[order], p_rj[order], pc[order]
    firsts = np.ones(p_oi.size, dtype=bool)
    firsts[1:] = p_oi[1:] != p_oi[:-1]
    b_oi, b_rj = p_oi[firsts], p_rj[firsts]
    b_ov = pc[firsts].astype(np.float64)
    iou = b_ov / (ao[b_oi] + ar[b_rj] - b_ov)
    matched = iou >= float(iou_thresh)
    if not np.any(matched):
        out["grain_matched_frac"] = 0.0
        return out
    m_oi, m_rj = b_oi[matched], b_rj[matched]
    w = ao[m_oi]
    dis = disorientation_deg(mqo[m_oi], mqr[m_rj])        # (M,)
    out["grain_mean_disorientation_deg"] = float(np.sum(w * dis) / (np.sum(w) + 1e-12))
    out["grain_matched_frac"] = float(ao[m_oi].sum() / (ao.sum() + 1e-12))
    out["n_grains_matched"] = float(matched.sum())
    return out



# ── Top-level per-image driver ───────────────────────────────────────────────

def material_metrics_pair(
    orig_u8: np.ndarray,
    recon_u8: np.ndarray,
    mean_q_xyzw: np.ndarray,
    *,
    seg_recon: str = "sam",
    orig_label_path: Optional[str] = None,
    recon_label_path: Optional[str] = None,
) -> Dict[str, float]:
    """Compute per-image materials-science metrics for one (orig, recon) pair.

    Parameters
    ----------
    orig_u8, recon_u8
        Both (H, W, 3) uint8 arrays at the same resolution (caller is
        expected to have resized to ``NATIVE_SIZE`` already).
    mean_q_xyzw
        Length-4 numpy array — the class-mean quaternion in scipy [x, y, z, w]
        order (this is what ``class_means.json`` stores and what
        ``decode_pixelwise`` expects).
    seg_recon
        "sam" (default) or "clean" — segmenter applied to **both** the
        original and the reconstruction (must match for a fair grain-stat
        comparison; otherwise the size/AR error is dominated by the
        segmenter gap rather than reconstruction quality).
    """
    out: Dict[str, float] = {}

    # 1) Grain shape stats — same segmenter on both sides
    if seg_recon == "sam":
        lbl_orig  = segment_sam_cached(orig_u8, orig_label_path)
        lbl_recon = segment_sam_cached(recon_u8, recon_label_path)
    elif seg_recon == "clean":
        # Loose tolerance handles noisy reconstructions; on the clean
        # encoder originals it just merges near-duplicate colours.
        lbl_orig  = segment_clean_uint8(orig_u8,  colour_tol=8)
        lbl_recon = segment_clean_uint8(recon_u8, colour_tol=8)
    else:
        raise ValueError(f"unknown seg_recon={seg_recon!r}")

    s_o = grain_stats(lbl_orig)
    s_r = grain_stats(lbl_recon)

    out["n_grains_orig"]  = float(s_o["n_grains"])
    out["n_grains_recon"] = float(s_r["n_grains"])
    out["mean_area_orig"]  = s_o["mean_area"]
    out["mean_area_recon"] = s_r["mean_area"]
    out["mean_aspect_ratio_orig"]  = s_o["mean_aspect_ratio"]
    out["mean_aspect_ratio_recon"] = s_r["mean_aspect_ratio"]

    # Relative error on grain size (guard against zero/NaN)
    if (np.isfinite(s_o["mean_area"]) and s_o["mean_area"] > 0
            and np.isfinite(s_r["mean_area"])):
        out["avg_grain_size_rel_err"] = float(
            abs(s_r["mean_area"] - s_o["mean_area"]) / s_o["mean_area"]
        )
    else:
        out["avg_grain_size_rel_err"] = float("nan")

    # Absolute error on aspect ratio
    if np.isfinite(s_o["mean_aspect_ratio"]) and np.isfinite(s_r["mean_aspect_ratio"]):
        out["avg_aspect_ratio_abs_err"] = float(
            abs(s_r["mean_aspect_ratio"] - s_o["mean_aspect_ratio"])
        )
    else:
        out["avg_aspect_ratio_abs_err"] = float("nan")

    # 2) Orientation distribution distance — Earth Mover's Distance (degrees).
    #    Spatial-correspondence-free texture-fidelity metric: the 1-Wasserstein
    #    distance between the binned orientation distributions (ODFs) of orig
    #    and recon, using HCP disorientation as the ground metric. Identical
    #    ODFs give exactly 0 (true zero floor). Invariant to the class-mean
    #    quaternion. Replaces the saturating pixel-registered disorientation.
    try:
        q_o = orientation_field_quat(orig_u8, mean_q_xyzw)   # (N, 4) wxyz
        q_r = orientation_field_quat(recon_u8, mean_q_xyzw)  # (N, 4) wxyz
        out.update(orientation_emd(q_o, q_r))
        out["mean_disorientation_deg"] = mean_disorientation_deg(q_o, q_r)
        # 3) Per-grain orientation fidelity: the downstream simulation
        #    consumes one orientation per grain, so compare orientations per
        #    segmented grain (matched orig<->recon by pixel IoU) in addition
        #    to the pixel-wise ODF metrics above.
        out.update(grain_orientation_metrics(lbl_orig, q_o, lbl_recon, q_r))
    except Exception as exc:  # decoder hiccup (e.g. all-zero image)
        out["orientation_emd_deg"]     = float("nan")
        out["mean_disorientation_deg"] = float("nan")
        out["grain_orientation_emd_deg"]     = float("nan")
        out["grain_mean_disorientation_deg"] = float("nan")
        out["grain_matched_frac"]           = float("nan")
        out["n_grains_matched"]             = float("nan")
        out["_orientation_error"] = repr(exc)  # debug hint, not aggregated

    return out


# Keys that compute_metrics.py will write to CSV / aggregate in summary.
PER_IMAGE_KEYS: Tuple[str, ...] = (
    "n_grains_orig", "n_grains_recon",
    "mean_area_orig", "mean_area_recon",
    "mean_aspect_ratio_orig", "mean_aspect_ratio_recon",
    "avg_grain_size_rel_err",
    "avg_aspect_ratio_abs_err",
    "orientation_emd_deg", "mean_disorientation_deg",
    # Per-grain orientation fidelity (downstream simulation is per grain)
    "grain_orientation_emd_deg", "grain_mean_disorientation_deg",
    "grain_matched_frac", "n_grains_matched",
)
