#!/usr/bin/env python
"""B2/B3: post-hoc ODF calibration of FM-DiT reconstructions.

Motivation (Phase A findings): the ODE rollout collapses texture sharpness
toward the class mean (maxMRD ratio ~0.5) and the conditioning z is texture
blind, so no sampler/guidance tweak can fully restore the per-image ODF. This
stage restores it EXACTLY, using the input image the reconstruction was
conditioned on (legitimate for reconstruction: input is available at recon
time by definition).

Two calibration levels, applied to recon RGB (300x300, uint8):
  B2 (radial): segmentation-free monotone quantile match of the stereographic
      radius |S| (recon -> input marginal), direction preserved per pixel.
      Fixes the misorientation-from-mean marginal; leaves directions alone.
  B3 (grain OT): on top of B2, segment recon (colour_tol CC) + input (exact CC),
      per-grain median stereo -> capacity-constrained greedy transport in HCP
      disorientation distance -> repaint each recon grain with its assigned
      INPUT grain colour. Guarantees the grain ODF is an area-weighted
      resampling of the input's grain ODF while keeping recon geometry.

Outputs, next to each recon: <stem>_calibB2.png, <stem>_calibB3.png
Scores (TI/maxMRD ratios + folded-angle EMD) printed per stage.

Usage:
  python -m microstructure_ed.eval.odf_calibrate --recon_root eval_outputs/<name> \
      [--recon_name recon_vitfmdit_z1280.png] [--tol 8] [--min_area 4]
"""
from __future__ import annotations
import os
import argparse, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(os.environ.get("MSED_ROOT", os.getcwd())).resolve()

from microstructure_ed.orientation_loss import (                      # noqa: E402
    OrientationDistributionLoss, rgb_to_quat_wxyz, disorientation_cos_half,
    _build_hcp_syms,
)
from orientation_codec.dataset import segment_grains        # noqa: E402

SYMS = _build_hcp_syms().double()                           # (12,4) wxyz


# ── scoring ──────────────────────────────────────────────────────────────────
class Scorer:
    def __init__(self, n_pix=8192, pf_grid=512, pf_kappa=256.0, seed=0):
        self.m = OrientationDistributionLoss(
            n_pix=n_pix, pf_grid=pf_grid, pf_kappa=pf_kappa,
            pf_weight=1.0, pf_prismatic=True, seed=seed)
        self.n_pix = n_pix
        self.seed = seed

    def stats(self, u8):
        x = torch.from_numpy(u8.astype(np.float32) / 255.0 * 2 - 1).permute(2, 0, 1)[None]
        q = rgb_to_quat_wxyz(x).flatten(2).transpose(1, 2)  # (1,HW,4)
        g = torch.Generator().manual_seed(self.seed)
        idx = torch.randperm(q.shape[1], generator=g)[: self.n_pix]
        qn = q[:, idx]
        iq = torch.zeros_like(qn); iq[..., 0] = 1.0
        ch = disorientation_cos_half(iq, qn, self.m.syms)
        fz = torch.rad2deg(2 * torch.acos(ch.clamp(-1, 1)))
        h = self.m._pf_hist(qn)
        G = self.m.pf_nodes.shape[0]
        p, pr = h[:, :G], h[:, G:]
        return dict(fz=fz,
                    ti_b=(G * (p ** 2).sum(-1)).item(), mx_b=(G * p.amax(-1)).item(),
                    ti_p=(G * (pr ** 2).sum(-1)).item(), mx_p=(G * pr.amax(-1)).item())


def emd_deg(a, b):
    a, _ = torch.sort(a.flatten()); b, _ = torch.sort(b.flatten())
    n = min(len(a), len(b))
    ia = torch.linspace(0, len(a) - 1, n).long()
    ib = torch.linspace(0, len(b) - 1, n).long()
    return (a[ia] - b[ib]).abs().mean().item()


# ── B2: radial quantile match ────────────────────────────────────────────────
def calib_radial(recon_u8, orig_u8):
    S_r = recon_u8.astype(np.float64) / 255.0 * 2 - 1       # (H,W,3)
    S_o = orig_u8.astype(np.float64) / 255.0 * 2 - 1
    r_r = np.linalg.norm(S_r, axis=-1)                       # (H,W)
    r_o = np.sort(np.linalg.norm(S_o, axis=-1).ravel())
    flat = r_r.ravel()
    order = np.argsort(flat, kind="stable")
    ranks = np.empty_like(order); ranks[order] = np.arange(flat.size)
    # map recon rank -> orig quantile value
    qpos = (ranks / max(flat.size - 1, 1) * (r_o.size - 1)).astype(np.int64)
    r_new = r_o[qpos].reshape(r_r.shape)
    scale = np.where(r_r > 1e-9, r_new / np.maximum(r_r, 1e-9), 0.0)
    S_new = S_r * scale[..., None]
    out = np.clip((np.clip(S_new, -1, 1) + 1) / 2 * 255.0, 0, 255).round().astype(np.uint8)
    return out


# ── B3: grain-level transport ────────────────────────────────────────────────
def _grain_median_stereo(u8, labels, min_area):
    S = u8.astype(np.float64) / 255.0 * 2 - 1
    flat = S.reshape(-1, 3)
    lab = labels.ravel()
    uniq, inv, cnt = np.unique(lab, return_inverse=True, return_counts=True)
    keep = cnt >= min_area
    med = np.zeros((uniq.size, 3))
    for k in np.nonzero(keep)[0]:
        med[k] = np.median(flat[inv == k], axis=0)
    return uniq[keep], med[keep], cnt[keep].astype(np.float64)


def _stereo_to_q_wxyz(S):
    S2 = (S ** 2).sum(-1, keepdims=True)
    q = np.concatenate([(1 - S2) / (1 + S2), 2 * S / (1 + S2)], axis=-1)  # wxyz
    q /= np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12
    return q * np.where(q[..., :1] < 0, -1.0, 1.0)


def _diso_matrix(qa, qb):
    """(Go,4),(Gr,4) wxyz -> (Go,Gr) HCP-folded misorientation degrees."""
    ta = torch.from_numpy(qa).double()
    tb = torch.from_numpy(qb).double()
    ch = disorientation_cos_half(ta[:, None, :], tb[None, :, :], SYMS)
    return torch.rad2deg(2 * torch.acos(ch.clamp(-1, 1))).numpy()


from microstructure_ed.segmentation import default_sam_checkpoint  # noqa: E402

SAM_CKPT = default_sam_checkpoint()


def segment_sam(img_u8, cache_path=None):
    """SAM AutomaticMaskGenerator label map via microstructure_ed.segmentation
    (_sam_segment: vit_b, pps=32, iou=0.86, min_region=200, Voronoi fill).
    Identical segmenter to material_metrics.py and png_to_dream3d, so the
    calibrated grains coincide with the grains the simulation codec will
    extract downstream. Cached to .npy next to the recon (SAM ~1.3 s/img)."""
    if cache_path is not None and Path(cache_path).exists():
        return np.load(cache_path)
    from microstructure_ed.segmentation import _sam_segment
    lbl = _sam_segment(np.ascontiguousarray(img_u8), checkpoint=SAM_CKPT)
    if cache_path is not None:
        np.save(cache_path, lbl)
    return lbl


def _merge_small(lbl, min_px):
    """Absorb fragments < min_px into the nearest large fragment (EDT fill)."""
    from scipy import ndimage
    sizes = np.bincount(lbl.ravel())
    small = sizes[lbl] < min_px
    if not small.any() or small.all():
        return lbl
    _, (iy, ix) = ndimage.distance_transform_edt(small, return_indices=True)
    out = lbl.copy()
    out[small] = lbl[iy[small], ix[small]]
    return out


def _absorb_specks(img_u8, min_px):
    """Post-paint cleanup: recolor connected components < min_px from the
    nearest pixel of a large component (EDT). Bounded ODF impact."""
    from scipy import ndimage
    lbl = segment_grains(img_u8, colour_tol=2)
    sizes = np.bincount(lbl.ravel())
    small = sizes[lbl] < min_px
    if not small.any() or small.all():
        return img_u8
    _, (iy, ix) = ndimage.distance_transform_edt(small, return_indices=True)
    out = img_u8.copy()
    out[small] = img_u8[iy[small], ix[small]]
    return out


def _split_large(lbl, cell_px):
    """Subdivide fragments > 1.6*cell_px into equiaxed cells via coordinate
    k-means (4 Lloyd iterations). Prevents giant smeared regions from being
    painted as image-wide slabs."""
    uniq, cnt = np.unique(lbl, return_counts=True)
    out = lbl.copy()
    nxt = int(lbl.max()) + 1
    for u, c in zip(uniq, cnt):
        k = int(round(c / float(cell_px)))
        if c <= 1.6 * cell_px or k < 2:
            continue
        P = np.argwhere(lbl == u).astype(np.float32)
        order = np.argsort(P[:, 0] * 1e5 + P[:, 1])
        seeds = P[order[np.linspace(0, len(P) - 1, k).astype(int)]].copy()
        for _ in range(4):
            d = ((P[:, None, :] - seeds[None]) ** 2).sum(-1)
            a = d.argmin(1)
            n = np.bincount(a, minlength=k).astype(np.float32)
            sy = np.bincount(a, weights=P[:, 0], minlength=k)
            sx = np.bincount(a, weights=P[:, 1], minlength=k)
            nz = n > 0
            seeds[nz, 0] = sy[nz] / n[nz]
            seeds[nz, 1] = sx[nz] / n[nz]
        ij = P.astype(np.int64)
        out[ij[:, 0], ij[:, 1]] = nxt + a
        nxt += k
    return out


def _assign_whole(D, ar, caps, eps=0.25, iters=1500):
    """One target orientation per fragment (NO subdivision): entropic optimal
    transport (log-domain Sinkhorn) between fragment areas and target
    capacities under the HCP disorientation cost, rounded by per-fragment
    argmax of the transport plan. eps=0.25 deg keeps the plan near the
    unregularised optimum (swept 0.25-3.0; EMD grows monotonically with eps,
    and greedy best-fit-decreasing was 2-5x worse). Residual ODF error is the
    indivisibility floor of about half a fragment per target orientation."""
    C = torch.from_numpy(-D / eps)
    la = torch.log(torch.from_numpy(caps / caps.sum()))
    lb = torch.log(torch.from_numpy(ar / ar.sum()))
    f = torch.zeros(D.shape[0], dtype=torch.float64)
    g = torch.zeros(D.shape[1], dtype=torch.float64)
    for _ in range(iters):
        f = la - torch.logsumexp(C + g[None, :], dim=1)
        g = lb - torch.logsumexp(C + f[:, None], dim=0)
    return (C + f[:, None] + g[None, :]).argmax(0).numpy()


def calib_grains(reconB2_u8, orig_u8, tol=8, min_area=4, target=None, min_px=24,
                 cell_px=None, whole=False, lbl_override=None):
    if target is not None:
        # Predicted-B3 (GT-free): target = (S_centers (K,3), weights (K,)).
        # Keep entries with meaningful predicted mass; areas from weights.
        So_t, w_t = target
        keep = w_t > 1e-5
        So, ao = So_t[keep], w_t[keep].astype(np.float64)
        cell = int(cell_px) if cell_px else 120
    else:
        lbl_o = segment_grains(orig_u8, colour_tol=1)
        uo, So, ao = _grain_median_stereo(orig_u8, lbl_o, min_area)
        cell = int(cell_px) if cell_px else int(np.clip(np.median(ao), 60, 400))
    if lbl_override is not None:
        lbl_r = lbl_override.astype(np.int64)
    else:
        lbl_r = segment_grains(reconB2_u8, colour_tol=tol)
    if min_px > 0:
        # Grain-coherent mode (default): absorb speckle, then break giant
        # smeared regions into equiaxed grain-sized cells.
        lbl_r = _merge_small(lbl_r, min_px)
        if whole:
            # Whole-grain mode: keep the recon's grain size/aspect ratio.
            # Only regions larger than the biggest input grain are broken
            # (at that physical scale) since they cannot be real grains.
            big = int(ao.max()) if target is None else int(
                np.percentile(ao / ao.sum(), 99.9) * lbl_r.size) or lbl_r.size
            lbl_r = _split_large(lbl_r, max(big, cell))
        else:
            lbl_r = _split_large(lbl_r, cell)
    # Recon side: repaint EVERY fragment (min_area=1) so soft VAE boundary
    # halos cannot dilute the calibrated ODF; orig side keeps min_area.
    ur, Sr, ar = _grain_median_stereo(reconB2_u8, lbl_r, 1)
    if len(ur) == 0 or len(So) == 0:
        return reconB2_u8.copy(), 0

    # Pixel-granular capacities: no orientation may exceed its input area
    # share. Fragments larger than the nearest grain's remaining capacity are
    # SPLIT across several input grains (concentric fill from the fragment
    # centroid keeps each part spatially coherent). This bounds every
    # orientation's painted area at cap+1 px, so the area-weighted grain ODF
    # matches the input's at pixel granularity (no single-fragment spikes).
    total_px = int(ar.sum())
    caps = np.round(ao / ao.sum() * total_px).astype(np.int64)
    if target is None:
        caps = np.maximum(caps, 1)   # every real input grain stays representable
    else:
        pos = caps > 0
        So, ao, caps = So[pos], ao[pos], caps[pos]
    D = _diso_matrix(_stereo_to_q_wxyz(So), _stereo_to_q_wxyz(Sr))  # (Go,Gr)
    out = reconB2_u8.copy()
    palette = np.clip((np.clip(So, -1, 1) + 1) / 2 * 255.0, 0, 255).round().astype(np.uint8)
    if whole:
        assign = _assign_whole(D, ar, caps.astype(np.float64))
        lut = np.zeros(int(lbl_r.max()) + 1, dtype=np.int64)
        lut[ur] = assign
        out = palette[lut[lbl_r]]
        return out, len(ur)
    order = np.argsort(-ar)                                   # big fragments first
    H, W = lbl_r.shape
    yy, xx = np.mgrid[0:H, 0:W]
    for j in order:
        mask = lbl_r == ur[j]
        ys, xs = yy[mask], xx[mask]
        # Order pixels by projection on the fragment principal axis so a
        # split paints parallel coherent SLABS (simulable), never rings.
        Y = ys - ys.mean(); X = xs - xs.mean()
        th = 0.5 * np.arctan2(2 * (X * Y).mean(), (X * X).mean() - (Y * Y).mean())
        px_order = np.argsort(X * np.cos(th) + Y * np.sin(th))
        ys, xs = ys[px_order], xs[px_order]
        pos = 0
        while pos < ys.size:
            open_i = np.nonzero(caps > 0)[0]
            if open_i.size == 0:
                caps[:] = 1
                open_i = np.arange(len(So))
            i = open_i[np.argmin(D[open_i, j])]
            rem = ys.size - pos
            take = min(int(caps[i]), rem)
            if min_px > 0:
                if rem - take < min_px:
                    take = rem            # absorb tiny tail into this piece
                elif take < min_px:
                    take = min(min_px, rem)  # never paint a sliver
            out[ys[pos:pos + take], xs[pos:pos + take]] = palette[i]
            caps[i] -= take
            pos += take
    if min_px > 0:
        out = _absorb_specks(out, min(min_px, 9))
    return out, len(ur)


_TEXPRIOR = REPO_ROOT / "eval_outputs" / "texture_prior"


def calibrate_recon_png(recon_png, pred_hist_npz, sample_id=None,
                        codebook_npy=None, out_png=None, min_px=24):
    """MANDATORY z-only calibration of ONE FM-DiT decoder output.

    This is the single-image API for the inference chain that must ALWAYS be
    applied before a decoder PNG is used for simulation:

        z -> FM-DiT recon -> SAM grains -> Sinkhorn whole-grain repaint
          -> calibPW_sam.png -> png_to_dream3d -> DAMASK

    Equivalent to ``--whole --seg sam --pred_hist``: SAM grain map (cached to
    <stem>_samlbl.npy) + predicted-ODF whole-grain Sinkhorn repaint. Needs no
    ground truth. Returns the calibrated PNG path (<stem>_calibPW_sam.png
    next to the recon unless out_png is given). Prefer calling this through
    copilot.simulation.codec.recon_to_dream3d, which chains the dream3d step.
    """
    rp = Path(recon_png)
    recon = np.asarray(Image.open(rp).convert("RGB"))
    hists = np.load(pred_hist_npz)
    if sample_id is None:
        if len(hists.files) != 1:
            raise ValueError(
                f"sample_id required: {pred_hist_npz} has {len(hists.files)} entries")
        sample_id = hists.files[0]
    if sample_id not in hists.files:
        raise KeyError(f"{sample_id!r} not in {pred_hist_npz} ({hists.files})")
    centers = np.load(codebook_npy or _TEXPRIOR / "codebook.npy").astype(np.float64)
    lbl = segment_sam(recon, cache_path=rp.with_name(rp.stem + "_samlbl.npy"))
    out, _ = calib_grains(recon, None, min_px=min_px,
                          target=(centers, hists[sample_id].astype(np.float64)),
                          whole=True, lbl_override=lbl)
    op = Path(out_png) if out_png else rp.with_name(rp.stem + "_calibPW_sam.png")
    Image.fromarray(out).save(op)
    return op


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon_root", required=True)
    ap.add_argument("--recon_name", default="recon_vitfmdit_z1280.png")
    ap.add_argument("--tol", type=int, default=8)
    ap.add_argument("--min_area", type=int, default=4)
    ap.add_argument("--min_px", type=int, default=24,
                    help="merge recon fragments smaller than this and paint "
                         "whole grains (simulable output); 0 = legacy "
                         "pixel-exact splitting (speckled)")
    ap.add_argument("--seg", choices=["cc", "sam"], default="cc",
                    help="recon segmentation backend: cc = colour-tolerance "
                         "connected components (default), sam = SAM AMG via "
                         "copilot.simulation.codec (same grains as "
                         "png_to_dream3d)")
    ap.add_argument("--whole", action="store_true",
                    help="one orientation per recon grain, NO subdivision "
                         "(preserves grain size/aspect ratio; approximate ODF)")
    ap.add_argument("--pred_hist", default=None,
                    help="npz of per-ID predicted histograms (texture_prior.py "
                         "predict) -> GT-free predicted-B3 mode")
    ap.add_argument("--codebook", default="eval_outputs/texture_prior/codebook.npy")
    args = ap.parse_args()

    scorer = Scorer()
    root = Path(args.recon_root)
    recons = sorted(root.glob(f"*/sam/{args.recon_name}"))
    if not recons:
        raise SystemExit(f"no {args.recon_name} under {root}")

    hdr = f"{'sample':10s} {'stage':7s} {'ti_b':>6s} {'mx_b':>6s} {'ti_p':>6s} {'mx_p':>6s} {'emd':>6s}"
    print(hdr)
    for rp in recons:
        eid = rp.parent.parent.name
        orig = np.asarray(Image.open(rp.parent / "original.png").convert("RGB"))
        recon = np.asarray(Image.open(rp).convert("RGB"))
        if recon.shape != orig.shape:
            recon = np.asarray(Image.fromarray(recon).resize(orig.shape[1::-1], Image.NEAREST))

        so = scorer.stats(orig)
        lbl_sam = None
        if args.seg == "sam":
            lbl_sam = segment_sam(recon, cache_path=rp.with_name(rp.stem + "_samlbl.npy"))
        if args.pred_hist is not None:
            hists = np.load(args.pred_hist)
            centers = np.load(args.codebook).astype(np.float64)
            if eid not in hists.files:
                print(f"{eid:10s} SKIP (no predicted hist)"); continue
            bp, n_g = calib_grains(recon, None, tol=args.tol, min_px=args.min_px,
                                   target=(centers, hists[eid].astype(np.float64)),
                                   whole=args.whole, lbl_override=lbl_sam)
            for tag, im in [("raw", recon), ("P", bp)]:
                s = scorer.stats(im)
                print(f"{eid:10s} {tag:7s} "
                      f"{s['ti_b']/so['ti_b']:6.3f} {s['mx_b']/so['mx_b']:6.3f} "
                      f"{s['ti_p']/so['ti_p']:6.3f} {s['mx_p']/so['mx_p']:6.3f} "
                      f"{emd_deg(s['fz'], so['fz']):6.2f}", flush=True)
            sfx = "_calibPW.png" if args.whole else "_calibP.png"
            if args.seg == "sam":
                sfx = sfx.replace(".png", "_sam.png")
            Image.fromarray(bp).save(rp.with_name(rp.stem + sfx))
            continue

        b2 = calib_radial(recon, orig)
        b3, n_g = calib_grains(b2, orig, tol=args.tol, min_area=args.min_area,
                               min_px=args.min_px, whole=args.whole,
                               lbl_override=lbl_sam)

        for tag, im in [("raw", recon), ("B2", b2), ("B3", b3)]:
            s = scorer.stats(im)
            print(f"{eid:10s} {tag:7s} "
                  f"{s['ti_b']/so['ti_b']:6.3f} {s['mx_b']/so['mx_b']:6.3f} "
                  f"{s['ti_p']/so['ti_p']:6.3f} {s['mx_p']/so['mx_p']:6.3f} "
                  f"{emd_deg(s['fz'], so['fz']):6.2f}", flush=True)
        Image.fromarray(b2).save(rp.with_name(rp.stem + "_calibB2.png"))
        sfx = "_calibB3W.png" if args.whole else "_calibB3.png"
        if args.seg == "sam":
            sfx = sfx.replace(".png", "_sam.png")
        Image.fromarray(b3).save(rp.with_name(rp.stem + sfx))
    print("[calib] done")


if __name__ == "__main__":
    main()
