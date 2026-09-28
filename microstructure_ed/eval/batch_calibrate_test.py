#!/usr/bin/env python
"""Batch z-only calibPred (whole-grain, SAM) for a full recon directory.

The MANDATORY inference chain (README 13.7) at test-set scale:
  recon PNG -> SAM grains -> Sinkhorn whole-grain repaint onto the predicted
  ODF -> <out_dir>/<stem>.png (flat-coloured, png_to_dream3d-ready).

Two stages so GPU (SAM) and CPU (Sinkhorn) parts parallelise independently:
  --stage sam     : GPU. Segment every recon, cache label maps to
                    <recon_dir>/../sam_cache/<stem>.npy (~1.3 s/img).
  --stage repaint : CPU pool (--workers). Needs the caches. Resumable.
"""
from __future__ import annotations
import argparse
import os
import sys
import time

# Cap BLAS/OpenMP threads BEFORE numpy/torch load: with 12 concurrent procs on
# one node, default (=ncores) thread pools thrash the machine (load >600).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")
from pathlib import Path

import numpy as np
from PIL import Image


_G = {}


def _repaint_one(job):
    stem, recon_path, lbl_path, out_path = job
    import torch
    torch.set_num_threads(2)
    from microstructure_ed.eval.odf_calibrate import calib_grains  # deferred for fork workers
    recon = np.asarray(Image.open(recon_path).convert("RGB"))
    lbl = np.load(lbl_path)
    hist = _G["hists"][stem].astype(np.float64)
    out, _ = calib_grains(recon, None, target=(_G["centers"], hist),
                          whole=True, lbl_override=lbl)
    Image.fromarray(out).save(out_path)
    return stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon_dir", required=True)
    ap.add_argument("--pred_hist", required=True)
    ap.add_argument("--codebook", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--stage", choices=["sam", "repaint"], required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()

    rdir = Path(args.recon_dir)
    cache = rdir.parent / "sam_cache"
    cache.mkdir(parents=True, exist_ok=True)
    recons = sorted(rdir.glob("*.png"))
    if args.num_shards > 1:
        recons = recons[args.shard_idx::args.num_shards]
    print(f"[calib/{args.stage}] {len(recons)} recons in {rdir} "
          f"(shard {args.shard_idx}/{args.num_shards})", flush=True)

    if args.stage == "sam":
        import torch
        torch.set_num_threads(4)
        from microstructure_ed.eval.odf_calibrate import segment_sam
        t0 = time.time()
        done = 0
        for i, rp in enumerate(recons):
            cp = cache / f"{rp.stem}.npy"
            if cp.exists():
                continue
            img = np.asarray(Image.open(rp).convert("RGB"))
            segment_sam(img, cache_path=cp)
            done += 1
            if done % 200 == 0:
                el = time.time() - t0
                print(f"[sam] {i+1}/{len(recons)} ({el:.0f}s, "
                      f"{el/max(done,1):.2f}s/img)", flush=True)
        print(f"[sam] done ({done} new)", flush=True)
        return

    # repaint stage (CPU pool)
    from multiprocessing import Pool
    odir = Path(args.out_dir)
    odir.mkdir(parents=True, exist_ok=True)
    hists = np.load(args.pred_hist)
    # materialise: NpzFile shares one zip handle across forked workers (unsafe)
    _G["hists"] = {k: hists[k] for k in hists.files}
    _G["centers"] = np.load(args.codebook).astype(np.float64)
    jobs = []
    for rp in recons:
        op = odir / rp.name
        cp = cache / f"{rp.stem}.npy"
        if op.exists():
            continue
        if rp.stem not in _G["hists"]:
            print(f"[repaint] SKIP {rp.stem}: no predicted hist", flush=True)
            continue
        if not cp.exists():
            print(f"[repaint] SKIP {rp.stem}: no SAM cache", flush=True)
            continue
        jobs.append((rp.stem, str(rp), str(cp), str(op)))
    print(f"[repaint] {len(jobs)} to do, {args.workers} workers", flush=True)
    t0 = time.time()
    with Pool(args.workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(_repaint_one, jobs, chunksize=8)):
            if (k + 1) % 500 == 0:
                el = time.time() - t0
                print(f"[repaint] {k+1}/{len(jobs)} ({el:.0f}s, "
                      f"{el/(k+1):.2f}s/img)", flush=True)
    print("[repaint] done", flush=True)


if __name__ == "__main__":
    main()
