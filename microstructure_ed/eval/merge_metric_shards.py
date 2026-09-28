"""
Merge sharded ``per_image_metrics.shard{S}of{K}.csv`` files into one
``per_image_metrics.csv``, then compute corpus-level FID and write
``summary.json`` -- exactly matching the format produced by
``compute_metrics.py`` in unsharded mode.

Usage
-----
    python -m microstructure_ed.eval.merge_metric_shards --variant orientation_codec
    python -m microstructure_ed.eval.merge_metric_shards --variant orientation_codec --no_fid
"""
from __future__ import annotations

import os
import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(os.environ.get("MSED_ROOT", os.getcwd())).resolve()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--output_root", type=str,
                    default=str(REPO_ROOT / "eval_outputs"))
    ap.add_argument("--no_fid", action="store_true",
                    help="Skip FID computation (CV/materials only).")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--eval_size", type=int, default=300)
    args = ap.parse_args()

    var_dir = Path(args.output_root) / args.variant
    if not var_dir.is_dir():
        sys.exit(f"[merge] missing variant dir {var_dir}")

    shard_csvs = sorted(var_dir.glob("per_image_metrics.shard*of*.csv"))
    if not shard_csvs:
        sys.exit(f"[merge] no shard csvs in {var_dir}")
    print(f"[merge/{args.variant}] found {len(shard_csvs)} shards")

    # Concatenate (header from first shard, data deduped by filename).
    out_csv = var_dir / "per_image_metrics.csv"
    header = None
    seen = set()
    rows = []
    for p in shard_csvs:
        with open(p, newline="") as f:
            rdr = csv.reader(f)
            try:
                hdr = next(rdr)
            except StopIteration:
                continue
            if header is None:
                header = hdr
            elif hdr != header:
                sys.exit(f"[merge] header mismatch in {p}\n  expected {header}\n  got      {hdr}")
            for row in rdr:
                if not row:
                    continue
                fn = row[0]
                if fn in seen:
                    continue
                seen.add(fn)
                rows.append(row)
    rows.sort(key=lambda r: r[0])

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[merge/{args.variant}] wrote {out_csv}  rows={len(rows)}")

    # ── aggregate means / stds (skip-NaN) ────────────────────────────────────
    keys = header[1:]
    sums = {k: 0.0 for k in keys}
    sumsq = dict(sums)
    n_per_key = {k: 0 for k in keys}
    for row in rows:
        for k, v in zip(keys, row[1:]):
            if v == "" or v is None:
                continue
            try:
                fv = float(v)
            except ValueError:
                continue
            if math.isnan(fv):
                continue
            sums[k] += fv
            sumsq[k] += fv * fv
            n_per_key[k] += 1

    means = {k: (sums[k]/n_per_key[k] if n_per_key[k] else float("nan")) for k in keys}
    stds  = {k: (math.sqrt(max(0.0, sumsq[k]/n_per_key[k] - means[k]**2))
                 if n_per_key[k] else float("nan")) for k in keys}

    summary = {
        "variant":    args.variant,
        "num_images": len(rows),
        "n_per_key":  n_per_key,
        "mean":       means,
        "std":        stds,
    }

    # ── FID (optional, full-set) ─────────────────────────────────────────────
    if not args.no_fid:
        print(f"[merge/{args.variant}] computing FID over {len(rows)} pairs ...", flush=True)
        import torch
        from torch.utils.data import DataLoader
        from microstructure_ed.eval.compute_metrics import PairDataset
        from torchmetrics.image.fid import FrechetInceptionDistance

        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        orig_dir = Path(args.output_root) / "originals"
        recon_dir = var_dir / "reconstructions"
        ds = PairDataset(orig_dir, recon_dir, size=args.eval_size)
        # Restrict to the merged stems so FID matches the per-image table.
        merged_stems = {Path(r[0]).stem for r in rows}
        ds.stems = sorted(set(ds.stems) & merged_stems)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=args.num_workers > 0)

        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        n_pairs = 0
        with torch.no_grad():
            for orig, recon, _stems in loader:
                orig = orig.to(device, non_blocking=True)
                recon = recon.to(device, non_blocking=True)
                fid.update(orig, real=True)
                fid.update(recon, real=False)
                n_pairs += orig.shape[0]
        fid_val = float(fid.compute().item())
        summary["fid"] = fid_val
        summary["n_fid_pairs"] = n_pairs
        summary["n_psnr_mse_pairs"] = len(rows)
        print(f"[merge/{args.variant}] FID = {fid_val:.4f} over {n_pairs} pairs")
    else:
        summary["fid"] = None

    summary_path = var_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[merge/{args.variant}] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
