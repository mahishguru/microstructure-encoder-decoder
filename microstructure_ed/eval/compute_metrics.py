"""
Compute SSIM, LPIPS (per-image) and FID (corpus-level) for a reconstructed
test set produced by `run_reconstruction.py`.

Inputs (per variant):
    eval_outputs/originals/<stem>.png             (shared across variants)
    eval_outputs/<variant>/reconstructions/<stem>.png
    eval_outputs/<variant>/per_image_recon.csv    (psnr, mse already filled)

Outputs:
    eval_outputs/<variant>/per_image_metrics.csv
        filename, psnr, mse, ssim, ms_ssim, lpips_alex, lpips_vgg
    eval_outputs/<variant>/summary.json
        variant, num_images, mean/std for each metric, fid

Run:
    python -m microstructure_ed.eval.compute_metrics --variant vitfmdit
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO_ROOT = Path(os.environ.get("MSED_ROOT", os.getcwd())).resolve()
from microstructure_ed.eval import material_metrics as matmet  # noqa: E402

# ── Pair dataset ──────────────────────────────────────────────────────────────

class PairDataset(Dataset):
    """Yields aligned (orig, recon) pairs as float tensors in [0, 1], 3xHxW."""

    def __init__(self, originals_dir: Path, recon_dir: Path, size: int = 300):
        orig_stems = {p.stem for p in originals_dir.glob("*.png")}
        recon_stems = {p.stem for p in recon_dir.glob("*.png")}
        self.stems = sorted(orig_stems & recon_stems)
        self.originals_dir = originals_dir
        self.recon_dir = recon_dir
        # NEAREST keeps orientation RGB codes intact when reconstructions are
        # not yet at 300x300 (e.g. vitvqgan saved at 256x256).
        self.tf = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),  # → [0, 1], 3xHxW
        ])

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        o = self.tf(Image.open(self.originals_dir / f"{stem}.png").convert("RGB"))
        r = self.tf(Image.open(self.recon_dir   / f"{stem}.png").convert("RGB"))
        return o, r, stem


# ── Metrics ───────────────────────────────────────────────────────────────────

def init_metrics(device: torch.device):
    """Build SSIM/LPIPS callables and the FID accumulator."""
    from pytorch_msssim import ssim as _ssim, ms_ssim as _ms_ssim
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    lpips_alex = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    lpips_vgg  = LearnedPerceptualImagePatchSimilarity(net_type="vgg",  normalize=True).to(device)
    lpips_alex.eval(); lpips_vgg.eval()

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    def per_image_metrics(orig: torch.Tensor, recon: torch.Tensor) -> Dict[str, List[float]]:
        """orig, recon: (B, 3, H, W) in [0, 1] on `device`."""
        with torch.no_grad():
            ssim_b   = _ssim   (orig, recon, data_range=1.0, size_average=False)              # (B,)
            ms_ssim_b = _ms_ssim(orig, recon, data_range=1.0, size_average=False)             # (B,)
            # LearnedPerceptualImagePatchSimilarity has no per-sample API; loop.
            la, lv = [], []
            for i in range(orig.shape[0]):
                la.append(float(lpips_alex(orig[i:i+1], recon[i:i+1]).item()))
                lv.append(float(lpips_vgg (orig[i:i+1], recon[i:i+1]).item()))
        return {
            "ssim":       ssim_b.cpu().tolist(),
            "ms_ssim":    ms_ssim_b.cpu().tolist(),
            "lpips_alex": la,
            "lpips_vgg":  lv,
        }

    return per_image_metrics, fid


# ── CSV helpers ───────────────────────────────────────────────────────────────

def load_recon_csv(path: Path) -> Dict[str, Tuple[float, float]]:
    """{stem: (psnr, mse)} keyed by filename stem."""
    out: Dict[str, Tuple[float, float]] = {}
    if not path.exists():
        return out
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            stem = Path(row["filename"]).stem
            out[stem] = (float(row["psnr"]), float(row["mse"]))
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True,
                        help="Variant name; reads eval_outputs/<variant>/reconstructions.")
    parser.add_argument("--output_root", type=str,
                        default=str(REPO_ROOT / "eval_outputs"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap (for smoke tests).")
    parser.add_argument("--no_mat_metrics", action="store_true",
                        help="Skip materials metrics (grain size, AR, orientation EMD).")
    parser.add_argument("--class_means_json", type=str,
                        default=matmet.DEFAULT_CLASS_MEANS,
                        help="Path to class_means.json (per-class mean quaternions).")
    parser.add_argument("--seg_recon", choices=("sam", "clean"), default="sam",
                        help="Segmenter for the reconstructed image.")
    parser.add_argument("--eval_size", type=int, default=300,
                        help="Resolution at which CV metrics are computed (NEAREST resize).")
    parser.add_argument("--resume", action="store_true",
                        help="Append to existing per_image_metrics.csv and skip already-evaluated stems.")
    parser.add_argument("--shard", type=int, default=0,
                        help="Shard index in [0, num_shards). When --num_shards>1 the run "
                             "only processes stems whose hash matches this shard, writes to "
                             "per_image_metrics.shard{S}of{K}.csv, and skips FID/summary.")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of shards. Set >1 to enable parallel sharded mode.")
    args = parser.parse_args()

    if args.num_shards < 1:
        sys.exit("--num_shards must be >= 1")
    if args.num_shards > 1 and not (0 <= args.shard < args.num_shards):
        sys.exit(f"--shard {args.shard} out of range for --num_shards {args.num_shards}")
    SHARDED = args.num_shards > 1

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_root = Path(args.output_root)
    orig_dir = out_root / "originals"
    orig_sam_dir = out_root / "originals_sam_cache"  # shared orig SAM labels
    var_dir = out_root / args.variant
    recon_dir = var_dir / "reconstructions"
    recon_sam_dir = var_dir / "recon_sam_cache"  # per-variant recon SAM labels
    if not recon_dir.exists():
        sys.exit(f"[metrics] missing {recon_dir}")
    if not orig_dir.exists():
        sys.exit(f"[metrics] missing {orig_dir}")

    recon_psnr_mse = load_recon_csv(var_dir / "per_image_recon.csv")

    dataset = PairDataset(orig_dir, recon_dir, size=args.eval_size)
    if args.limit is not None:
        dataset.stems = dataset.stems[: args.limit]
    print(f"[metrics/{args.variant}] {len(dataset)} paired images on {device}", flush=True)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0)

    per_image_metrics, fid = init_metrics(device)

    # Materials metrics setup
    mat_keys = list(matmet.PER_IMAGE_KEYS) if not args.no_mat_metrics else []
    class_means: Dict[str, np.ndarray] = {}
    if not args.no_mat_metrics:
        class_means = matmet.load_class_means(args.class_means_json)
        print(f"[metrics/{args.variant}] materials metrics enabled "
              f"(class_means: {len(class_means)} keys, seg_recon={args.seg_recon})",
              flush=True)

    cv_keys = ["psnr", "mse", "ssim", "ms_ssim", "lpips_alex", "lpips_vgg"]
    if SHARDED:
        out_csv = var_dir / f"per_image_metrics.shard{args.shard}of{args.num_shards}.csv"
        import hashlib
        def _belongs(stem: str) -> bool:
            h = int(hashlib.md5(stem.encode()).hexdigest(), 16)
            return (h % args.num_shards) == args.shard
        _before = len(dataset.stems)
        dataset.stems = [s for s in dataset.stems if _belongs(s)]
        print(f"[metrics/{args.variant}] shard {args.shard}/{args.num_shards}: "
              f"{len(dataset.stems)} of {_before} stems", flush=True)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=args.num_workers > 0)
    else:
        out_csv = var_dir / "per_image_metrics.csv"

    done_stems: set = set()
    if args.resume and out_csv.exists():
        with open(out_csv, "r", newline="") as _f:
            _r = csv.reader(_f)
            try:
                next(_r)  # skip header
            except StopIteration:
                pass
            for _row in _r:
                if not _row:
                    continue
                _fn = _row[0]
                done_stems.add(_fn[:-4] if _fn.endswith(".png") else _fn)
        before = len(dataset.stems)
        dataset.stems = [st for st in dataset.stems if st not in done_stems]
        print(f"[metrics/{args.variant}] resume: {len(done_stems)} already done, "
              f"{len(dataset.stems)} of {before} remaining", flush=True)
        # rebuild loader with the trimmed dataset
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=args.num_workers > 0)
        fp = open(out_csv, "a", newline="")
        writer = csv.writer(fp)
    else:
        fp = open(out_csv, "w", newline="")
        writer = csv.writer(fp)
        writer.writerow(["filename"] + cv_keys + mat_keys)

    sums: Dict[str, float] = {k: 0.0 for k in cv_keys + mat_keys}
    sumsq: Dict[str, float] = dict(sums)
    n_per_key: Dict[str, int] = {k: 0 for k in sums}
    n = 0

    for batch_idx, (orig, recon, stems) in enumerate(loader):
        orig = orig.to(device, non_blocking=True)
        recon = recon.to(device, non_blocking=True)

        m = per_image_metrics(orig, recon)

        for i, stem in enumerate(stems):
            psnr, mse = recon_psnr_mse.get(stem, (float("nan"), float("nan")))
            row: Dict[str, float] = {
                "psnr":       psnr,
                "mse":        mse,
                "ssim":       m["ssim"][i],
                "ms_ssim":    m["ms_ssim"][i],
                "lpips_alex": m["lpips_alex"][i],
                "lpips_vgg":  m["lpips_vgg"][i],
            }

            if not args.no_mat_metrics:
                try:
                    orig_u8 = matmet.load_uint8_at_native(orig_dir / f"{stem}.png")
                    recon_u8 = matmet.load_uint8_at_native(recon_dir / f"{stem}.png")
                    ck = matmet.class_key_for_stem(stem, list(class_means))
                    mq = class_means[ck]
                    if args.seg_recon == "sam":
                        olp = str(orig_sam_dir / f"{stem}.npy")
                        rlp = str(recon_sam_dir / f"{stem}.npy")
                    else:
                        olp = rlp = None
                    mat = matmet.material_metrics_pair(
                        orig_u8, recon_u8, mq, seg_recon=args.seg_recon,
                        orig_label_path=olp, recon_label_path=rlp,
                    )
                except Exception as exc:
                    print(f"[metrics/{args.variant}] WARN mat metrics failed for {stem}: {exc!r}",
                          flush=True)
                    mat = {k: float("nan") for k in mat_keys}
                for k in mat_keys:
                    row[k] = float(mat.get(k, float("nan")))

            csv_cells = [f"{stem}.png"]
            for k in cv_keys + mat_keys:
                v = row.get(k, float("nan"))
                csv_cells.append("" if (isinstance(v, float) and math.isnan(v))
                                  else f"{v:.6f}")
            writer.writerow(csv_cells)

            for k, v in row.items():
                if not (isinstance(v, float) and math.isnan(v)):
                    sums[k]  += v
                    sumsq[k] += v * v
                    n_per_key[k] += 1
            n += 1

        # Flush CSV every batch so progress is visible on disk.
        fp.flush()

        if (batch_idx + 1) % 5 == 0:
            print(f"[metrics/{args.variant}] batch {batch_idx + 1}/{len(loader)} "
                  f"({n} done)", flush=True)

    fp.close()

    if SHARDED:
        print(f"[metrics/{args.variant}] shard {args.shard}/{args.num_shards} done; "
              f"wrote {out_csv}.  (Skipping FID/summary in sharded mode; run "
              f"microstructure_ed.eval.merge_metric_shards to finalize.)", flush=True)
        return

    # ── Recompute means/stds from the FULL on-disk CSV so resumed runs
    #    aggregate over all rows, not just newly processed ones.
    sums = {k: 0.0 for k in cv_keys + mat_keys}
    sumsq = dict(sums)
    n_per_key = {k: 0 for k in sums}
    n_total = 0
    with open(out_csv, "r", newline="") as _f:
        _r = csv.DictReader(_f)
        for _row in _r:
            n_total += 1
            for k in cv_keys + mat_keys:
                v = _row.get(k, "")
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

    means = {
        k: (sums[k] / n_per_key[k] if n_per_key[k] else float("nan"))
        for k in sums
    }
    stds = {
        k: (math.sqrt(max(0.0, sumsq[k] / n_per_key[k] - means[k] ** 2))
            if n_per_key[k] else float("nan"))
        for k in sums
    }

    # ── FID over the FULL paired dataset (independent of resume state).
    print(f"[metrics/{args.variant}] computing FID over full dataset ...", flush=True)
    full_dataset = PairDataset(orig_dir, recon_dir, size=args.eval_size)
    fid_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=args.num_workers > 0)
    n_fid = 0
    for _orig, _recon, _stems in fid_loader:
        _orig = _orig.to(device, non_blocking=True)
        _recon = _recon.to(device, non_blocking=True)
        fid.update(_orig,  real=True)
        fid.update(_recon, real=False)
        n_fid += len(_stems)
    print(f"[metrics/{args.variant}] FID accumulated over {n_fid} pairs", flush=True)
    fid_score = float(fid.compute().item())

    summary = {
        "variant":     args.variant,
        "num_images":  n_total,
        "fid":         fid_score,
        "n_fid_pairs": n_fid,
        "mean":        means,
        "std":         stds,
        "csv":         str(out_csv),
        "originals":   str(orig_dir),
        "reconstructions": str(recon_dir),
    }
    summary_path = var_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2), flush=True)
    print(f"[metrics/{args.variant}] wrote {out_csv}", flush=True)
    print(f"[metrics/{args.variant}] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
