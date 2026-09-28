"""Compute PSNR/MSE for a variant from saved 300x300 originals + reconstructions
and patch summary.json with mean ± std. Also writes per_image_psnr_mse.csv."""
from __future__ import annotations
import argparse, csv, json, math, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from microstructure_ed.eval.compute_metrics import PairDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--output_root", default="eval_outputs")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eval_size", type=int, default=300)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_root = Path(args.output_root)
    var_dir = out_root / args.variant
    orig_dir = out_root / "originals"
    recon_dir = var_dir / "reconstructions"
    if not (orig_dir.exists() and recon_dir.exists()):
        sys.exit(f"[psnr] missing dirs: {orig_dir} / {recon_dir}")

    ds = PairDataset(orig_dir, recon_dir, size=args.eval_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0)
    print(f"[psnr/{args.variant}] {len(ds)} pairs on {device}", flush=True)

    out_csv = var_dir / "per_image_psnr_mse.csv"
    fp = open(out_csv, "w", newline="")
    w = csv.writer(fp); w.writerow(["filename","psnr","mse"])

    sum_p=sum_p2=sum_m=sum_m2=0.0; n=0
    for orig, recon, stems in loader:
        orig = orig.to(device, non_blocking=True)
        recon = recon.to(device, non_blocking=True)
        # PairDataset returns float in [0,1]; per-image MSE then PSNR (peak=1.0).
        mse = ((orig - recon) ** 2).mean(dim=(1,2,3))   # (B,)
        psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
        for stem, m_i, p_i in zip(stems, mse.tolist(), psnr.tolist()):
            w.writerow([f"{stem}.png", f"{p_i:.6f}", f"{m_i:.8f}"])
            sum_p += p_i; sum_p2 += p_i*p_i
            sum_m += m_i; sum_m2 += m_i*m_i
            n += 1
        fp.flush()
    fp.close()

    mean_p = sum_p/n; mean_m = sum_m/n
    std_p = math.sqrt(max(0.0, sum_p2/n - mean_p**2))
    std_m = math.sqrt(max(0.0, sum_m2/n - mean_m**2))
    print(f"[psnr/{args.variant}] PSNR={mean_p:.3f}±{std_p:.3f}  MSE={mean_m:.5f}±{std_m:.5f}  n={n}", flush=True)

    sj_path = var_dir / "summary.json"
    sj = json.loads(sj_path.read_text())
    sj.setdefault("mean", {})["psnr"] = mean_p
    sj["mean"]["mse"] = mean_m
    sj.setdefault("std", {})["psnr"] = std_p
    sj["std"]["mse"] = std_m
    sj["n_psnr_mse_pairs"] = n
    sj_path.write_text(json.dumps(sj, indent=2))
    print(f"[psnr/{args.variant}] patched {sj_path} and wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
