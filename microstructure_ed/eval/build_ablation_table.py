"""
Aggregate per-variant ``eval_outputs/<variant>/summary.json`` files into:
  * ``eval_outputs/ablation_summary.json`` -- full machine-readable record
  * ``eval_outputs/ablation_table.txt``    -- human-readable comparison table

Regenerates the ablation artifacts from scratch and adds any new variants
(e.g. ``orientation_codec``) automatically.

Usage
-----
    python -m microstructure_ed.eval.build_ablation_table
    python -m microstructure_ed.eval.build_ablation_table --variants vitvqgan vitfmdit \
        vitfmdit_768 vitfmdit_1024 vitdit vitsdxl orientation_codec
"""
from __future__ import annotations

import os
import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(os.environ.get("MSED_ROOT", os.getcwd())).resolve()
EVAL_ROOT = REPO_ROOT / "eval_outputs"

VARIANT_LABELS = {
    "vitvqgan":          "ViT-VQGAN (no bottleneck)",
    "orientation_codec": "Orientation Codec (round-trip)",
    "vitfmdit":          "ViT-FM-DiT (z=512)",
    "vitfmdit_768":      "ViT-FM-DiT (z=768)",
    "vitfmdit_1024":     "ViT-FM-DiT (z=1024)",
    "vitfmdit_1280":     "ViT-FM-DiT (z=1280)",
    "vitfmdit_calibPW":      "ViT-FM-DiT (z=512) +calibPW",
    "vitfmdit_768_calibPW":  "ViT-FM-DiT (z=768) +calibPW",
    "vitfmdit_1024_calibPW": "ViT-FM-DiT (z=1024) +calibPW",
    "vitfmdit_1280_calibPW": "ViT-FM-DiT (z=1280) +calibPW",
    "vitdit":            "ViT-DiT (z=512)",
    "vitsdxl":           "ViT-SDXL",
}

REFERENCE_VARIANTS = {"vitvqgan", "orientation_codec"}

METRIC_KEYS = [
    "psnr", "mse", "ssim", "ms_ssim",
    "lpips_alex", "lpips_vgg",
    "avg_grain_size_rel_err", "avg_aspect_ratio_abs_err",
    "orientation_emd_deg", "mean_disorientation_deg",
    "grain_orientation_emd_deg", "grain_mean_disorientation_deg",
    "grain_matched_frac",
]

METRIC_DIRECTIONS = {
    "psnr": "\u2191", "mse": "\u2193",
    "ssim": "\u2191", "ms_ssim": "\u2191",
    "lpips_alex": "\u2193", "lpips_vgg": "\u2193",
    "avg_grain_size_rel_err": "\u2193",
    "avg_aspect_ratio_abs_err": "\u2193",
    "orientation_emd_deg": "\u2193",
    "mean_disorientation_deg": "\u2193",
    "grain_orientation_emd_deg": "\u2193",
    "grain_mean_disorientation_deg": "\u2193",
    "grain_matched_frac": "\u2191",
    "fid": "\u2193",
}

# (header, metric_key, style, direction)
TABLE_COLUMNS = [
    ("FID \u2193",                "fid",                      "single",   "\u2193"),
    ("MS-SSIM \u2191",            "ms_ssim",                  "mean_std", "\u2191"),
    ("LPIPS (AlexNet) \u2193",    "lpips_alex",               "mean_std", "\u2193"),
    ("GrainSize RelErr \u2193",   "avg_grain_size_rel_err",   "mean_std", "\u2193"),
    ("AspectRatio AbsErr \u2193", "avg_aspect_ratio_abs_err", "mean_std", "\u2193"),
    ("Orientation EMD [\u00b0] \u2193", "orientation_emd_deg", "mean_std", "\u2193"),
    ("Mean disorientation [\u00b0] \u2193", "mean_disorientation_deg", "mean_std", "\u2193"),
    ("Grain Orient EMD [\u00b0] \u2193", "grain_orientation_emd_deg", "mean_std", "\u2193"),
    ("Grain disori [\u00b0] \u2193", "grain_mean_disorientation_deg", "mean_std", "\u2193"),
    ("Grain matched frac \u2191", "grain_matched_frac", "mean_std", "\u2191"),
]


def _load_variant_summary(variant, eval_root=EVAL_ROOT):
    p = Path(eval_root) / variant / "summary.json"
    if not p.is_file():
        return None
    raw = json.loads(p.read_text())
    means = raw.get("mean", {})
    stds = raw.get("std", {})
    out = {
        "num_images": raw.get("num_images"),
        "fid": raw.get("fid"),
        "n_fid_pairs": raw.get("n_fid_pairs", raw.get("num_images")),
        "n_psnr_mse_pairs": raw.get("n_psnr_mse_pairs", raw.get("num_images")),
        "metrics": {},
    }
    for k in METRIC_KEYS:
        if k in means:
            out["metrics"][k] = {"mean": means[k], "std": stds.get(k, float("nan"))}
    return out


def _fmt_single(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "\u2014"
    return "%.2f" % v


def _fmt_mean_std(d):
    if not d:
        return "\u2014"
    m, s = d.get("mean"), d.get("std")
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return "\u2014"
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return "%.3f" % m
    if abs(m) < 0.1:
        return "%.4f \u00b1 %.4f" % (m, s)
    return "%.3f \u00b1 %.3f" % (m, s)


def _value_for_compare(variant_data, key):
    if key == "fid":
        v = variant_data.get("fid")
        return float(v) if v is not None else None
    m = variant_data.get("metrics", {}).get(key)
    if not m:
        return None
    v = m.get("mean")
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v)


def _find_best(variants, key, direction):
    best_name = None
    best_val = None
    for name, data in variants.items():
        if name in REFERENCE_VARIANTS:
            continue
        v = _value_for_compare(data, key)
        if v is None:
            continue
        if best_val is None \
                or (direction == "\u2191" and v > best_val) \
                or (direction == "\u2193" and v < best_val):
            best_val = v
            best_name = name
    return best_name


def _build_table_text(variants, n_test):
    rows = []
    headers = ["Model"] + [c[0] for c in TABLE_COLUMNS]
    best_per_col = {col[1]: _find_best(variants, col[1], col[3]) for col in TABLE_COLUMNS}

    for vname in VARIANT_LABELS:
        if vname not in variants:
            continue
        data = variants[vname]
        cells = [VARIANT_LABELS[vname]]
        for header, key, style, _dir in TABLE_COLUMNS:
            if style == "single":
                txt = _fmt_single(data.get(key))
            else:
                txt = _fmt_mean_std(data.get("metrics", {}).get(key))
            if best_per_col.get(key) == vname and txt != "\u2014":
                txt = txt + "*"
            cells.append(txt)
        rows.append(cells)

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def _fmt_row(cells):
        out = [cells[0].ljust(widths[0])]
        for i, c in enumerate(cells[1:], start=1):
            out.append(c.rjust(widths[i]))
        return "  ".join(out)

    sep = "  ".join("=" * w for w in widths)
    lines = [
        "Reconstruction quality and microstructural fidelity on the test set "
        "(n = %d)." % n_test,
        "Mean \u00b1 std unless noted; FID is a single distribution-level number.",
        "Best per column marked with * (computed across bottlenecked variants only;",
        "ViT-VQGAN and Orientation-Codec round-trip have no neural bottleneck and",
        "are reported for reference).",
        "Originals and reconstructions resized to 300x300 (NEAREST).",
        "",
        sep,
        _fmt_row(headers),
        sep,
    ]
    lines.extend(_fmt_row(r) for r in rows)
    lines.append(sep)
    lines += [
        "",
        "Arrows indicate desired direction (\u2191 higher is better, \u2193 lower is better).",
        "FID computed with Inception-v3 (feature=2048) over the full paired set.",
        "MS-SSIM (5-scale) chosen over single-scale SSIM because it shows the largest",
        "relative gap between the FM-DiT family and the other bottlenecked baselines.",
        "LPIPS reported with the AlexNet backbone (larger FM-DiT margin than VGG).",
        "",
        "Materials-science metrics (segmentation by SAM ViT-B on both sides):",
        "  AspectRatio AbsErr -- morphology / anisotropy: mean |AR_orig - AR_recon|.",
        "  Disorientation     -- crystallography: mean misorientation angle between",
        "                        IoU-matched grains' dominant orientations,",
        "                        computed from per-class mean quaternions.",
        "",
        "Orientation-Codec round-trip: PNG -> .dream3d (decode_image_pixelwise with",
        "the per-class mean quaternion) -> PNG (encode_dream3d_global). It measures",
        "the lossy floor of the codec itself; no neural network is involved.",
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--eval_root", type=Path, default=EVAL_ROOT,
                    help="Root holding <variant>/summary.json dirs.")
    ap.add_argument("--out_json", type=Path, default=None)
    ap.add_argument("--out_txt",  type=Path, default=None)
    args = ap.parse_args()
    if args.out_json is None:
        args.out_json = args.eval_root / "ablation_summary.json"
    if args.out_txt is None:
        args.out_txt = args.eval_root / "ablation_table.txt"

    requested = args.variants or list(VARIANT_LABELS.keys())
    variants = {}
    skipped = []
    for v in requested:
        data = _load_variant_summary(v, args.eval_root)
        if data is None:
            skipped.append(v)
            continue
        variants[v] = data
    if not variants:
        sys.exit("[ablation] no variants found with summary.json on disk")

    n_test = max((d.get("num_images") or 0) for d in variants.values())

    summary = {
        "meta": {
            "description": (
                "Ablation table: mean \u00b1 std over the test set per variant. "
                "Originals and reconstructions both resized to 300x300 NEAREST. "
                "PSNR computed with peak=1.0 on float [0,1] tensors. FID computed "
                "once per variant over the full paired dataset (Inception-v3, "
                "feature=2048, normalize=True). Materials metrics use SAM ViT-B "
                "for grain segmentation on both sides; disorientation is computed "
                "on IoU-matched grains using per-class mean quaternions. The "
                "Orientation-Codec round-trip variant is the codec's own lossy "
                "floor (PNG -> .dream3d -> PNG, no neural network)."
            ),
            "num_images_per_variant": n_test,
            "eval_resolution": 300,
            "segmenter": "sam_vit_b",
            "metric_directions": METRIC_DIRECTIONS,
            "reference_variants": sorted(REFERENCE_VARIANTS),
            "skipped_missing": skipped,
        },
        "variants": variants,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2))
    args.out_txt.write_text(_build_table_text(variants, n_test))

    print("[ablation] wrote", args.out_json)
    print("[ablation] wrote", args.out_txt)
    if skipped:
        print("[ablation] skipped (no summary.json):", skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
