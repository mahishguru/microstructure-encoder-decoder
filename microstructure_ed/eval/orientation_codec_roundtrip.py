"""
Orientation-codec round-trip: turn each original PNG into a .dream3d file
(via pixel-wise inverse stereographic projection using the per-class mean
quaternion) and then re-encode the .dream3d back to an 8-bit PNG using the
same global-mean pipeline that produced the originals. Output PNGs land in
``eval_outputs/orientation_codec/reconstructions/`` so the existing
``compute_metrics.py`` can score them as a regular variant.

The round trip measures the lossy floor of the orientation codec itself:
no neural network is involved.

Pipeline per image
------------------
    PNG  --decode_image_pixelwise-->  .dream3d
    .dream3d  --encode_dream3d_global-->  PNG (8-bit)

Usage
-----
    python -m microstructure_ed.eval.orientation_codec_roundtrip \
        --originals  eval_outputs/originals \
        --output_dir eval_outputs/orientation_codec/reconstructions \
        --class_means microstructure_ed/assets/class_means.json \
        --workers 8
"""
from __future__ import annotations

import os
import argparse
import json
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional


def class_key_for_stem(stem: str, available: list) -> str:
    # Mirror-augmented training images carry a "mirror_" prefix; strip it
    # before matching so the underlying alloy-class key is recovered.
    probe = stem[len("mirror_"):] if stem.startswith("mirror_") else stem
    for key in sorted(available, key=len, reverse=True):
        if probe.startswith(key):
            return key
    raise KeyError("no class_means key matches stem=" + repr(stem))


def _roundtrip_one(args_tuple):
    (png_path_str, out_png_str, class_means_json, dream3d_workdir,
     keep_dream3d, target_size) = args_tuple
    png_path = Path(png_path_str)
    out_png = Path(out_png_str)
    stem = png_path.stem
    try:
        from orientation_codec.dataset import (
            decode_image_pixelwise, encode_dream3d_global, load_class_means,
        )
        from PIL import Image

        means = load_class_means(class_means_json)
        ckey = class_key_for_stem(stem, list(means))
        mean_q = means[ckey]

        if dream3d_workdir is None:
            scratch = Path(tempfile.mkdtemp(prefix="oc_rt_"))
            cleanup_scratch = True
        else:
            scratch = Path(dream3d_workdir) / stem
            scratch.mkdir(parents=True, exist_ok=True)
            cleanup_scratch = False

        d3d_path = scratch / (stem + ".dream3d")

        # Step 1: PNG -> .dream3d (pixel-wise; segmentation recovers labels).
        decode_image_pixelwise(
            image_path=png_path, mean_q=mean_q, output_path=d3d_path,
        )

        # Step 2: .dream3d -> 8-bit PNG using the SAME class mean.
        produced = encode_dream3d_global(
            dream3d_path=d3d_path, global_mean_q=mean_q,
            output_dir=out_png.parent, name=stem, fmt="png",
        )
        produced = Path(produced)

        if target_size is not None:
            with Image.open(produced) as im:
                if im.size != tuple(target_size):
                    im = im.convert("RGB").resize(tuple(target_size), Image.NEAREST)
                    im.save(produced)

        if not keep_dream3d and cleanup_scratch:
            for p in scratch.glob("*"):
                try: p.unlink()
                except OSError: pass
            try: scratch.rmdir()
            except OSError: pass

        return stem, "ok", None
    except Exception:
        return stem, "fail", traceback.format_exc()


def main() -> int:
    repo_root = Path(os.environ.get("MSED_ROOT", os.getcwd())).resolve()
    ap = argparse.ArgumentParser()
    ap.add_argument("--originals", type=Path,
                    default=repo_root / "eval_outputs" / "originals")
    ap.add_argument("--output_dir", type=Path,
                    default=repo_root / "eval_outputs" / "orientation_codec" / "reconstructions")
    ap.add_argument("--class_means", type=Path,
                    default=Path(__file__).resolve().parents[1] / "assets" / "class_means.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-skip", action="store_true")
    ap.add_argument("--keep_dream3d", action="store_true")
    ap.add_argument("--dream3d_workdir", type=Path, default=None)
    ap.add_argument("--target_size", type=int, default=300)
    args = ap.parse_args()

    if not args.originals.is_dir():
        sys.exit("[oc_rt] missing originals dir: " + str(args.originals))
    if not args.class_means.is_file():
        sys.exit("[oc_rt] missing class_means: " + str(args.class_means))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pngs = sorted(args.originals.glob("*.png"))
    if args.limit is not None:
        pngs = pngs[:args.limit]
    target_size = (args.target_size, args.target_size)

    work = []
    skipped = 0
    for p in pngs:
        out_png = args.output_dir / p.name
        if out_png.exists() and not args.no_skip:
            skipped += 1
            continue
        work.append((str(p), str(out_png), str(args.class_means),
                     str(args.dream3d_workdir) if args.dream3d_workdir else None,
                     args.keep_dream3d, target_size))

    print("[oc_rt] originals:", len(pngs), " skipped(existing):", skipped,
          " to_process:", len(work), " workers:", args.workers, flush=True)
    if not work:
        print("[oc_rt] nothing to do.")
        return 0

    n_ok = 0; n_fail = 0; failures = []; t0 = time.time()
    if args.workers <= 1:
        for w in work:
            stem, status, err = _roundtrip_one(w)
            if status == "ok": n_ok += 1
            else: n_fail += 1; failures.append((stem, err or ""))
            if (n_ok + n_fail) % 50 == 0:
                rate = (n_ok + n_fail) / max(1e-6, time.time() - t0)
                print("[oc_rt]", n_ok + n_fail, "/", len(work),
                      " ok=", n_ok, " fail=", n_fail,
                      " (", round(rate, 1), "/s)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_roundtrip_one, w): w[0] for w in work}
            for fut in as_completed(futs):
                stem, status, err = fut.result()
                if status == "ok": n_ok += 1
                else: n_fail += 1; failures.append((stem, err or ""))
                if (n_ok + n_fail) % 50 == 0:
                    rate = (n_ok + n_fail) / max(1e-6, time.time() - t0)
                    print("[oc_rt]", n_ok + n_fail, "/", len(work),
                          " ok=", n_ok, " fail=", n_fail,
                          " (", round(rate, 1), "/s)", flush=True)

    elapsed = time.time() - t0
    print("[oc_rt] done. ok=", n_ok, " fail=", n_fail,
          " elapsed=", round(elapsed, 1), "s (", round(elapsed/60, 1), " min)", flush=True)

    log_path = args.output_dir.parent / "roundtrip_log.json"
    log_path.write_text(json.dumps({
        "n_originals": len(pngs), "n_skipped_existing": skipped,
        "n_processed": len(work), "n_ok": n_ok, "n_fail": n_fail,
        "elapsed_s": elapsed, "workers": args.workers,
        "class_means": str(args.class_means),
        "originals": str(args.originals), "output_dir": str(args.output_dir),
        "failures": [{"stem": s, "error": e.splitlines()[-1] if e else ""}
                     for s, e in failures[:50]],
    }, indent=2))
    print("[oc_rt] log:", log_path, flush=True)
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
