#!/usr/bin/env python3
"""Export slim release checkpoints (trained weights only) for upload to the Hugging Face Hub.

    python scripts/export_release_checkpoints.py \
        --fmdit 512=checkpoints/fmdit_512_v4scratch/checkpoint_epoch_040.pth \
        --fmdit 768=checkpoints/fmdit_768_v4scratch/checkpoint_epoch_040.pth \
        --fmdit 1024=checkpoints/fmdit_1024_v4scratch/checkpoint_epoch_040.pth \
        --fmdit 1280=checkpoints/fmdit_1280_v4scratch/checkpoint_epoch_040.pth \
        --baseline vitdit=checkpoints/vitdit_final/checkpoint_epoch_020.pth \
        --baseline vitsdxl=checkpoints/vitsdxl_final/<file>.pth \
        --baseline vitvqgan=checkpoints/epoch_040.pth \
        --out release/

The output tree matches microstructure_ed.checkpoints.RELEASE_FILES and can be
uploaded as is, e.g. `huggingface-cli upload <org>/<repo> release/ .`.
"""
import argparse
from pathlib import Path

from microstructure_ed.checkpoints import RELEASE_FILES, export_release

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--fmdit", action="append", default=[], metavar="WIDTH=PATH")
ap.add_argument("--baseline", action="append", default=[], metavar="NAME=PATH")
ap.add_argument("--out", type=Path, default=Path("release"))
args = ap.parse_args()

jobs = []
for item in args.fmdit:
    w, p = item.split("=", 1)
    jobs.append(("vitfmdit" if w == "512" else f"vitfmdit_{w}", p, int(w)))
for item in args.baseline:
    name, p = item.split("=", 1)
    jobs.append((name, p, RELEASE_FILES[name][1]))

for variant, src, width in jobs:
    dst = args.out / RELEASE_FILES[variant][0]
    info = export_release(src, dst, target_dim=width,
                          strip_frozen_backbone=variant.startswith("vitfmdit"))
    size = dst.stat().st_size / 2**30
    print(f"{variant:14s} -> {dst} ({size:.2f} GiB, dropped {info['dropped_frozen_decoder_tensors']} frozen tensors)")
