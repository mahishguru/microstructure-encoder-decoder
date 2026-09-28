#!/usr/bin/env python3
"""Download released checkpoints and texture priors from the Hugging Face Hub.

    python scripts/download_weights.py                    # FM-DiT-512 + its texture prior
    python scripts/download_weights.py --all              # every model and texture prior
    python scripts/download_weights.py vitfmdit_1280 vitdit

Checkpoints are linked into checkpoints/<variant>.pth; texture priors are placed
in eval_outputs/texture_prior[_<variant>]/, where the calibration code expects them.
Model repository: https://huggingface.co/mahishguru/microstructure-encoder-decoder
"""
import argparse
import os
from pathlib import Path

from microstructure_ed.checkpoints import (
    RELEASE_FILES, TEXTURE_PRIOR_VARIANTS, download, download_texture_prior,
)

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("variants", nargs="*", default=["vitfmdit"],
                help=f"any of {', '.join(sorted(RELEASE_FILES))} (default: vitfmdit = FM-DiT-512)")
ap.add_argument("--all", action="store_true")
ap.add_argument("--root", type=Path, default=Path(os.environ.get("MSED_ROOT", ".")))
args = ap.parse_args()

variants = sorted(RELEASE_FILES) if args.all else args.variants
unknown = set(variants) - set(RELEASE_FILES)
if unknown:
    ap.error(f"unknown variant(s) {sorted(unknown)}")
(args.root / "checkpoints").mkdir(parents=True, exist_ok=True)
for v in variants:
    src = download(v)
    link = args.root / "checkpoints" / f"{v}.pth"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(src)
    print(f"{v:14s} -> {link}")
    if v in TEXTURE_PRIOR_VARIANTS:
        print(f"{'':14s}    texture prior -> {download_texture_prior(v, args.root)}")
if any(v in TEXTURE_PRIOR_VARIANTS for v in variants):
    print(f"{'shared prior':14s} -> {download_texture_prior(None, args.root)}")
