#!/usr/bin/env python3
"""Split orientation-codec PNGs into flat train/test folders (seeded 90/10 split).

The source tree holds one sub-folder per alloy class with images named
``<class>_orientation_<id>.png`` (the output of ``orientation-codec batch-global``).
Output folders are flat; the class is recovered from the filename prefix.

    python scripts/split_dataset.py /path/to/rve --out . --test-split 0.10 --seed 42
"""
import argparse
import random
import shutil
from pathlib import Path

from tqdm import tqdm

VALID_EXTS = {".png", ".jpg", ".jpeg"}

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("source", type=Path, help="root folder with one sub-folder per alloy class")
ap.add_argument("--out", type=Path, default=Path("."), help="creates <out>/dataset_train and <out>/dataset_test")
ap.add_argument("--test-split", type=float, default=0.10)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--overwrite", action="store_true", help="delete existing dataset_train/dataset_test first")
args = ap.parse_args()

random.seed(args.seed)
files = sorted(f for f in args.source.rglob("*") if f.suffix.lower() in VALID_EXTS)
random.shuffle(files)
n_test = int(len(files) * args.test_split)
splits = {"dataset_train": files[n_test:], "dataset_test": files[:n_test]}
print(f"{len(files)} images -> train {len(splits['dataset_train'])} | test {n_test}")

for name, subset in splits.items():
    d = args.out / name
    if d.exists() and any(d.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{d} is not empty; pass --overwrite to replace it")
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    for f in tqdm(subset, desc=name):
        shutil.copy2(f, d / f.name)
