#!/usr/bin/env python
"""Per-variant texture-prior refresh WITHOUT redoing codebook/hists.

The K-means codebook and per-image pixel histograms depend only on the
IMAGES, not on any encoder, so they are shared across variants. Only z
(encoder output) and the head must be rebuilt per variant.

  encode : copy codebook/hist/names from the shared prior dir, then encode
           z_train / z_test with THIS variant's encoder (env TEXPRIOR_VARIANT,
           TEXPRIOR_DIR; --ckpt = variant best checkpoint).
  dump   : head.pt + z_test.npy -> pred_hists_test.npz keyed by image stem
           (the z-only predicted ODFs used by calibPW).
"""
from __future__ import annotations
import os
import argparse
import shutil
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(os.environ.get("MSED_ROOT", os.getcwd())).resolve()

from microstructure_ed.eval.texture_prior import (  # noqa: E402  (env-driven VARIANT)
    PRIOR_DIR, TF, build_encoder, HistHead,
)

SHARED = REPO_ROOT / "eval_outputs" / "texture_prior"
SHARED_FILES = ("codebook.npy", "hist_train.npy", "hist_test.npy",
                "names_train.npy", "names_test.npy")


def cmd_encode(args):
    device = torch.device(args.device)
    PRIOR_DIR.mkdir(parents=True, exist_ok=True)
    for f in SHARED_FILES:
        if not (PRIOR_DIR / f).exists():
            shutil.copy(SHARED / f, PRIOR_DIR / f)
    enc = build_encoder(device, args.ckpt)
    for tag, ddir in (("train", "dataset_train"), ("test", "dataset_test")):
        names = np.load(PRIOR_DIR / f"names_{tag}.npy")
        files = [REPO_ROOT / ddir / f"{n}.png" for n in names]
        N = len(files)
        zs = None
        pool = ThreadPoolExecutor(8)
        B = args.batch
        t0 = time.time()
        for s in range(0, N, B):
            xs = torch.stack(list(pool.map(
                lambda f: TF(Image.open(f).convert("RGB")), files[s:s + B]))).to(device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                z = enc(xs)
            if zs is None:
                zs = np.zeros((N, z.shape[1]), dtype=np.float32)
            zs[s:s + B] = z.float().cpu().numpy()
            if s % (B * 80) == 0:
                print(f"[encode] {tag} {s}/{N} ({time.time()-t0:.0f}s)", flush=True)
        np.save(PRIOR_DIR / f"z_{tag}.npy", zs)
        print(f"[encode] {tag} done -> z_{tag}.npy {zs.shape}", flush=True)


def cmd_dump(args):
    device = torch.device(args.device)
    blob = torch.load(PRIOR_DIR / "head.pt", map_location=device)
    head = HistHead(blob["in_dim"], blob["K"]).to(device)
    head.load_state_dict(blob["state"]); head.eval()
    z = torch.from_numpy(np.load(PRIOR_DIR / "z_test.npy")).to(device)
    names = np.load(PRIOR_DIR / "names_test.npy")
    out = {}
    with torch.no_grad():
        for s in range(0, len(z), 2048):
            h = torch.softmax(head(z[s:s + 2048].float()), -1).cpu().numpy()
            for j, n in enumerate(names[s:s + 2048]):
                out[str(n)] = h[j].astype(np.float32)
    np.savez(PRIOR_DIR / "pred_hists_test.npz", **out)
    print(f"[dump] wrote {PRIOR_DIR/'pred_hists_test.npz'} ({len(out)} entries)",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--device", default="cuda:0")
    e.add_argument("--batch", type=int, default=48)
    d = sub.add_parser("dump")
    d.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    {"encode": cmd_encode, "dump": cmd_dump}[args.cmd](args)


if __name__ == "__main__":
    main()
