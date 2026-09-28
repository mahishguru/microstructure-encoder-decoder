"""Precompute SAM label maps for all ORIGINAL images into a shared on-disk
cache (``eval_outputs/originals_sam_cache/<stem>.npy``).

Each original is segmented identically for every model variant, so doing it
once here removes the ~3x redundant SAM cost during the per-variant metrics
phase (SAM is the dominant ~1.3 s/segmentation bottleneck). Sharded for
parallelism: a stem belongs to this shard iff md5(stem) % num_shards == shard.
Already-cached stems are skipped, so the job is restartable.
"""
from __future__ import annotations

import os
import argparse
import hashlib
import sys
import time
from pathlib import Path

REPO_ROOT = Path(os.environ.get("MSED_ROOT", os.getcwd())).resolve()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", default=str(REPO_ROOT / "eval_outputs"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()

    import microstructure_ed.eval.material_metrics as mm

    out = Path(args.output_root)
    orig_dir = out / "originals"
    cache_dir = out / "originals_sam_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted(p.stem for p in orig_dir.glob("*.png"))
    if args.num_shards > 1:
        stems = [s for s in stems
                 if int(hashlib.md5(s.encode()).hexdigest(), 16) % args.num_shards == args.shard]
    print(f"[precompute s{args.shard}/{args.num_shards}] {len(stems)} originals",
          flush=True)

    t0 = time.time()
    done = 0
    skipped = 0
    for i, s in enumerate(stems):
        cp = cache_dir / f"{s}.npy"
        if cp.exists():
            skipped += 1
            continue
        img = mm.load_uint8_at_native(orig_dir / f"{s}.png")
        mm.segment_sam_cached(img, str(cp))
        done += 1
        if done % 50 == 0:
            el = time.time() - t0
            print(f"[precompute s{args.shard}] {i + 1}/{len(stems)} "
                  f"done={done} skip={skipped} {el / max(done, 1):.2f}s/seg",
                  flush=True)
    print(f"[precompute s{args.shard}] COMPLETE done={done} skipped={skipped} "
          f"in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
