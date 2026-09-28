#!/usr/bin/env bash
# Train one of the baseline decoders compared against FM-DiT.
#   vitdit   : ViT-H/14 encoder + DiT-XL/2 decoder (facebook/DiT-XL-2-256, sd-vae-ft-mse), z = 512
#   vitsdxl  : ViT-H/14 encoder + frozen SDXL UNet (latent diffusion), z = 512
#   vitvqgan : ViT-VQGAN (paintmind vit-s-vqgan), no continuous bottleneck; reference only
#
# usage: scripts/train_baseline.sh <vitdit|vitsdxl|vitvqgan> [master_port]
# env:   NGPU (default 4), PYTHON (default python), MSED_ROOT (default: repo root)
set -euo pipefail
MODEL=${1:?usage: train_baseline.sh <vitdit|vitsdxl|vitvqgan> [master_port]}
PORT=${2:-29681}
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export MSED_ROOT=${MSED_ROOT:-$PWD}
PY=${PYTHON:-python}
NGPU=${NGPU:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$MSED_ROOT/logs"
LOG=$MSED_ROOT/logs/${MODEL}_$(date +%Y%m%d_%H%M%S).log

case "$MODEL" in
  vitdit|vitsdxl)
    "$PY" -m torch.distributed.run --nproc_per_node="$NGPU" --master_port="$PORT" \
        -m "microstructure_ed.baselines.${MODEL}.trainer" 2>&1 | tee "$LOG" ;;
  vitvqgan)
    # single-GPU trainer; reads 256x256 images from $VQGAN_TRAIN_DIR (default dataset_train)
    "$PY" -m microstructure_ed.baselines.vitvqgan.trainer 2>&1 | tee "$LOG" ;;
  *) echo "unknown baseline '$MODEL'"; exit 1 ;;
esac
