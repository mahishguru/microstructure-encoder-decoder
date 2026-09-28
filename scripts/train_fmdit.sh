#!/usr/bin/env bash
# Train one FM-DiT variant (ViT-H/14 encoder + frozen SD3.5 MMDiT decoder) from
# scratch with the recipe used for the paper models:
#   40 epochs, batch 24 x 4 GPUs x accumulation 6 (effective 576),
#   LoRA rank/alpha 64 on attention + FFN projections from epoch 4,
#   ViT upper blocks unfrozen at epoch 8, CFG dropout 0.1, texture head (lambda 0.5),
#   orientation-distribution loss (config.json: training.fmdit.orient_v2).
#
# usage: scripts/train_fmdit.sh <512|768|1024|1280> [master_port]
# env:   NGPU (default 4), PYTHON (default python), MSED_ROOT (default: repo root),
#        HF_TOKEN (SD3.5-medium is a gated model)
set -euo pipefail
WIDTH=${1:?usage: train_fmdit.sh <512|768|1024|1280> [master_port]}
PORT=${2:-29671}
case "$WIDTH" in 512|768|1024|1280) ;; *) echo "width must be 512, 768, 1024 or 1280"; exit 1;; esac

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export MSED_ROOT=${MSED_ROOT:-$PWD}
PY=${PYTHON:-python}
NGPU=${NGPU:-4}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FMDIT_TARGET_DIM=$WIDTH
export FMDIT_NUM_EPOCHS_OVERRIDE=40
export FMDIT_BATCH_SIZE_OVERRIDE=24
export FMDIT_ACCUM_OVERRIDE=6
export FMDIT_LORA_RANK_OVERRIDE=64
export FMDIT_LORA_ALPHA_OVERRIDE=64
export FMDIT_LORA_FFN=1
export FMDIT_LORA_START_EPOCH=4
export FMDIT_PIXEL_MAX_SAMPLES_OVERRIDE=6
export FMDIT_CFG_DROPOUT=0.1
export FMDIT_TEXHEAD=1
export FMDIT_TEXHEAD_LAMBDA=0.5
export FMDIT_CKPT_DIR=${FMDIT_CKPT_DIR:-$MSED_ROOT/checkpoints/fmdit_${WIDTH}_v4scratch}
mkdir -p "$FMDIT_CKPT_DIR" "$MSED_ROOT/logs"

LOG=$MSED_ROOT/logs/fmdit_${WIDTH}_$(date +%Y%m%d_%H%M%S).log
echo "[train_fmdit] width=$WIDTH gpus=$NGPU ckpt=$FMDIT_CKPT_DIR log=$LOG"
"$PY" -m torch.distributed.run --nproc_per_node="$NGPU" --master_port="$PORT" \
    -m microstructure_ed.fmdit.trainer 2>&1 | tee "$LOG"
