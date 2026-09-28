#!/usr/bin/env bash
# Test-set reconstruction + z-only ODF calibration for ONE FM-DiT variant on ONE GPU.
#   1 encode z for train + test images          (texture prior inputs)
#   2 train the z -> ODF-histogram head          (texture prior)
#   3 predict test-set ODF histograms from z
#   4 reconstruct the test set                   (25 Euler steps, noise 0.9, shift 1.0, CFG 3.0)
#   5 SAM grain segmentation of the reconstructions
#   6 whole-grain repaint to the predicted ODF   ("calibPW" rows)
#
# usage: scripts/eval_testset.sh <vitfmdit|vitfmdit_768|vitfmdit_1024|vitfmdit_1280> <checkpoint.pth> [gpu] [master_port]
# env:   PYTHON, MSED_ROOT (expects dataset_train/ and dataset_test/), SAM_CHECKPOINT, WORKERS (default 16)
set -euo pipefail
VAR=${1:?variant}; CKPT=${2:?checkpoint}; GPU=${3:-0}; PORT=${4:-29700}
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export MSED_ROOT=${MSED_ROOT:-$PWD}
PY=${PYTHON:-python}
OUT=$MSED_ROOT/eval_outputs/testset_v4
PRIOR=$MSED_ROOT/eval_outputs/texture_prior_$VAR

export CUDA_VISIBLE_DEVICES=$GPU
export FMDIT_LORA_RANK_OVERRIDE=64 FMDIT_LORA_ALPHA_OVERRIDE=64 FMDIT_LORA_FFN=1 FMDIT_TEXHEAD=1
export TEXPRIOR_VARIANT=$VAR TEXPRIOR_DIR=$PRIOR

step() { echo "--- [$VAR] $1 $(date) ---"; }
step "1/6 encode z (train+test)"
"$PY" -m microstructure_ed.eval.texprior_variant_z encode --ckpt "$CKPT" --batch 64
step "2/6 train hist head"
"$PY" -m microstructure_ed.eval.texture_prior train --epochs 12
step "3/6 dump predicted test hists"
"$PY" -m microstructure_ed.eval.texprior_variant_z dump
step "4/6 test-set reconstruction"
"$PY" -m torch.distributed.run --nproc_per_node=1 --master_port "$PORT" \
    -m microstructure_ed.eval.run_reconstruction --variant "$VAR" --checkpoint "$CKPT" \
    --output_root "$OUT" --batch_size 24 --num_steps 25 \
    --noise_temp 0.9 --shift 1.0 --guidance 3.0 --skip_existing
step "5/6 SAM segmentation"
"$PY" -m microstructure_ed.eval.batch_calibrate_test --stage sam \
    --recon_dir "$OUT/$VAR/reconstructions" \
    --pred_hist "$PRIOR/pred_hists_test.npz" --codebook "$PRIOR/codebook.npy" \
    --out_dir "$OUT/$VAR/calibPW_sam"
step "6/6 whole-grain predicted-ODF repaint"
"$PY" -m microstructure_ed.eval.batch_calibrate_test --stage repaint --workers "${WORKERS:-16}" \
    --recon_dir "$OUT/$VAR/reconstructions" \
    --pred_hist "$PRIOR/pred_hists_test.npz" --codebook "$PRIOR/codebook.npy" \
    --out_dir "$OUT/$VAR/calibPW_sam"
echo "=== [$VAR] done $(date) ==="
