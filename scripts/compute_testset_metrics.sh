#!/usr/bin/env bash
# Full-test-set (n=10,100) metrics for the 4 v4 variants, RAW + calibPW rows.
# Layout root: eval_outputs/testset_v4 (originals/, <variant>/reconstructions,
# <variant>/recon_sam_cache -> sam_cache from the calibration pipeline,
# <variant>_calibPW/{reconstructions,recon_sam_cache} symlinks).
# Recon SAM labels are REUSED from the calibration pipeline (identical
# segmenter: microstructure_ed.segmentation._sam_segment, vit_b pps=32 iou=0.86),
# so only the originals need a SAM pass here. Run scripts/eval_testset.sh for
# every variant first.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_ROOT=${MSED_ROOT:-$PWD}
export MSED_ROOT=$REPO_ROOT
PY="${PYTHON:-python}"
OUT="$REPO_ROOT/eval_outputs/testset_v4"
LOGD="$OUT/logs/shards"; mkdir -p "$LOGD"
RAW_VARIANTS=(vitfmdit vitfmdit_768 vitfmdit_1024 vitfmdit_1280)
ALL_VARIANTS=(vitfmdit vitfmdit_768 vitfmdit_1024 vitfmdit_1280 \
              vitfmdit_calibPW vitfmdit_768_calibPW vitfmdit_1024_calibPW vitfmdit_1280_calibPW)
NUM_SHARDS="${NUM_SHARDS:-10}"      # per variant -> 8*10 = 80 procs
BATCH="${BATCH:-16}"
NGPU="${NGPU:-4}"
THREADS="${THREADS:-2}"
export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
       MKL_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS" \
       VECLIB_MAXIMUM_THREADS="$THREADS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "############################################################"
echo "# TESTSET_V4 METRICS START $(date)"
echo "# root=$OUT shards=$NUM_SHARDS variants=${ALL_VARIANTS[*]}"
echo "############################################################"

# ---------- PHASE 1: precompute ORIGINAL SAM labels (shared cache) ----------
PRE_SHARDS="${PRE_SHARDS:-$(( NGPU * 6 ))}"
if [[ "${SKIP_PRECOMPUTE:-0}" != "1" ]]; then
  echo "===== PHASE 1: originals SAM precompute (${PRE_SHARDS} procs) $(date) ====="
  ppids=()
  for ((s=0; s<PRE_SHARDS; s++)); do
    gpu=$(( s % NGPU ))
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u -m microstructure_ed.eval.precompute_orig_sam \
        --output_root "$OUT" --num_shards "$PRE_SHARDS" --shard "$s" \
        > "$LOGD/precompute_shard${s}of${PRE_SHARDS}.out" 2>&1 &
    ppids+=($!); sleep 0.3
  done
  pfail=0
  for p in "${ppids[@]}"; do if ! wait "$p"; then pfail=1; fi; done
  ncached=$(ls "$OUT/originals_sam_cache"/*.npy 2>/dev/null | wc -l)
  echo "===== PHASE 1 done $(date) fail=$pfail cached=$ncached ====="
else
  echo "===== PHASE 1 SKIPPED ====="
fi

# ---------- PHASE 2a: sharded per-image metrics (all variants concurrent) ----
echo "===== PHASE 2a: sharded metrics $(date) ====="
pids=(); tags=(); idx=0
for V in "${ALL_VARIANTS[@]}"; do
  for ((s=0; s<NUM_SHARDS; s++)); do
    gpu=$(( idx % NGPU ))
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u -m microstructure_ed.eval.compute_metrics \
        --variant "$V" --output_root "$OUT" \
        --batch_size "$BATCH" --num_workers 0 --device cuda:0 \
        --num_shards "$NUM_SHARDS" --shard "$s" \
        > "$LOGD/${V}_shard${s}of${NUM_SHARDS}.out" 2>&1 &
    pids+=($!); tags+=("${V}#${s}"); idx=$((idx+1))
    sleep 0.3
  done
done
echo "[launch] ${#pids[@]} shard procs $(date)"
fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then echo "[shard] ${tags[$i]} FAILED"; fail=1; fi
done
echo "===== PHASE 2a done $(date) shard_fail=$fail ====="

# ---------- PHASE 2b: merge shards + FID + summary (2 per GPU) --------------
echo "===== PHASE 2b: merge + FID $(date) ====="
mpids=(); mtags=(); gpu=0
for V in "${ALL_VARIANTS[@]}"; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u -m microstructure_ed.eval.merge_metric_shards \
      --variant "$V" --output_root "$OUT" \
      --batch_size 64 --num_workers 4 --device cuda:0 \
      > "$OUT/logs/${V}_merge.out" 2>&1 &
  mpids+=($!); mtags+=("$V"); gpu=$(( (gpu+1) % NGPU ))
done
mfail=0
for i in "${!mpids[@]}"; do
  if wait "${mpids[$i]}"; then echo "[merge] ${mtags[$i]} OK"; else echo "[merge] ${mtags[$i]} FAILED"; mfail=1; fi
done
echo "===== PHASE 2b done $(date) merge_fail=$mfail ====="

# ---------- PHASE 2c: backfill PSNR/MSE for calibPW rows ---------------------
echo "===== PHASE 2c: PSNR/MSE backfill (calibPW) $(date) ====="
bpids=(); gpu=0
for V in "${RAW_VARIANTS[@]}"; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u -m microstructure_ed.eval.backfill_psnr_mse \
      --variant "${V}_calibPW" --output_root "$OUT" \
      > "$OUT/logs/${V}_calibPW_psnr.out" 2>&1 &
  bpids+=($!); gpu=$(( (gpu+1) % NGPU ))
done
bfail=0
for p in "${bpids[@]}"; do if ! wait "$p"; then bfail=1; fi; done
echo "===== PHASE 2c done $(date) backfill_fail=$bfail ====="

# ---------- PHASE 3: ablation table ------------------------------------------
echo "===== PHASE 3: ablation table $(date) ====="
"$PY" -u -m microstructure_ed.eval.build_ablation_table --eval_root "$OUT" \
    --variants "${ALL_VARIANTS[@]}" \
    > "$OUT/logs/build_table.out" 2>&1
echo "--- ablation_table.txt ---"
cat "$OUT/ablation_table.txt" 2>/dev/null || echo "(no table produced)"

echo "############################################################"
echo "# TESTSET_V4 METRICS COMPLETE $(date) shard_fail=$fail merge_fail=$mfail backfill_fail=$bfail"
echo "############################################################"
touch "$OUT/.metrics_done"
