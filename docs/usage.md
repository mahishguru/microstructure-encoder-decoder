# Usage: data, training, evaluation

## Weights

```bash
python scripts/download_weights.py            # FM-DiT-512 + texture prior -> checkpoints/, eval_outputs/
python scripts/download_weights.py --all      # every model
mkdir -p checkpoints_sam && wget -P checkpoints_sam \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth   # SAM ViT-B, for evaluation / calibration
```

Stable Diffusion 3.5 medium is a gated model. Accept its licence on Hugging Face and `export HF_TOKEN=...`. The OpenCLIP ViT-H/14, DiT-XL/2, SDXL and VAE weights download automatically.

## From a decoded image to an RVE

To obtain an RVE, resize the decoded image to 300 × 300 with nearest-neighbour interpolation, save it as PNG, and decode it with [orientation-codec](https://github.com/mahishguru/orientation-codec) (`decode_image_pixelwise`) using `microstructure_ed/assets/class_means.json`. For decoder outputs, segment grains with `microstructure_ed.segmentation.sam_segment` rather than the codec's colour-tolerance rule.

## Data

The training dataset is published as [Zenodo record 23036836](https://zenodo.org/records/23036836) (DOI [10.5281/zenodo.23036836](https://doi.org/10.5281/zenodo.23036836), CC BY 4.0; files available on request through Zenodo). Extract it and split it with `scripts/split_dataset.py` as below.

Training images are 8-bit orientation-codec PNGs of 300 × 300 RVEs, named `<class>_orientation_<id>.png` with one sub-folder per alloy class. They are produced with `orientation-codec batch-global` and the class means in [`microstructure_ed/assets/class_means.json`](../microstructure_ed/assets/class_means.json). The papers use 101,000 RVEs, augmented from measured ODF and grain-statistics conditions of extruded Mg alloys, split 90/10:

```bash
python scripts/split_dataset.py /path/to/rve --out . --test-split 0.10 --seed 42
# -> dataset_train/ (90,900 images), dataset_test/ (10,100 images)
```

The trainers take a further 10 % validation split of `dataset_train/` (seed 42).

## Training

```bash
export HF_TOKEN=...
scripts/train_fmdit.sh 512          # also 768, 1024, 1280
scripts/train_baseline.sh vitdit    # vitsdxl, vitvqgan
```

`train_fmdit.sh` runs the recipe of the released FM-DiT models:
- 40 epochs, AdamW, batch 24 per GPU × 4 GPUs × gradient accumulation 6 (effective 576);
- LoRA 64/64 on attention and feed-forward projections from epoch 4, ViT unfreeze at epoch 8;
- CFG dropout 0.1, texture head with λ = 0.5;
- orientation-distribution loss settings from `config.json → training.fmdit.orient_v2`.

Checkpoints go to `checkpoints/fmdit_<W>_v4scratch/`. The paper models were trained on 4 × NVIDIA H200 (141 GB). On smaller GPUs, lower `FMDIT_BATCH_SIZE_OVERRIDE` and raise `FMDIT_ACCUM_OVERRIDE` so that the effective batch stays at 576.

Training resumes from `FMDIT_RESUME_FROM_OVERRIDE=/path/to/checkpoint.pth`. All hyperparameters live in [`microstructure_ed/config.json`](../microstructure_ed/config.json); point `MSED_CONFIG` at your own copy to change them.

## Evaluation

```bash
# per FM-DiT variant (one GPU each): texture prior, test-set reconstruction, SAM, ODF calibration
scripts/eval_testset.sh vitfmdit      checkpoints/fmdit_512_v4scratch/checkpoint_epoch_040.pth  0
scripts/eval_testset.sh vitfmdit_768  checkpoints/fmdit_768_v4scratch/checkpoint_epoch_040.pth  1
scripts/eval_testset.sh vitfmdit_1024 checkpoints/fmdit_1024_v4scratch/checkpoint_epoch_040.pth 2
scripts/eval_testset.sh vitfmdit_1280 checkpoints/fmdit_1280_v4scratch/checkpoint_epoch_040.pth 3

# metrics for all variants (raw and calibrated rows) and the results table
scripts/compute_testset_metrics.sh      # -> eval_outputs/testset_v4/ablation_table.txt
```

The released checkpoints from `scripts/download_weights.py` work in place of the training checkpoints.

Metrics are implemented in `microstructure_ed/eval/`:
- **Image metrics** (`compute_metrics.py`): FID, SSIM, MS-SSIM, LPIPS.
- **Materials metrics** (`material_metrics.py`): grain-size and aspect-ratio error, orientation EMD on a binned fundamental-zone ODF, mean HCP disorientation, and grain-matched orientation errors.

The z-only **ODF calibration** (`texture_prior.py`, `odf_calibrate.py`) predicts an ODF histogram from z and repaints whole grains towards it. MERIDIAN uses it when converting decoded images into RVEs.

## Environment variables

| Variable | Meaning | Default |
|---|---|---|
| `MSED_ROOT` | root for `dataset_*`, `checkpoints/`, `eval_outputs/`, `logs/` | current directory |
| `MSED_CONFIG` | alternative `config.json` | packaged config |
| `HF_TOKEN` | Hugging Face token (SD3.5 access) | — |
| `SAM_CHECKPOINT` | SAM ViT-B weights | `$MSED_ROOT/checkpoints_sam/sam_vit_b_01ec64.pth` |
| `MSED_HF_REPO` | model repository for `download_weights.py` | `mahishguru/microstructure-encoder-decoder` |
| `FMDIT_TARGET_DIM` | default FM-DiT bottleneck width in the process | 512 |
| `FMDIT_LORA_RANK_OVERRIDE`, `FMDIT_LORA_ALPHA_OVERRIDE`, `FMDIT_LORA_FFN`, `FMDIT_TEXHEAD` | FM-DiT architecture flags; must match the checkpoint (64, 64, 1, 1 for the released models) | config / off |
| `FMDIT_*_OVERRIDE`, `FMDIT_LORA_START_EPOCH`, `FMDIT_CFG_DROPOUT`, `FMDIT_TEXHEAD_LAMBDA` | training-recipe overrides (set by `scripts/train_fmdit.sh`) | config |

## Released checkpoints and licences

The Hugging Face repository holds the trained weights only: the encoder, adapters and LoRA. The frozen SD3.5 backbone is fetched from Stability AI. The weights inherit the terms of the models they were fine-tuned from:

| Checkpoint | Terms |
|---|---|
| FM-DiT-512/768/1024/1280 | [Stability AI Community License](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/blob/main/LICENSE.md) for the SD3.5-derived parts; encoder MIT |
| ViT-DiT | CC BY-NC 4.0 (fine-tuned DiT-XL/2), **non-commercial** |
| ViT-SDXL | MIT (adapters); use with SDXL subject to CreativeML Open RAIL++-M |
| ViT-VQGAN | Apache-2.0 (fine-tuned PaintMind) |
