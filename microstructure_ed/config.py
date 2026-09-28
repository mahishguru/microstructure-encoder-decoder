"""
config.py - Unified Configuration Loader
Loads settings from config.json and exposes them as module-level variables
for the ViT Encoder, ViT-SDXL, ViT-FMDiT, and ViT-DiT Decoders.
"""
import os
import json
import torch

# Data, checkpoints and logs are resolved relative to MSED_ROOT (default: the
# current working directory). The configuration itself ships with the package
# and can be replaced with MSED_CONFIG=/path/to/config.json.
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.environ.get("MSED_ROOT", os.getcwd()))
CONFIG_PATH = os.environ.get("MSED_CONFIG", os.path.join(PACKAGE_DIR, "config.json"))
ASSETS_DIR = os.path.join(PACKAGE_DIR, "assets")

with open(CONFIG_PATH, "r") as f:
    _config = json.load(f)

# ── Paths ────────────────────────────────────────────────────────────────────
CHECKPOINTS_DIR      = os.path.join(ROOT_DIR, _config["paths"]["checkpoints_dir"])
ENCODER_CKPT         = os.path.join(ROOT_DIR, _config["paths"]["encoder_ckpt"])
INPUT_IMAGES_DIR     = os.path.join(ROOT_DIR, _config["paths"]["data_dir"])
DATA_DIR             = INPUT_IMAGES_DIR  # Alias used by FMDiT/ViTDiT trainers
TEST_DIR             = os.path.join(ROOT_DIR, _config["paths"].get("test_dir", "dataset_test"))
INFERENCE_OUTPUT_DIR = os.path.join(ROOT_DIR, _config["paths"]["inference_output_dir"])
LOGS_DIR             = os.path.join(ROOT_DIR, _config["paths"]["logs_dir"])
EPOCH_SAMPLES_DIR    = os.path.join(ROOT_DIR, _config["paths"]["epoch_samples_dir"])
LOSS_LOG_FILE        = os.path.join(LOGS_DIR, "training_loss.csv")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Encoder (ViT-H/14 → 512-D) ───────────────────────────────────────────────
VIT_MODEL_NAME         = _config["models"]["encoder"]["vit_model_name"]
VIT_PRETRAINED         = _config["models"]["encoder"]["vit_pretrained"]
VIT_DIM                = _config["models"]["encoder"]["vit_dim"]           # 1280
# Default bottleneck width used by the encoder constructor when no `target_dim`
# argument is passed. Each decoder variant declares its OWN module-level
# `TARGET_DIM` constant inside its `decoder_arch_pretrained.py` (e.g.
# vitfmdit/=512, vitfmdit_768/=768, vitfmdit_1024/=1024). Do NOT rely on this
# global to determine a variant's bottleneck — it is only a fallback.
DEFAULT_TARGET_DIM     = _config["models"]["encoder"]["target_dim"]
TARGET_DIM             = DEFAULT_TARGET_DIM  # backward-compat alias
ATTENTION_POOLER_HEADS = _config["models"]["encoder"]["attention_pooler_heads"]
ATTENTION_POOLER_NUM_QUERIES = _config["models"]["encoder"].get("attention_pooler_num_queries", 4)

# ── SDXL Decoder ──────────────────────────────────────────────────────────────
SDXL_MODEL_NAME = _config["models"]["sdxl"]["model_name"]
VAE_MODEL_NAME  = SDXL_MODEL_NAME  # Alias used in SDXL trainer
VAE_SUBFOLDER   = _config["models"]["sdxl"]["vae_subfolder"]

# SDXL Architecture constants
LATENT_CHANNELS = 4
LATENT_SIZE     = 64
IMAGE_SIZE      = 512
NUM_TOKENS      = 1024
NUM_POS_EMBED   = 1025
N_TOKENS        = 77
COND_DIM        = 2048             # SDXL cross-attn dim
LATENT_TO_TOKENS_DEPTH = 3

# SDXL Training
COMPRESSOR_LR       = _config["training"]["sdxl"]["compressor_lr"]
SDXL_BATCH_SIZE     = _config["training"]["sdxl"]["batch_size"]
BATCH_SIZE          = SDXL_BATCH_SIZE # Alias for SDXL trainer
SDXL_ACCUM_STEPS    = _config["training"]["sdxl"]["accumulation_steps"]
ACCUMULATION_STEPS  = SDXL_ACCUM_STEPS # Alias for SDXL trainer
P_DROP              = _config["training"]["sdxl"]["p_drop"]
SDXL_NUM_INFERENCE_STEPS = _config["training"]["sdxl"]["num_inference_steps"]
NUM_INFERENCE_STEPS = SDXL_NUM_INFERENCE_STEPS  # SDXL-only alias used by SDXL decoder/inference
USE_AMP             = _config["training"]["sdxl"]["use_amp"]

# SDXL Loss Weights
LAMBDA_DIFF_START        = _config["training"]["sdxl"]["loss_weights"]["lambda_diff_start"]
LAMBDA_DIFF_END          = _config["training"]["sdxl"]["loss_weights"]["lambda_diff_end"]
LAMBDA_CONTRASTIVE_START = _config["training"]["sdxl"]["loss_weights"]["lambda_contrastive_start"]
LAMBDA_CONTRASTIVE_END   = _config["training"]["sdxl"]["loss_weights"]["lambda_contrastive_end"]
LAMBDA_RECON_START       = _config["training"]["sdxl"]["loss_weights"]["lambda_recon_start"]
LAMBDA_RECON_END         = _config["training"]["sdxl"]["loss_weights"]["lambda_recon_end"]
LAMBDA_VICREG_START      = _config["training"]["sdxl"]["loss_weights"]["lambda_vicreg_start"]
LAMBDA_VICREG_END        = _config["training"]["sdxl"]["loss_weights"]["lambda_vicreg_end"]
LAMBDA_LAB_START         = _config["training"]["sdxl"]["loss_weights"]["lambda_lab_start"]
LAMBDA_LAB_END           = _config["training"]["sdxl"]["loss_weights"]["lambda_lab_end"]
LAMBDA_FREQ_START        = _config["training"]["sdxl"]["loss_weights"]["lambda_freq_start"]
LAMBDA_FREQ_END          = _config["training"]["sdxl"]["loss_weights"]["lambda_freq_end"]
WARMUP_EPOCHS_LOSS       = _config["training"]["sdxl"]["loss_weights"]["warmup_epochs"]

CONTRASTIVE_TEMPERATURE = _config["training"]["sdxl"]["contrastive"]["temperature"]
PIXEL_NOISE_SCALE      = _config["training"]["sdxl"]["contrastive"]["pixel_noise_scale"]
VICREG_VAR_WEIGHT      = _config["training"]["sdxl"]["vicreg"]["var_weight"]
VICREG_COV_WEIGHT      = _config["training"]["sdxl"]["vicreg"]["cov_weight"]
VICREG_VAR_TARGET      = _config["training"]["sdxl"]["vicreg"]["var_target"]
PIXEL_LOSS_ALPHA_T_THRESHOLD = _config["training"]["sdxl"]["pixel_loss"]["alpha_t_threshold"]

SDXL_SAVE_EVERY_EPOCH = _config["checkpointing"]["sdxl_save_every_epoch"]
SAVE_EVERY_EPOCH      = SDXL_SAVE_EVERY_EPOCH # Alias for SDXL trainer
SDXL_RESUME_FROM      = _config["checkpointing"]["sdxl_resume_from"]
RESUME_FROM           = SDXL_RESUME_FROM       # Alias for SDXL trainer
if RESUME_FROM: RESUME_FROM = os.path.join(ROOT_DIR, RESUME_FROM)

# ── FM-DiT Decoder (SD3.5 Medium Hijack) ─────────────────────────────────────
FMDIT_MODEL_ID              = _config["models"]["fmdit"]["sd35_model_id"]
FMDIT_VAE_SUBFOLDER         = _config["models"]["fmdit"]["sd35_vae_subfolder"]
FMDIT_JOINT_ATTENTION_DIM   = _config["models"]["fmdit"]["sd35_joint_attention_dim"]
FMDIT_POOLED_DIM            = _config["models"]["fmdit"]["sd35_pooled_dim"]
FMDIT_VAE_CHANNELS          = _config["models"]["fmdit"]["sd35_vae_channels"]
FMDIT_VAE_DOWNSAMPLE_FACTOR = _config["models"]["fmdit"]["sd35_vae_downsample_factor"]
FMDIT_ADAPTER_INNER_DIM     = _config["models"]["fmdit"]["adapter_inner_dim"]
FMDIT_NUM_TOKENS            = _config["models"]["fmdit"]["adapter_num_tokens"]
FMDIT_ADAPTER_DEPTH         = _config["models"]["fmdit"]["adapter_depth"]
FMDIT_ADAPTER_NUM_HEADS     = _config["models"]["fmdit"]["adapter_num_heads"]
FMDIT_DECODER_LORA_RANK     = _config["models"]["fmdit"].get("decoder_lora_rank", 0)
FMDIT_DECODER_LORA_ALPHA    = _config["models"]["fmdit"].get("decoder_lora_alpha", FMDIT_DECODER_LORA_RANK)
FMDIT_VAE_SCALING_FACTOR    = 1.5305  # Standard for SD3/SD3.5

FMDIT_LR                      = _config["training"]["fmdit"]["lr"]
FMDIT_WEIGHT_DECAY            = _config["training"]["fmdit"]["weight_decay"]
FMDIT_BATCH_SIZE              = _config["training"]["fmdit"]["batch_size"]
FMDIT_ACCUMULATION_STEPS      = _config["training"]["fmdit"]["accumulation_steps"]
FMDIT_USE_AMP                 = _config["training"]["fmdit"]["use_amp"]
FMDIT_FLOW_SHIFT              = _config["training"]["fmdit"]["flow_shift"]
FM_NUM_TRAIN_TIMESTEPS        = _config["training"]["fmdit"]["num_train_timesteps"]
FMDIT_NUM_INFERENCE_STEPS     = _config["training"]["fmdit"]["num_inference_steps"]
FMDIT_NUM_INFERENCE_STEPS_TRAIN = _config["training"]["fmdit"]["num_inference_steps_sample"]
# ── FMDiT pixel-space loss sampling (shared by frequency + orientation) ──────
FMDIT_PIXEL_MAX_SAMPLES       = _config["training"]["fmdit"].get("pixel_max_samples", 2)
FMDIT_PIXEL_T_THRESHOLD       = _config["training"]["fmdit"].get("pixel_t_threshold", 0.7)
# ── FMDiT orientation loss (differentiable, HCP-symmetry-aware) ──────────────
_fmdit_orient = _config["training"]["fmdit"].get("orient", {})
FMDIT_LAMBDA_ORIENT          = _fmdit_orient.get("lambda_orient", 0.3)
FMDIT_ORIENT_WARMUP_EPOCHS   = _fmdit_orient.get("warmup_epochs", 6)
FMDIT_ORIENT_RAMP_EPOCHS     = _fmdit_orient.get("ramp_epochs", 3)
FMDIT_ORIENT_PIX_WEIGHT      = _fmdit_orient.get("pix_weight", 0.7)
FMDIT_ORIENT_POOL_WEIGHT     = _fmdit_orient.get("pool_weight", 0.3)
FMDIT_ORIENT_POOL_SIZE       = _fmdit_orient.get("pool_size", 16)
FMDIT_ORIENT_EPS             = _fmdit_orient.get("eps", 1e-4)
# v2 metric-aware orientation loss (Track B; consumed ONLY by vitfmdit_1280).
# Missing block -> empty dict -> variant falls back to legacy behaviour.
FMDIT_ORIENT_V2              = _config["training"]["fmdit"].get("orient_v2", {})
FMDIT_SAVE_EVERY_EPOCH        = _config["checkpointing"]["fmdit_save_every_epoch"]
FMDIT_RESUME_FROM             = _config["checkpointing"]["fmdit_resume_from"]
if FMDIT_RESUME_FROM: FMDIT_RESUME_FROM = os.path.join(ROOT_DIR, FMDIT_RESUME_FROM)

# Aliases for FMDIT backward compatibility
FM_SHIFT             = FMDIT_FLOW_SHIFT
# NOTE: NUM_INFERENCE_STEPS is intentionally bound to the SDXL value below
# (SDXL imports it by that name).  FMDIT code must use FMDIT_NUM_INFERENCE_STEPS
# directly — do NOT alias it here, or SDXL inference will silently use the
# FMDIT step count.

# ── ViTDiT Decoder (DiT-XL/2 + DDPM) ────────────────────────────────────────
VITDIT_PRETRAINED_URL        = _config["models"]["vitdit"]["dit_pretrained"] # "facebook/DiT-XL-2-256"
VITDIT_VAE_MODEL_ID          = _config["models"]["vitdit"]["sd_vae_model_id"]
VITDIT_HIDDEN_SIZE           = _config["models"]["vitdit"]["dit_hidden_size"]
VITDIT_DEPTH                 = _config["models"]["vitdit"]["dit_depth"]
VITDIT_NUM_HEADS             = _config["models"]["vitdit"]["dit_num_heads"]
VITDIT_PATCH_SIZE            = _config["models"]["vitdit"]["dit_patch_size"]
VITDIT_CLASS_EMB_DIM         = _config["models"]["vitdit"]["dit_class_emb_dim"]
VITDIT_VAE_CHANNELS          = _config["models"]["vitdit"]["sd_vae_channels"]
VITDIT_VAE_DOWNSAMPLE_FACTOR = _config["models"]["vitdit"]["sd_vae_downsample_factor"]

VITDIT_LR                      = _config["training"]["vitdit"]["lr"]
VITDIT_WEIGHT_DECAY            = _config["training"]["vitdit"]["weight_decay"]
VITDIT_BATCH_SIZE              = _config["training"]["vitdit"]["batch_size"]
VITDIT_ACCUMULATION_STEPS      = _config["training"]["vitdit"]["accumulation_steps"]
VITDIT_USE_AMP                 = _config["training"]["vitdit"]["use_amp"]
VITDIT_NUM_TIMESTEPS           = _config["training"]["vitdit"]["ddpm_num_train_timesteps"]
VITDIT_BETA_START              = _config["training"]["vitdit"]["ddpm_beta_start"]
VITDIT_BETA_END                = _config["training"]["vitdit"]["ddpm_beta_end"]
VITDIT_NUM_INFERENCE_STEPS     = _config["training"]["vitdit"]["num_inference_steps"]
VITDIT_NUM_INFERENCE_STEPS_TRAIN = _config["training"]["vitdit"]["num_inference_steps_sample"]
# ── ViTDiT pixel-space loss sampling (shared by frequency + orientation) ────
VITDIT_PIXEL_MAX_SAMPLES      = _config["training"]["vitdit"].get("pixel_max_samples", 2)
VITDIT_PIXEL_T_THRESHOLD      = _config["training"]["vitdit"].get("pixel_t_threshold", 0.7)
# ── ViTDiT orientation loss (differentiable, HCP-symmetry-aware) ────────────
_vitdit_orient = _config["training"]["vitdit"].get("orient", {})
VITDIT_LAMBDA_ORIENT          = _vitdit_orient.get("lambda_orient", 0.6)
VITDIT_ORIENT_WARMUP_EPOCHS   = _vitdit_orient.get("warmup_epochs", 6)
VITDIT_ORIENT_RAMP_EPOCHS     = _vitdit_orient.get("ramp_epochs", 3)
VITDIT_ORIENT_PIX_WEIGHT      = _vitdit_orient.get("pix_weight", 0.8)
VITDIT_ORIENT_POOL_WEIGHT     = _vitdit_orient.get("pool_weight", 0.2)
VITDIT_ORIENT_POOL_SIZE       = _vitdit_orient.get("pool_size", 16)
VITDIT_ORIENT_EPS             = _vitdit_orient.get("eps", 1e-4)
VITDIT_SAVE_EVERY_EPOCH        = _config["checkpointing"]["vitdit_save_every_epoch"]
VITDIT_RESUME_FROM             = _config["checkpointing"]["vitdit_resume_from"]
if VITDIT_RESUME_FROM: VITDIT_RESUME_FROM = os.path.join(ROOT_DIR, VITDIT_RESUME_FROM)

# Aliases for ViTDiT backward compatibility (so decoder doesn't break)
DIT_HIDDEN_SIZE          = VITDIT_HIDDEN_SIZE
DIT_DEPTH                = VITDIT_DEPTH
DIT_NUM_HEADS            = VITDIT_NUM_HEADS
DIT_PATCH_SIZE           = VITDIT_PATCH_SIZE
DIT_CLASS_EMB_DIM        = VITDIT_CLASS_EMB_DIM
SD_VAE_MODEL_ID          = VITDIT_VAE_MODEL_ID
SD_VAE_CHANNELS          = VITDIT_VAE_CHANNELS
SD_VAE_DOWNSAMPLE_FACTOR = VITDIT_VAE_DOWNSAMPLE_FACTOR
DDPM_NUM_TIMESTEPS       = VITDIT_NUM_TIMESTEPS
DDPM_BETA_START          = VITDIT_BETA_START
DDPM_BETA_END            = VITDIT_BETA_END

# ── Shared Training ───────────────────────────────────────────────────────────
NUM_EPOCHS    = _config["training"]["shared"]["num_epochs"]
NUM_WORKERS   = _config["training"]["shared"]["num_workers"]
MAX_IMAGES    = _config["training"]["shared"]["max_images"]
WARMUP_STEPS  = _config["training"]["shared"]["warmup_steps"]
MAX_GRAD_NORM = _config["training"]["shared"]["max_grad_norm"]
VAL_SPLIT     = _config["training"]["shared"]["val_split"]

LOG_INTERVAL = 50 # Kept hardcoded as it's just a print frequency