"""Checkpoint export, loading and download.

Training checkpoints (``checkpoint_epoch_XXX.pth``) hold
``{epoch, encoder, decoder, optimizer, lr_scheduler, curr_step, best_val_loss}``,
and the FM-DiT ``decoder`` state dict includes the frozen 2.5 B-parameter SD3.5
MMDiT backbone. Release checkpoints keep only what was trained:

* the full encoder state dict (ViT-H/14, attention pooler, projection),
* the decoder adapter (``token_generator``), ``pooled_proj``, ``tex_head``,
* the LoRA matrices injected into the MMDiT (``transformer.*lora_*``).

The frozen backbone is re-downloaded from ``stabilityai/stable-diffusion-3.5-medium``
when the decoder is constructed, so release and training checkpoints load into
the same modules.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

RELEASE_FORMAT = "microstructure_ed.release.v1"

# Hugging Face model repository holding the release checkpoints
HF_REPO_ID = os.environ.get("MSED_HF_REPO", "mahishguru/microstructure-encoder-decoder")

# variant -> (file in the HF repo, bottleneck width)
RELEASE_FILES = {
    "vitfmdit": ("fmdit_512/fmdit_512.pth", 512),
    "vitfmdit_768": ("fmdit_768/fmdit_768.pth", 768),
    "vitfmdit_1024": ("fmdit_1024/fmdit_1024.pth", 1024),
    "vitfmdit_1280": ("fmdit_1280/fmdit_1280.pth", 1280),
    "vitdit": ("baselines/vitdit.pth", 512),
    "vitsdxl": ("baselines/vitsdxl.pth", 512),
    "vitvqgan": ("baselines/vitvqgan.pth", None),
}


def _is_trained_decoder_key(key: str) -> bool:
    return not key.startswith("transformer.") or "lora_" in key


def export_release(src: str | Path, dst: str | Path, target_dim: int | None = None,
                   strip_frozen_backbone: bool = False) -> dict:
    """Reduce a training checkpoint to what is needed for inference.

    Optimiser / scheduler / discriminator state is always dropped. With
    ``strip_frozen_backbone=True`` (FM-DiT only) the frozen SD3.5 MMDiT
    weights are dropped as well, keeping the adapter, pooled projection,
    texture head and LoRA matrices. ViT-DiT trains its whole DiT, so its
    decoder is kept in full.
    """
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if "encoder" in ckpt and "decoder" in ckpt:
        dec = ckpt["decoder"]
        keep = _is_trained_decoder_key if strip_frozen_backbone else (lambda k: True)
        out = {
            "format": RELEASE_FORMAT,
            "epoch": ckpt.get("epoch"),
            "target_dim": target_dim,
            "stripped_frozen_backbone": bool(strip_frozen_backbone),
            "encoder": ckpt["encoder"],
            "decoder": {k: v for k, v in dec.items() if keep(k)},
        }
        n_drop = len(dec) - len(out["decoder"])
    else:
        drop = {"optimizer", "optimizer_state_dict", "lr_scheduler", "lr_scheduler_state_dict", "disc"}
        out = {k: v for k, v in ckpt.items() if k not in drop}
        out["format"] = RELEASE_FORMAT
        n_drop = 0
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    return {"dropped_frozen_decoder_tensors": n_drop, "path": str(dst)}


def load_encoder_decoder(encoder: torch.nn.Module, decoder: torch.nn.Module,
                         path: str | Path, map_location="cpu") -> dict:
    """Load a training or release checkpoint into (encoder, decoder).

    For stripped FM-DiT release checkpoints, the only decoder keys allowed to
    be missing are those of the frozen SD3.5 backbone (already initialised from
    Hugging Face); everything else is loaded strictly.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    if ckpt.get("stripped_frozen_backbone", False):
        res = decoder.load_state_dict(ckpt["decoder"], strict=False)
        bad_missing = [k for k in res.missing_keys if _is_trained_decoder_key(k)]
        if bad_missing or res.unexpected_keys:
            raise RuntimeError(
                f"checkpoint/decoder mismatch: missing trained keys {bad_missing[:5]}, "
                f"unexpected {res.unexpected_keys[:5]}")
    else:
        decoder.load_state_dict(ckpt["decoder"])
    return ckpt


TEXTURE_PRIOR_VARIANTS = ("vitfmdit", "vitfmdit_768", "vitfmdit_1024", "vitfmdit_1280")


def download(variant: str, cache_dir: str | Path | None = None) -> Path:
    """Download a release checkpoint from the Hugging Face Hub; returns the local path."""
    from huggingface_hub import hf_hub_download

    if variant not in RELEASE_FILES:
        raise KeyError(f"unknown variant {variant!r}; choose from {sorted(RELEASE_FILES)}")
    return Path(hf_hub_download(HF_REPO_ID, RELEASE_FILES[variant][0], cache_dir=cache_dir))


def download_texture_prior(variant: str | None, root: str | Path | None = None) -> Path:
    """Download a z -> ODF-histogram texture prior (``head.pt`` + ``codebook.npy``).

    ``variant=None`` fetches the shared default codebook/head (``texture_prior/``).
    Files are placed where the calibration code looks for them:
    ``$MSED_ROOT/eval_outputs/texture_prior[_<variant>]/``.
    """
    import shutil
    from huggingface_hub import hf_hub_download

    if variant is not None and variant not in TEXTURE_PRIOR_VARIANTS:
        raise KeyError(f"no texture prior for {variant!r}; choose from {TEXTURE_PRIOR_VARIANTS}")
    sub = "texture_prior" if variant is None else f"texture_prior_{variant}"
    root = Path(root or os.environ.get("MSED_ROOT", os.getcwd()))
    out = root / "eval_outputs" / sub
    out.mkdir(parents=True, exist_ok=True)
    for name in ("head.pt", "codebook.npy"):
        shutil.copy(hf_hub_download(HF_REPO_ID, f"texture_priors/{sub}/{name}"), out / name)
    return out
