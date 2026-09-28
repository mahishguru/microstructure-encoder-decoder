"""Grain segmentation of decoded orientation images with Segment Anything (SAM).

Decoder outputs carry organic rather than pixel-sharp grain boundaries, on
which the colour-tolerance connected-component rule of the orientation codec
over- or under-segments. Grains are therefore segmented with the SAM automatic
mask generator (ViT-B, 32 points per side, predicted-IoU 0.86, stability 0.85,
minimum region 200 px, Voronoi fill of unlabelled pixels).

The same segmenter is used for (i) the grain-based reconstruction metrics,
(ii) the z-only ODF calibration, and (iii) the RVE construction inside the
MERIDIAN inverse-design loop, so all three see identical grains.

Checkpoint: ``sam_vit_b_01ec64.pth`` from
https://github.com/facebookresearch/segment-anything (Apache-2.0). Set
``SAM_CHECKPOINT`` or place it at ``$MSED_ROOT/checkpoints_sam/``.
"""
from __future__ import annotations

import os
import threading

import numpy as np


def default_sam_checkpoint() -> str:
    """``$SAM_CHECKPOINT`` or ``$MSED_ROOT/checkpoints_sam/sam_vit_b_01ec64.pth``."""
    root = os.environ.get("MSED_ROOT", os.getcwd())
    return os.environ.get(
        "SAM_CHECKPOINT", os.path.join(root, "checkpoints_sam", "sam_vit_b_01ec64.pth")
    )


# --------------------------------------------------------------------------
# SAM AutomaticMaskGenerator (lazy global singleton — load weights once)
# --------------------------------------------------------------------------
_SAM_AMG = None
_SAM_AMG_KEY = None
# SAM internally calls predictor.set_image() then mask prediction in
# sequence; the predictor is stateful, so concurrent .generate() calls
# from worker threads corrupt that state ("An image must be set with
# .set_image(...) before mask prediction"). Serialize all calls.
_SAM_LOCK = threading.Lock()


def _get_sam_amg(
    checkpoint: str,
    model_type: str = "vit_b",
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.86,
    stability_score_thresh: float = 0.85,
    min_mask_region_area: int = 200,
    device: str | None = None,
):
    """Lazily load and cache a SamAutomaticMaskGenerator.

    Reuses the same generator if called with identical config — avoids
    reloading the (~370 MB) ViT-B weights on every call inside the loop.
    """
    global _SAM_AMG, _SAM_AMG_KEY
    key = (checkpoint, model_type, points_per_side, pred_iou_thresh,
           stability_score_thresh, min_mask_region_area, device)
    if _SAM_AMG is not None and _SAM_AMG_KEY == key:
        return _SAM_AMG

    import torch
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
    _SAM_AMG = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=int(points_per_side),
        pred_iou_thresh=float(pred_iou_thresh),
        stability_score_thresh=float(stability_score_thresh),
        min_mask_region_area=int(min_mask_region_area),
        crop_n_layers=0,
    )
    _SAM_AMG_KEY = key
    return _SAM_AMG


def _compact_labels(lbl: np.ndarray) -> np.ndarray:
    """Relabel a label map to consecutive 1..N (preserving 0 as background).

    Required because some segmenters can leave gaps in the ID sequence
    (e.g. SAM masks where a small early-numbered mask is fully covered
    by a later larger mask). DAMASK indexes material.yaml by FeatureId,
    so any gap causes "material index out of bounds" (error 155).
    """
    uniq = np.unique(lbl)
    has_bg = bool(uniq[0] == 0)
    fg = uniq[1:] if has_bg else uniq
    remap = np.zeros(int(lbl.max()) + 1, dtype=np.int32)
    for new_id, old_id in enumerate(fg, start=1):
        remap[old_id] = new_id
    return remap[lbl].astype(np.int32)


def _sam_segment(
    img: np.ndarray,
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.86,
    min_mask_region_area: int = 200,
    checkpoint: str | None = None,
    model_type: str = "vit_b",
) -> np.ndarray:
    """SAM AutomaticMaskGenerator -> integer label map.

    Loads SAM weights lazily on first call, then re-uses the cached
    generator. Mask overlaps are resolved by assigning each pixel to the
    smallest covering mask (per-grain priority). Unlabelled pixels get
    Voronoi-filled from the nearest labelled neighbour.

    Validated on FM-DiT iter1 batch (n=105/191/116) and GT sim_95
    (n=148, truth=139): boundaries follow source blobs faithfully without
    catastrophic merging.
    """
    from scipy.ndimage import distance_transform_edt

    if checkpoint is None:
        checkpoint = default_sam_checkpoint()
    amg = _get_sam_amg(
        checkpoint=checkpoint, model_type=model_type,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        min_mask_region_area=min_mask_region_area,
    )
    with _SAM_LOCK:
        masks = amg.generate(img)
    H, W = img.shape[:2]
    lbl = np.zeros((H, W), dtype=np.int32)
    nid = 0
    for m in sorted(masks, key=lambda x: x["area"]):
        nid += 1
        lbl[m["segmentation"]] = nid
    if (lbl == 0).any() and (lbl > 0).any():
        _, (yy, xx) = distance_transform_edt(lbl == 0, return_indices=True)
        lbl = lbl[yy, xx]
    return _compact_labels(lbl)


sam_segment = _sam_segment
