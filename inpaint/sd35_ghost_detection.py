"""Ghost and seam detection for AI Replace."""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class GhostDetectionResult:
    opacity_score: float
    contrast_score: float
    edge_seam_score: float
    conf_drop: float
    rejected: bool
    reject_reason: str | None


def _as_arr(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def compute_opacity_score(composite: Image.Image, original: Image.Image, object_mask: np.ndarray) -> float:
    comp = _as_arr(composite)
    orig = _as_arr(original)
    active = object_mask > 0.5
    if not np.any(active):
        return 0.0
    comp_luma = comp[..., 0] * 0.299 + comp[..., 1] * 0.587 + comp[..., 2] * 0.114
    orig_luma = orig[..., 0] * 0.299 + orig[..., 1] * 0.587 + orig[..., 2] * 0.114
    internal_texture = float(np.std(comp_luma[active]))
    foreground_delta = float(np.mean(np.abs(comp_luma[active] - orig_luma[active])))
    texture_score = np.clip(internal_texture / 26.0, 0.0, 1.0)
    delta_score = np.clip(foreground_delta / 18.0, 0.0, 1.0)
    return float(np.clip(0.4 * texture_score + 0.6 * delta_score, 0.0, 1.0))


def compute_contrast_score(composite: Image.Image, original: Image.Image, object_mask: np.ndarray) -> float:
    comp = _as_arr(composite)
    orig = _as_arr(original)
    active = object_mask > 0.5
    if not np.any(active):
        return 0.0
    diff = np.mean(np.abs(comp[active] - orig[active]))
    return float(np.clip(diff / 32.0, 0.0, 1.0))


def compute_edge_seam_score(composite: Image.Image, original: Image.Image, object_mask: np.ndarray) -> float:
    mask_img = Image.fromarray((object_mask > 0.5).astype(np.uint8) * 255, mode="L")
    edge = np.asarray(mask_img.filter(ImageFilter.FIND_EDGES), dtype=np.float32) > 0
    if not np.any(edge):
        return 1.0
    comp = _as_arr(composite)
    orig = _as_arr(original)
    seam = float(np.mean(np.abs(comp[edge] - orig[edge])))
    return float(np.clip(1.0 - seam / 96.0, 0.0, 1.0))


def detect_ghost(original, composite, object_mask, detector_conf_before=1.0, detector_conf_after=1.0) -> GhostDetectionResult:
    opacity_score = compute_opacity_score(composite, original, object_mask)
    contrast_score = compute_contrast_score(composite, original, object_mask)
    edge_seam_score = compute_edge_seam_score(composite, original, object_mask)
    conf_drop = float(detector_conf_before) - float(detector_conf_after)
    reject_reason = None
    if opacity_score < 0.35:
        reject_reason = "ghost_low_opacity"
    if contrast_score < 0.30:
        reject_reason = "ghost_low_contrast"
    if conf_drop > 0.15:
        reject_reason = "ghost_after_harmonization"
    return GhostDetectionResult(
        opacity_score=round(float(opacity_score), 4),
        contrast_score=round(float(contrast_score), 4),
        edge_seam_score=round(float(edge_seam_score), 4),
        conf_drop=round(float(conf_drop), 4),
        rejected=reject_reason is not None,
        reject_reason=reject_reason,
    )
