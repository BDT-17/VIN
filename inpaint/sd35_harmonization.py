"""Layer-aware harmonization for AI Replace composites."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

from .sd35_mask_refinement import _pil_max_filter


@dataclass(frozen=True)
class MaskLayers:
    core: np.ndarray
    soft_boundary: np.ndarray
    fine_detail: np.ndarray
    contact_zone: np.ndarray


@dataclass(frozen=True)
class HarmonizationResult:
    image: Image.Image
    harmonization_score: float
    color_transfer_strength: float
    max_core_blend: float
    max_boundary_blend: float
    max_edge_blend: float
    shadow_alpha: float
    shadow_blur: int
    sharpen_strength: float


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return (mask > 0.5).astype(np.float32)
    inv = 1.0 - (mask > 0.5).astype(np.float32)
    return 1.0 - _pil_max_filter(inv, radius)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    return _pil_max_filter((mask > 0.5).astype(np.float32), radius)


def decompose_object_mask(object_mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> MaskLayers:
    obj = (object_mask > 0.5).astype(np.float32)
    core = _erode(obj, radius=12)
    soft_boundary = np.clip(obj - core, 0.0, 1.0)
    fine_detail = np.clip(_dilate(obj, radius=3) - _erode(obj, radius=3), 0.0, 1.0)
    contact_zone = np.zeros_like(obj, dtype=np.float32)
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    contact_y_start = int(y2 - max(1, (y2 - y1)) * 0.12)
    contact_y_start = max(0, min(obj.shape[0], contact_y_start))
    x1, x2 = max(0, x1), min(obj.shape[1], x2)
    y2 = max(0, min(obj.shape[0], y2))
    if x2 > x1 and y2 > contact_y_start:
        contact_zone[contact_y_start:y2, x1:x2] = obj[contact_y_start:y2, x1:x2]
    return MaskLayers(core=core, soft_boundary=soft_boundary, fine_detail=fine_detail, contact_zone=contact_zone)


def _ring_mask(mask: np.ndarray, outer: int = 24, inner: int = 4) -> np.ndarray:
    return np.clip(_dilate(mask, outer) - _dilate(mask, inner), 0.0, 1.0)


def _stats(image: np.ndarray, mask: np.ndarray):
    active = mask > 0.05
    if not np.any(active):
        return None, None
    pixels = image[active].astype(np.float32)
    return pixels.mean(axis=0), pixels.std(axis=0) + 1e-6


def _blend_layer(composite: np.ndarray, original: np.ndarray, mask: np.ndarray, max_alpha: float) -> np.ndarray:
    alpha = np.clip(mask[..., None] * float(max_alpha), 0.0, float(max_alpha))
    return composite * (1.0 - alpha) + original * alpha


def _apply_contact_shadow(composite: Image.Image, bbox, contact_zone: np.ndarray, config) -> Image.Image:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    shadow = Image.new("L", composite.size, 0)
    draw = ImageDraw.Draw(shadow)
    shadow_w = int(width * 0.35)
    shadow_h = max(2, int(height * 0.04))
    cx = int((x1 + x2) / 2)
    cy = y2 - shadow_h
    draw.ellipse((cx - shadow_w, cy - shadow_h, cx + shadow_w, cy + shadow_h), fill=int(255 * config.SHADOW_ALPHA))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=int(config.SHADOW_BLUR)))
    shadow_arr = np.asarray(shadow, dtype=np.float32) / 255.0
    if contact_zone.shape == shadow_arr.shape:
        shadow_arr *= np.clip(_dilate(contact_zone, 8), 0.0, 1.0)
    comp = np.asarray(composite.convert("RGB"), dtype=np.float32)
    comp = comp * (1.0 - shadow_arr[..., None] * 0.45)
    return Image.fromarray(np.clip(comp, 0, 255).astype(np.uint8), mode="RGB")


def _apply_core_sharpen(composite: Image.Image, core_mask: np.ndarray, strength: float) -> Image.Image:
    sharpened = ImageEnhance.Sharpness(composite).enhance(1.0 + float(strength))
    comp = np.asarray(composite.convert("RGB"), dtype=np.float32)
    sharp = np.asarray(sharpened.convert("RGB"), dtype=np.float32)
    alpha = np.clip(core_mask[..., None], 0.0, 1.0)
    out = comp * (1.0 - alpha) + sharp * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def harmonize_object_with_background(original_image, generated_composite, object_mask, mask_layers, config) -> HarmonizationResult:
    original = np.asarray(original_image.convert("RGB"), dtype=np.float32)
    composite = np.asarray(generated_composite.convert("RGB").resize(original_image.size), dtype=np.float32)
    obj = (object_mask > 0.5).astype(np.float32)
    ring = _ring_mask(obj)
    bg_mean, bg_std = _stats(original, ring)
    obj_mean, obj_std = _stats(composite, obj)
    strength = min(float(config.COLOR_TRANSFER_STRENGTH), float(config.MAX_COLOR_TRANSFER_STRENGTH), 0.15)
    if bg_mean is not None and obj_mean is not None:
        transferred = (composite - obj_mean) * (bg_std / obj_std) + bg_mean
        layer = np.clip(mask_layers.soft_boundary + mask_layers.fine_detail, 0.0, 1.0)[..., None] * strength
        composite = composite * (1.0 - layer) + transferred * layer
    composite = _blend_layer(composite, original, mask_layers.core, config.MAX_CORE_BLEND)
    composite = _blend_layer(composite, original, mask_layers.soft_boundary, config.MAX_SOFT_BOUNDARY_BLEND)
    composite = _blend_layer(composite, original, mask_layers.fine_detail, config.MAX_FINE_EDGE_BLEND)
    image = Image.fromarray(np.clip(composite, 0, 255).astype(np.uint8), mode="RGB")
    image = _apply_contact_shadow(image, _bbox_from_mask(obj), mask_layers.contact_zone, config)
    image = _apply_core_sharpen(image, mask_layers.core, config.SHARPEN_STRENGTH)
    score = 1.0 - min(1.0, strength + float(config.MAX_CORE_BLEND))
    return HarmonizationResult(
        image=image,
        harmonization_score=round(float(score), 4),
        color_transfer_strength=strength,
        max_core_blend=float(config.MAX_CORE_BLEND),
        max_boundary_blend=float(config.MAX_SOFT_BOUNDARY_BLEND),
        max_edge_blend=float(config.MAX_FINE_EDGE_BLEND),
        shadow_alpha=float(config.SHADOW_ALPHA),
        shadow_blur=int(config.SHADOW_BLUR),
        sharpen_strength=float(config.SHARPEN_STRENGTH),
    )


def _bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, mask.shape[1], mask.shape[0])
    return (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
