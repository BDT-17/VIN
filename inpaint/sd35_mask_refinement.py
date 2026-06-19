"""Mask generation and refinement for AI Replace."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - import is optional for non-torch tests
    torch = None
    F = None


BBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class AIReplaceMaskBundle:
    hard_mask: np.ndarray
    soft_mask: np.ndarray
    bbox: BBox
    expanded_bbox: BBox
    mask_area_ratio: float

    def to_latent(self, scale: int = 8):
        """Downsample hard_mask to latent space with nearest-neighbor binary values."""
        if torch is None or F is None:
            raise RuntimeError("torch is required for AIReplaceMaskBundle.to_latent()")
        tensor = torch.from_numpy(self.hard_mask.astype(np.float32))[None, None]
        return F.interpolate(tensor, scale_factor=1 / scale, mode="nearest")

    def to_pil(self, soft: bool = False) -> Image.Image:
        arr = self.soft_mask if soft else self.hard_mask
        return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="L")


def clamp_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(width, int(round(x1)))),
        max(0, min(height, int(round(y1)))),
        max(0, min(width, int(round(x2)))),
        max(0, min(height, int(round(y2)))),
    )


def bbox_to_mask(image_size: tuple, bbox: tuple) -> np.ndarray:
    """Create a binary mask. image_size=(H,W), bbox=(x1,y1,x2,y2)."""
    height, width = image_size
    x1, y1, x2, y2 = clamp_bbox(tuple(bbox), width, height)
    mask = np.zeros((height, width), dtype=np.float32)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1.0
    return mask


def _pil_max_filter(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.float32)
    size = radius * 2 + 1
    pil = Image.fromarray((mask > 0.5).astype(np.uint8) * 255, mode="L")
    return (np.asarray(pil.filter(ImageFilter.MaxFilter(size)), dtype=np.float32) / 255.0)


def _pil_blur(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.clip(mask, 0.0, 1.0).astype(np.float32)
    pil = Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8), mode="L")
    return (np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32) / 255.0)


def compute_expanded_bbox(mask: np.ndarray) -> BBox:
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))


def refine_mask(raw_mask: np.ndarray, bbox: tuple, config) -> AIReplaceMaskBundle:
    hard_mask = (raw_mask > 0.5).astype(np.float32)
    expanded = _pil_max_filter(hard_mask, int(config.AI_REPLACE_MASK_EXPAND_PX))
    soft_mask = _pil_blur(expanded, int(config.AI_REPLACE_MASK_BLUR_PX))
    return AIReplaceMaskBundle(
        hard_mask=hard_mask,
        soft_mask=np.clip(soft_mask, 0.0, 1.0).astype(np.float32),
        bbox=tuple(int(round(v)) for v in bbox),
        expanded_bbox=compute_expanded_bbox(expanded),
        mask_area_ratio=float(hard_mask.mean()),
    )


def hard_restore_outside_mask(original: Image.Image, generated: Image.Image, mask_bundle: AIReplaceMaskBundle) -> Image.Image:
    """Pixel-space safety net: generated pixels outside hard mask are discarded."""
    orig = np.asarray(original.convert("RGB"), dtype=np.uint8)
    gen = np.asarray(generated.convert("RGB").resize(original.size), dtype=np.uint8).copy()
    mask = mask_bundle.hard_mask
    if mask.shape != orig.shape[:2]:
        mask_img = mask_bundle.to_pil().resize(original.size, Image.NEAREST)
        mask = np.asarray(mask_img, dtype=np.float32) / 255.0
    gen[mask <= 0.5] = orig[mask <= 0.5]
    return Image.fromarray(gen, mode="RGB")


def outside_mask_diff(original: Image.Image, composite: Image.Image, mask_bundle: AIReplaceMaskBundle) -> float:
    orig = np.asarray(original.convert("RGB"), dtype=np.float32)
    comp = np.asarray(composite.convert("RGB").resize(original.size), dtype=np.float32)
    mask = mask_bundle.hard_mask
    if mask.shape != orig.shape[:2]:
        mask_img = mask_bundle.to_pil().resize(original.size, Image.NEAREST)
        mask = np.asarray(mask_img, dtype=np.float32) / 255.0
    outside = mask <= 0.5
    if not np.any(outside):
        return 0.0
    return float(np.mean(np.abs(comp[outside] - orig[outside])))
