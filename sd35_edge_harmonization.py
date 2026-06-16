"""Boundary-only harmonization for pasted pedestrians.

The operations here are intentionally local: only the contour band around the
person mask is softened and color-matched. The pedestrian core is preserved so
YOLO detectability remains high.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

try:
    import cv2
except ImportError:
    cv2 = None

from sd35_config import *


@dataclass
class EdgeHarmonizationResult:
    image: Image.Image
    mask: Image.Image
    metadata: dict


def _cfg(name, default):
    config = globals().get("EFFECTIVE_CONFIG", {})
    if isinstance(config, dict) and name in config:
        return config[name]
    return globals().get(name, default)


def edge_enabled():
    return bool(_cfg("EDGE_HARMONIZATION_ENABLED", True))


def edge_band_width():
    return max(2, int(_cfg("EDGE_BAND_WIDTH", 12)))


def effective_feather_radius(mask):
    radius = max(1, int(_cfg("EDGE_FEATHER_RADIUS", 7)))
    bbox = mask.getbbox()
    if bbox is None:
        return radius
    person_h = max(1, bbox[3] - bbox[1])
    # Tiny pedestrians lose detectability quickly; shrink feather automatically.
    return max(2, min(radius, int(max(2, person_h * 0.045))))


def edge_blur_radius():
    return max(0.0, float(_cfg("EDGE_BLUR_RADIUS", 3)))


def edge_color_strength():
    return max(0.0, min(1.0, float(_cfg("EDGE_COLOR_MATCH_STRENGTH", 0.35))))


def poisson_enabled():
    return bool(_cfg("POISSON_BLEND_ENABLED", False))


def clean_person_mask(mask):
    mask = mask.convert("L")
    hard = mask.point(lambda p: 255 if p >= 96 else 0)
    closed = hard.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    opened = closed.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    return opened


def create_edge_masks(mask):
    clean = clean_person_mask(mask)
    band = edge_band_width()
    filter_size = band * 2 + 1
    inner = clean.filter(ImageFilter.MinFilter(filter_size))
    outer = clean.filter(ImageFilter.MaxFilter(filter_size))
    edge_band = ImageChops.subtract(outer, inner)
    inner_boundary = ImageChops.subtract(clean, inner)
    outer_boundary = ImageChops.subtract(outer, clean)
    feather = effective_feather_radius(clean)
    soft_alpha = clean.filter(ImageFilter.GaussianBlur(radius=feather))
    edge_alpha = edge_band.filter(ImageFilter.GaussianBlur(radius=feather))
    return {
        "clean": clean,
        "inner": inner,
        "outer": outer,
        "edge_band": edge_band,
        "inner_boundary": inner_boundary,
        "outer_boundary": outer_boundary,
        "soft_alpha": soft_alpha,
        "edge_alpha": edge_alpha,
        "feather": feather,
    }


def _mask_active(mask, threshold=8):
    return np.asarray(mask.convert("L"), dtype=np.float32) > threshold


def _rgb_stats(arr, active):
    if not np.any(active):
        return None, None
    pixels = arr[active]
    return pixels.mean(axis=0), pixels.std(axis=0) + 1e-6


def _luma(arr):
    return arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722


def _edge_contrast_score(source_arr, result_arr, inner_active, outer_active):
    if not np.any(inner_active) or not np.any(outer_active):
        return 1.0
    person_luma = float(_luma(result_arr)[inner_active].mean())
    bg_luma = float(_luma(source_arr)[outer_active].mean())
    return round(float(np.clip(1.0 - abs(person_luma - bg_luma) / 96.0, 0.0, 1.0)), 4)


def _boundary_laplacian_score(image, edge_active):
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if not np.any(edge_active):
        return 0.0
    if cv2 is not None:
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    else:
        shifted = (
            np.roll(gray, 1, axis=0)
            + np.roll(gray, -1, axis=0)
            + np.roll(gray, 1, axis=1)
            + np.roll(gray, -1, axis=1)
            - 4 * gray
        )
        lap = np.abs(shifted)
    return round(float(np.clip(lap[edge_active].mean() / 48.0, 0.0, 1.0)), 4)


def _apply_edge_blur(result_arr, blurred_arr, edge_alpha):
    blur_radius = edge_blur_radius()
    if blur_radius <= 0:
        return result_arr
    alpha = np.asarray(edge_alpha.convert("L"), dtype=np.float32) / 255.0
    alpha = np.expand_dims(np.clip(alpha * 0.12, 0.0, 0.12), axis=2)
    return result_arr * (1.0 - alpha) + blurred_arr * alpha


def _apply_boundary_color_match(source_arr, result_arr, masks):
    strength = edge_color_strength()
    if strength <= 0:
        return result_arr
    # Only touch the outside boundary. Recoloring the inner boundary makes the
    # generated person lose detail and visually sink into the background.
    inner_active = _mask_active(masks["outer_boundary"], threshold=8)
    outer_active = _mask_active(masks["outer_boundary"], threshold=8)
    bg_mean, bg_std = _rgb_stats(source_arr, outer_active)
    edge_mean, edge_std = _rgb_stats(result_arr, inner_active)
    if bg_mean is None or edge_mean is None:
        return result_arr
    corrected = (result_arr - edge_mean.reshape(1, 1, 3)) * (bg_std / edge_std).reshape(1, 1, 3) + bg_mean.reshape(1, 1, 3)
    corrected = np.clip(corrected, 0, 255)
    alpha = np.asarray(masks["outer_boundary"].filter(ImageFilter.GaussianBlur(radius=masks["feather"])), dtype=np.float32) / 255.0
    alpha = np.expand_dims(np.clip(alpha * strength, 0.0, strength), axis=2)
    return result_arr * (1.0 - alpha) + corrected * alpha


def _apply_outer_source_cleanup(source_arr, result_arr, masks):
    outer_alpha = np.asarray(masks["outer_boundary"].filter(ImageFilter.GaussianBlur(radius=masks["feather"])), dtype=np.float32) / 255.0
    outer_alpha = np.expand_dims(np.clip(outer_alpha * 0.10, 0.0, 0.10), axis=2)
    return result_arr * (1.0 - outer_alpha) + source_arr * outer_alpha


def _try_poisson_blend(source, result, mask):
    if not poisson_enabled() or cv2 is None:
        return result, False
    bbox = mask.getbbox()
    if bbox is None:
        return result, False
    x1, y1, x2, y2 = bbox
    center = (int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2)))
    try:
        src = cv2.cvtColor(np.asarray(source.convert("RGB")), cv2.COLOR_RGB2BGR)
        dst = cv2.cvtColor(np.asarray(result.convert("RGB")), cv2.COLOR_RGB2BGR)
        m = np.asarray(mask.convert("L"), dtype=np.uint8)
        blended = cv2.seamlessClone(dst, src, m, center, cv2.MIXED_CLONE)
        return Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB), mode="RGB"), True
    except Exception:
        return result, False


def _save_debug(debug_context, source, before, after, masks):
    if not debug_context:
        return ""
    debug_index = debug_context.get("debug_index")
    if debug_index is not None and debug_index >= PATCH_DEBUG_MAX_ITEMS:
        return ""
    record = debug_context.get("record")
    if record is None:
        return ""
    variant = debug_context.get("variant", "variant")
    seed = debug_context.get("seed", 0)
    debug_dir = EDGE_DEBUG_DIR / record.split / record.bucket
    debug_dir.mkdir(parents=True, exist_ok=True)

    overlay = before.copy()
    draw = ImageDraw.Draw(overlay)
    bbox = masks["clean"].getbbox()
    if bbox:
        draw.rectangle(bbox, outline="red", width=2)

    band_vis = ImageOps.colorize(masks["edge_band"].convert("L"), black="black", white="cyan")
    alpha_vis = masks["soft_alpha"].convert("RGB")
    panels = [
        ("raw_patch", before.convert("RGB")),
        ("hard_mask", overlay.convert("RGB")),
        ("soft_alpha", alpha_vis),
        ("edge_band", band_vis),
        ("after", after.convert("RGB")),
    ]
    w = max(panel.width for _, panel in panels)
    h = max(panel.height for _, panel in panels)
    label_h = 24
    strip = Image.new("RGB", (w * len(panels), h + label_h), "white")
    strip_draw = ImageDraw.Draw(strip)
    for index, (label, panel) in enumerate(panels):
        x = index * w
        strip_draw.text((x + 8, 6), label, fill="black")
        strip.paste(panel.resize((w, h)), (x, label_h))
    out_path = debug_dir / f"{record.path.stem}_edge_harmonization_{seed}_{variant}.png"
    strip.save(out_path)
    return str(out_path)


def harmonize_pedestrian_edge(source_image, generated_image, person_mask, insert_bbox=None, yolo_mask=None, debug_context=None):
    if not edge_enabled() or person_mask is None:
        return EdgeHarmonizationResult(
            generated_image,
            person_mask,
            {
                "edge_harmonization_applied": False,
                "contact_shadow_applied": False,
                "mask_feather_radius": "",
                "edge_contrast_score": "",
                "boundary_laplacian_score": "",
                "edge_harmonization_debug_path": "",
            },
        )
    source = source_image.convert("RGB")
    before = generated_image.convert("RGB")
    masks = create_edge_masks(yolo_mask or person_mask)
    source_arr = np.asarray(source, dtype=np.float32)
    result_arr = np.asarray(before, dtype=np.float32)
    blurred_arr = np.asarray(before.filter(ImageFilter.GaussianBlur(radius=edge_blur_radius())), dtype=np.float32)

    result_arr = _apply_edge_blur(result_arr, blurred_arr, masks["edge_alpha"])
    result_arr = _apply_boundary_color_match(source_arr, result_arr, masks)
    result_arr = _apply_outer_source_cleanup(source_arr, result_arr, masks)

    after = Image.fromarray(np.clip(result_arr, 0, 255).astype(np.uint8), mode="RGB")
    after, poisson_used = _try_poisson_blend(source, after, masks["clean"])

    inner_active = _mask_active(masks["inner_boundary"], threshold=8)
    outer_active = _mask_active(masks["outer_boundary"], threshold=8)
    edge_active = _mask_active(masks["edge_band"], threshold=8)
    debug_path = _save_debug(debug_context, source, before, after, masks)
    metadata = {
        "edge_harmonization_applied": True,
        "contact_shadow_applied": False,
        "poisson_blend_used": bool(poisson_used),
        "mask_feather_radius": masks["feather"],
        "edge_contrast_score": _edge_contrast_score(source_arr, np.asarray(after, dtype=np.float32), inner_active, outer_active),
        "boundary_laplacian_score": _boundary_laplacian_score(after, edge_active),
        "edge_harmonization_debug_path": debug_path,
    }
    return EdgeHarmonizationResult(after, masks["clean"], metadata)
