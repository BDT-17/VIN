"""SD3.5 CityPersons augmentation: generation and compositing pipeline."""

import csv
import gc
import json
from datetime import datetime
import math
import numpy as np
import os
import random
import re
import statistics
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps, ImageDraw, ImageFilter, ImageChops

try:
    import cv2
except ImportError:
    cv2 = None

from sd35_config import *
from sd35_data import build_generation_prompt, build_variant_negative_prompt
from sd35_utils import *
from sd35_evaluation import *

def add_contact_shadow(image, insert_bbox, variant):
    if not CONTACT_SHADOW_ENABLED:
        return image
    width, height = image.size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for bbox in guide_bboxes_for_variant(insert_bbox, variant):
        x1, y1, x2, y2 = bbox
        bw = max(4, x2 - x1)
        bh = max(8, y2 - y1)
        ground_y = y2
        scale = perspective_scale_for_ground_y(ground_y, resolution=height)
        opacity = int(CONTACT_SHADOW_OPACITY_FAR + scale * (CONTACT_SHADOW_OPACITY_NEAR - CONTACT_SHADOW_OPACITY_FAR))
        shadow_w = max(8, int(bw * (0.70 + 0.30 * scale)))
        shadow_h = max(3, int(bh * 0.055))
        blur = max(2, int(bh * 0.025))
        cx = int((x1 + x2) / 2 + bw * 0.06)
        cy = int(ground_y - shadow_h * 0.35)
        shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        shadow_draw.ellipse((cx - shadow_w // 2, cy - shadow_h // 2, cx + shadow_w // 2, cy + shadow_h // 2), fill=(0, 0, 0, opacity))
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur))
        overlay = Image.alpha_composite(overlay, shadow)
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def save_inpaint_debug_strip(record, variant, seed, source, mask_image, guided_source, generated, final, insert_bbox, debug_index=None):
    if not SAVE_PATCH_DEBUG:
        return ""
    if debug_index is not None and debug_index >= PATCH_DEBUG_MAX_ITEMS:
        return ""
    debug_dir = PATCH_DEBUG_DIR / record.split / record.bucket
    debug_dir.mkdir(parents=True, exist_ok=True)
    crop_bbox = expand_bbox_with_context(insert_bbox, resolution=source.width)
    panels = [
        ("source", source.crop(crop_bbox).convert("RGB")),
        ("mask", mask_image.crop(crop_bbox).convert("RGB")),
        ("guided", guided_source.crop(crop_bbox).convert("RGB")),
        ("generated", generated.crop(crop_bbox).convert("RGB")),
        ("final", final.crop(crop_bbox).convert("RGB")),
    ]
    panel_w = max(panel.width for _, panel in panels)
    panel_h = max(panel.height for _, panel in panels)
    label_h = 24
    canvas = Image.new("RGB", (panel_w * len(panels), panel_h + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        image = image.resize((panel_w, panel_h))
        x = index * panel_w
        draw.text((x + 8, 6), label, fill=(0, 0, 0))
        canvas.paste(image, (x, label_h))
    safe_variant = variant.replace("/", "_")
    debug_path = debug_dir / f"{record.path.stem}_inpaint_debug_{seed}_{safe_variant}.png"
    canvas.save(debug_path)
    return str(debug_path)


def context_crop_bbox_for_insert(insert_bbox, image_size, expand=CONTEXT_CROP_EXPAND, min_size=CONTEXT_CROP_MIN_SIZE):
    image_w, image_h = image_size
    x1, y1, x2, y2 = insert_bbox
    bw = x2 - x1
    bh = y2 - y1
    crop_size = int(max(min_size, bw * expand, bh * expand))
    crop_size = min(crop_size, image_w, image_h)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    left = int(round(cx - crop_size / 2))
    top = int(round(cy - crop_size / 2))
    left = max(0, min(image_w - crop_size, left))
    top = max(0, min(image_h - crop_size, top))
    return (left, top, left + crop_size, top + crop_size)


def map_bbox_to_resized_crop(bbox, crop_bbox, output_size=RESOLUTION):
    x1, y1, x2, y2 = bbox
    cx1, cy1, cx2, cy2 = crop_bbox
    scale_x = output_size / max(1, cx2 - cx1)
    scale_y = output_size / max(1, cy2 - cy1)
    return (
        int(round((x1 - cx1) * scale_x)),
        int(round((y1 - cy1) * scale_y)),
        int(round((x2 - cx1) * scale_x)),
        int(round((y2 - cy1) * scale_y)),
    )


def mask_stats_rgb(image, mask, threshold=0.18):
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    mask_arr = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    active = mask_arr > threshold
    if not np.any(active):
        return None, None
    pixels = arr[active]
    return pixels.mean(axis=0), pixels.std(axis=0) + 1e-6


def local_source_context_mask(mask, pad=COLOR_MATCH_CONTEXT_PAD):
    bbox = mask.getbbox()
    if bbox is None:
        return Image.new("L", mask.size, 0)
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(mask.size[0], x2 + pad)
    y2 = min(mask.size[1], y2 + pad)
    context = Image.new("L", mask.size, 0)
    draw = ImageDraw.Draw(context)
    draw.rectangle((x1, y1, x2, y2), fill=255)
    context = ImageChops.subtract(context, mask.filter(ImageFilter.GaussianBlur(radius=2)))
    return context


def horizontal_context_mask(mask, band_ratio=EDGE_HORIZON_BAND_RATIO, pad=None):
    bbox = mask.getbbox()
    if bbox is None:
        return Image.new("L", mask.size, 0)
    width, height = mask.size
    x1, y1, x2, y2 = bbox
    pad = EDGE_BG_CONTEXT_PAD if pad is None else min(max(int(pad), 10), 20)
    band = max(8, min(20, int(height * band_ratio)))
    x1 = max(0, x1 - pad)
    x2 = min(width, x2 + pad)
    y1 = max(0, y1 - band)
    y2 = min(height, y2 + max(pad, band // 2))
    context = Image.new("L", mask.size, 0)
    draw = ImageDraw.Draw(context)
    draw.rectangle((x1, y1, x2, y2), fill=255)
    context = ImageChops.subtract(context, mask.filter(ImageFilter.GaussianBlur(radius=2)))
    return context


def blended_context_mask(mask, pad=COLOR_MATCH_CONTEXT_PAD):
    local = local_source_context_mask(mask, pad=pad)
    horizontal = horizontal_context_mask(mask, pad=pad)
    return ImageChops.lighter(local, horizontal)


def color_match_person_crop(source_crop, person_rgb, person_mask):
    if not COLOR_MATCH_PERSON_TO_SCENE:
        return person_rgb
    src_mean, src_std = mask_stats_rgb(source_crop, blended_context_mask(person_mask), threshold=0.12)
    gen_mean, gen_std = mask_stats_rgb(person_rgb, person_mask, threshold=0.18)
    if src_mean is None or gen_mean is None:
        return person_rgb
    arr = np.asarray(person_rgb.convert("RGB"), dtype=np.float32)
    corrected = (arr - gen_mean) * (src_std / gen_std) + src_mean
    corrected = np.clip(corrected, 0, 255)
    blended = arr * (1.0 - COLOR_MATCH_STRENGTH) + corrected * COLOR_MATCH_STRENGTH
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def high_frequency_std(image, mask=None, threshold=0.12):
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    low = np.asarray(image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=1.0)), dtype=np.float32)
    high = arr - low
    if mask is not None:
        mask_arr = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
        active = mask_arr > threshold
        if np.any(active):
            high = high[active]
    return float(np.std(high)) + 1e-6


def match_person_texture_to_scene(source_crop, person_rgb, person_mask):
    if not TEXTURE_MATCH_PERSON_TO_SCENE:
        return person_rgb
    context_mask = local_source_context_mask(person_mask, pad=TEXTURE_MATCH_CONTEXT_PAD)
    src_hf = high_frequency_std(source_crop, context_mask, threshold=0.10)
    gen_hf = high_frequency_std(person_rgb, person_mask, threshold=0.18)
    if gen_hf <= src_hf * 1.02 and gen_hf <= MAX_GENERATED_PERSON_SHARPNESS_STD:
        return person_rgb
    sharp_ratio = max(1.0, min(3.0, max(gen_hf / max(src_hf, 1e-6), gen_hf / max(MAX_GENERATED_PERSON_SHARPNESS_STD, 1e-6))))
    blur_radius = TEXTURE_MATCH_MIN_BLUR + (sharp_ratio - 1.0) / 2.0 * (TEXTURE_MATCH_MAX_BLUR - TEXTURE_MATCH_MIN_BLUR)
    softened = person_rgb.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    arr = np.asarray(person_rgb.convert("RGB"), dtype=np.float32)
    soft = np.asarray(softened.convert("RGB"), dtype=np.float32)
    alpha_3 = foreground_harmonization_alpha(person_mask, TEXTURE_MATCH_STRENGTH, core_alpha=0.03)
    matched = arr * (1.0 - alpha_3) + soft * alpha_3
    return Image.fromarray(np.clip(matched, 0, 255).astype(np.uint8), mode="RGB")


def luminance_values(image, mask, threshold=0.12):
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    mask_arr = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    active = mask_arr > threshold
    if not np.any(active):
        return None
    luma = arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722
    return luma[active]


def local_ring_mask(person_mask, pad=None):
    pad = max(COLOR_MATCH_CONTEXT_PAD, TEXTURE_MATCH_CONTEXT_PAD) if pad is None else pad
    return blended_context_mask(person_mask, pad=pad)


def luminance_mean_std(image, mask, threshold=0.12):
    values = luminance_values(image, mask, threshold=threshold)
    if values is None:
        return None, None
    return float(np.mean(values)), float(np.std(values)) + 1e-6


def saturation_values(image, mask, threshold=0.12):
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    mask_arr = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    active = mask_arr > threshold
    if not np.any(active):
        return None
    max_c = arr.max(axis=2)
    min_c = arr.min(axis=2)
    sat = (max_c - min_c) / np.maximum(max_c, 1e-6)
    return sat[active]


def foreground_harmonization_alpha(person_mask, strength, core_alpha=None, edge_erode=None):
    mask_l = person_mask.convert("L")
    mask_arr = np.asarray(mask_l, dtype=np.float32) / 255.0
    if not np.any(mask_arr > 0.01):
        return np.zeros((*mask_arr.shape, 1), dtype=np.float32)
    edge_erode = FOREGROUND_HARMONIZATION_EDGE_ERODE if edge_erode is None else edge_erode
    core_alpha = FOREGROUND_HARMONIZATION_CORE_ALPHA if core_alpha is None else core_alpha
    hard = mask_l.point(lambda p: 255 if p >= PERSON_PASTE_HARD_THRESHOLD else 0)
    if edge_erode > 0:
        core = hard.filter(ImageFilter.MinFilter(edge_erode * 2 + 1))
    else:
        core = hard
    core_arr = np.asarray(core, dtype=np.float32) / 255.0
    edge_arr = np.clip(mask_arr - core_arr, 0.0, 1.0)
    alpha = edge_arr * strength + core_arr * min(strength, core_alpha)
    return np.expand_dims(np.clip(alpha, 0.0, 1.0), axis=2)


def local_color_transfer(source_crop, person_rgb, person_mask, strength=0.74):
    context_mask = local_ring_mask(person_mask)
    src_mean, src_std = mask_stats_rgb(source_crop, context_mask, threshold=0.10)
    gen_mean, gen_std = mask_stats_rgb(person_rgb, person_mask, threshold=0.18)
    if src_mean is None or gen_mean is None:
        return person_rgb
    arr = np.asarray(person_rgb.convert("RGB"), dtype=np.float32)
    alpha_3 = foreground_harmonization_alpha(person_mask, strength)
    ratio = np.clip(src_std / gen_std, 0.82, 1.18)
    corrected = (arr - gen_mean.reshape(1, 1, 3)) * ratio.reshape(1, 1, 3) + src_mean.reshape(1, 1, 3)
    matched = arr * (1.0 - alpha_3) + corrected * alpha_3
    return Image.fromarray(np.clip(matched, 0, 255).astype(np.uint8), mode="RGB")


def match_local_brightness(source_crop, person_rgb, person_mask, strength=0.72):
    context_mask = local_ring_mask(person_mask)
    src_mean, _ = luminance_mean_std(source_crop, context_mask, threshold=0.10)
    gen_mean, _ = luminance_mean_std(person_rgb, person_mask, threshold=0.18)
    if src_mean is None or gen_mean is None:
        return person_rgb
    shift = np.clip(src_mean - gen_mean, -18.0, 18.0)
    arr = np.asarray(person_rgb.convert("RGB"), dtype=np.float32)
    alpha_3 = foreground_harmonization_alpha(person_mask, strength)
    matched = arr + shift * alpha_3
    return Image.fromarray(np.clip(matched, 0, 255).astype(np.uint8), mode="RGB")


def match_local_contrast(source_crop, person_rgb, person_mask, strength=0.56):
    context_mask = local_ring_mask(person_mask)
    _, src_std = luminance_mean_std(source_crop, context_mask, threshold=0.10)
    gen_mean, gen_std = luminance_mean_std(person_rgb, person_mask, threshold=0.18)
    if src_std is None or gen_mean is None:
        return person_rgb
    ratio = np.clip(src_std / gen_std, 0.78, 1.16)
    arr = np.asarray(person_rgb.convert("RGB"), dtype=np.float32)
    alpha_3 = foreground_harmonization_alpha(person_mask, strength)
    matched_contrast = (arr - gen_mean) * ratio + gen_mean
    matched = arr * (1.0 - alpha_3) + matched_contrast * alpha_3
    return Image.fromarray(np.clip(matched, 0, 255).astype(np.uint8), mode="RGB")


def match_local_saturation(source_crop, person_rgb, person_mask, strength=0.45):
    context_mask = local_ring_mask(person_mask)
    src_sat = saturation_values(source_crop, context_mask, threshold=0.10)
    gen_sat = saturation_values(person_rgb, person_mask, threshold=0.18)
    if src_sat is None or gen_sat is None:
        return person_rgb
    src_mean = float(np.mean(src_sat))
    gen_mean = float(np.mean(gen_sat)) + 1e-6
    ratio = np.clip(src_mean / gen_mean, 0.72, 1.12)
    arr = np.asarray(person_rgb.convert("RGB"), dtype=np.float32)
    gray = np.sum(arr * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32).reshape(1, 1, 3), axis=2, keepdims=True)
    sat_matched = gray + (arr - gray) * ratio
    alpha_3 = foreground_harmonization_alpha(person_mask, strength)
    matched = arr * (1.0 - alpha_3) + sat_matched * alpha_3
    return Image.fromarray(np.clip(matched, 0, 255).astype(np.uint8), mode="RGB")


def add_sensor_noise(source_crop, person_rgb, person_mask, strength=0.95):
    context_mask = local_ring_mask(person_mask)
    src_hf = high_frequency_std(source_crop, context_mask, threshold=0.10)
    gen_hf = high_frequency_std(person_rgb, person_mask, threshold=0.18)
    if gen_hf >= src_hf * 0.92:
        return person_rgb
    noise_std = float(np.clip((src_hf - gen_hf) * 0.38, 0.0, 3.0))
    if noise_std <= 0.05:
        return person_rgb
    arr = np.asarray(person_rgb.convert("RGB"), dtype=np.float32)
    alpha_3 = foreground_harmonization_alpha(person_mask, strength, core_alpha=0.04)
    seed = ((person_rgb.size[0] * 73856093) ^ (person_rgb.size[1] * 19349663)) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_std, arr.shape).astype(np.float32)
    matched = arr + noise * alpha_3
    return Image.fromarray(np.clip(matched, 0, 255).astype(np.uint8), mode="RGB")


def gaussian_blur_person(person_rgb, person_mask, sigma=0.60, strength=0.55):
    blurred = person_rgb.filter(ImageFilter.GaussianBlur(radius=sigma))
    arr = np.asarray(person_rgb.convert("RGB"), dtype=np.float32)
    soft = np.asarray(blurred.convert("RGB"), dtype=np.float32)
    alpha_3 = foreground_harmonization_alpha(person_mask, strength, core_alpha=0.03)
    matched = arr * (1.0 - alpha_3) + soft * alpha_3
    return Image.fromarray(np.clip(matched, 0, 255).astype(np.uint8), mode="RGB")


def apply_subtle_scene_tone_filter(source_crop, person_rgb, person_mask):
    if not PERSON_TONE_FILTER_ENABLED:
        return person_rgb
    context_mask = local_ring_mask(person_mask)
    src_mean, _ = mask_stats_rgb(source_crop, context_mask, threshold=0.10)
    gen_mean, _ = mask_stats_rgb(person_rgb, person_mask, threshold=0.18)
    if src_mean is None or gen_mean is None:
        return person_rgb

    src_cast = src_mean - float(np.mean(src_mean))
    gen_cast = gen_mean - float(np.mean(gen_mean))
    color_shift = np.clip(
        (src_cast - gen_cast) * 0.45,
        -PERSON_TONE_FILTER_MAX_COLOR_SHIFT,
        PERSON_TONE_FILTER_MAX_COLOR_SHIFT,
    )

    src_luma, _ = luminance_mean_std(source_crop, context_mask, threshold=0.10)
    gen_luma, _ = luminance_mean_std(person_rgb, person_mask, threshold=0.18)
    brightness_shift = 0.0
    if src_luma is not None and gen_luma is not None:
        brightness_shift = float(np.clip(
            (src_luma - gen_luma) * 0.18,
            -PERSON_TONE_FILTER_MAX_BRIGHTNESS_SHIFT,
            PERSON_TONE_FILTER_MAX_BRIGHTNESS_SHIFT,
        ))

    arr = np.asarray(person_rgb.convert("RGB"), dtype=np.float32)
    filtered = arr + color_shift.reshape(1, 1, 3) + brightness_shift
    alpha_3 = foreground_harmonization_alpha(
        person_mask,
        PERSON_TONE_FILTER_STRENGTH,
        core_alpha=PERSON_TONE_FILTER_CORE_STRENGTH,
        edge_erode=1,
    )
    matched = arr * (1.0 - alpha_3) + filtered * alpha_3
    return Image.fromarray(np.clip(matched, 0, 255).astype(np.uint8), mode="RGB")


def harmonize_person_to_scene(source_crop, person_rgb, person_mask):
    person_rgb = color_match_person_crop(source_crop, person_rgb, person_mask)
    person_rgb = match_person_texture_to_scene(source_crop, person_rgb, person_mask)
    # Activate existing helpers for fuller edge harmonization.
    # All use foreground_harmonization_alpha: strong at edge, weak in core.
    person_rgb = local_color_transfer(source_crop, person_rgb, person_mask, strength=0.45)
    person_rgb = match_local_brightness(source_crop, person_rgb, person_mask, strength=0.50)
    person_rgb = match_local_contrast(source_crop, person_rgb, person_mask, strength=0.35)
    person_rgb = match_local_saturation(source_crop, person_rgb, person_mask, strength=0.30)
    person_rgb = add_sensor_noise(source_crop, person_rgb, person_mask, strength=0.80)
    person_rgb = apply_subtle_scene_tone_filter(source_crop, person_rgb, person_mask)
    person_rgb = neutralize_person_edge_halo(source_crop, person_rgb, person_mask)
    person_rgb = soften_dark_person_edge(source_crop, person_rgb, person_mask)
    return person_rgb


def match_person_appearance_to_scene(source_crop, person_rgb, person_mask):
    context_mask = blended_context_mask(person_mask, pad=max(COLOR_MATCH_CONTEXT_PAD, TEXTURE_MATCH_CONTEXT_PAD))
    src_mean, _ = mask_stats_rgb(source_crop, context_mask, threshold=0.10)
    gen_mean, _ = mask_stats_rgb(person_rgb, person_mask, threshold=0.18)
    if src_mean is None or gen_mean is None:
        return person_rgb

    arr = np.asarray(person_rgb.convert("RGB"), dtype=np.float32)
    alpha = np.asarray(person_mask.convert("L"), dtype=np.float32) / 255.0
    alpha_3 = np.expand_dims(np.clip(alpha, 0.0, 1.0), axis=2)

    # Color temperature: match the local red-vs-blue cast without repainting clothing colors.
    src_temp = float(src_mean[0] - src_mean[2])
    gen_temp = float(gen_mean[0] - gen_mean[2])
    temp_shift = np.clip((src_temp - gen_temp) * 0.28, -10.0, 10.0)
    temp_matched = arr.copy()
    temp_matched[..., 0] += temp_shift
    temp_matched[..., 2] -= temp_shift
    arr = arr * (1.0 - alpha_3 * 0.55) + temp_matched * (alpha_3 * 0.55)

    # Contrast: match local luminance spread so the person does not look too crisp or flat.
    src_luma = luminance_values(source_crop, context_mask, threshold=0.10)
    gen_luma = luminance_values(person_rgb, person_mask, threshold=0.18)
    if src_luma is not None and gen_luma is not None:
        src_std = float(np.std(src_luma)) + 1e-6
        gen_std = float(np.std(gen_luma)) + 1e-6
        ratio = np.clip(src_std / gen_std, 0.78, 1.16)
        contrast_matched = (arr - gen_mean.reshape(1, 1, 3)) * ratio + gen_mean.reshape(1, 1, 3)
        arr = arr * (1.0 - alpha_3 * 0.38) + contrast_matched * (alpha_3 * 0.38)

    # Noise/grain: add only when the generated person is cleaner than the surrounding crop.
    src_hf = high_frequency_std(source_crop, context_mask, threshold=0.10)
    gen_hf = high_frequency_std(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB"), person_mask, threshold=0.18)
    if gen_hf < src_hf * 0.92:
        noise_std = float(np.clip((src_hf - gen_hf) * 0.32, 0.0, 2.2))
        if noise_std > 0.05:
            seed = ((person_rgb.size[0] * 73856093) ^ (person_rgb.size[1] * 19349663)) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            noise = rng.normal(0.0, noise_std, arr.shape).astype(np.float32)
            arr = arr + noise * alpha_3 * 0.85

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")



def soften_dark_person_edge(source_crop, person_rgb, person_mask):
    mask_l = person_mask.convert("L")
    mask_arr = np.asarray(mask_l, dtype=np.float32) / 255.0
    if not np.any(mask_arr > 0.02):
        return person_rgb

    # Only touch transparent/soft edge pixels. Do not process the hard inner
    # silhouette, otherwise background tone can make the person look thinner.
    hard_threshold = PERSON_PASTE_HARD_THRESHOLD / 255.0
    edge = ((mask_arr > EDGE_HALO_MIN_ALPHA) & (mask_arr < hard_threshold)).astype(np.float32)
    if not np.any(edge > 0.02):
        return person_rgb

    arr = np.asarray(person_rgb.convert("RGB"), dtype=np.float32)
    blurred_person = np.asarray(person_rgb.convert("RGB").filter(ImageFilter.GaussianBlur(radius=0.55)), dtype=np.float32)
    background_tone = blended_background_mean_map(source_crop, person_mask)
    target = blurred_person * 0.72 + background_tone * 0.28

    luma = np.sum(arr * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32).reshape(1, 1, 3), axis=2)
    bg_luma_map = np.sum(background_tone * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32).reshape(1, 1, 3), axis=2)
    dark_edge_boost = np.clip((bg_luma_map - luma) / 95.0, 0.0, 0.35)
    alpha = np.clip(edge * (0.22 + dark_edge_boost), 0.0, 0.42)
    alpha_3 = np.expand_dims(alpha, axis=2)
    softened = arr * (1.0 - alpha_3) + target * alpha_3
    return Image.fromarray(np.clip(softened, 0, 255).astype(np.uint8), mode="RGB")


def neutralize_person_edge_halo(source_crop, person_rgb, person_mask):
    if not EDGE_HALO_NEUTRALIZE:
        return person_rgb
    mask_l = person_mask.convert("L")
    mask_arr = np.asarray(mask_l, dtype=np.float32) / 255.0
    filter_size = max(3, int(EDGE_HALO_WIDTH) * 2 + 1)
    dilated = np.asarray(mask_l.filter(ImageFilter.MaxFilter(filter_size)), dtype=np.float32) / 255.0
    eroded = np.asarray(mask_l.filter(ImageFilter.MinFilter(filter_size)), dtype=np.float32) / 255.0
    ring = np.clip(dilated - eroded, 0.0, 1.0)
    soft_edge = ((mask_arr >= EDGE_HALO_MIN_ALPHA) & (mask_arr <= EDGE_HALO_MAX_ALPHA)).astype(np.float32)
    edge_alpha = np.clip(np.maximum(ring, soft_edge) * EDGE_HALO_COLOR_MATCH_STRENGTH, 0.0, 1.0)
    edge_active = edge_alpha > 0.02
    if not np.any(edge_active):
        return person_rgb

    context_mask = blended_context_mask(person_mask, pad=COLOR_MATCH_CONTEXT_PAD)
    src_mean, src_std = mask_stats_rgb(source_crop, context_mask, threshold=0.10)
    if src_mean is None:
        return person_rgb

    arr = np.asarray(person_rgb.convert("RGB"), dtype=np.float32)
    edge_pixels = arr[edge_active]
    edge_mean = edge_pixels.mean(axis=0)
    edge_std = edge_pixels.std(axis=0) + 1e-6
    corrected = (arr - edge_mean) * (src_std / edge_std) + src_mean
    corrected = np.clip(corrected, 0, 255)
    edge_alpha_3 = np.expand_dims(edge_alpha, axis=2)
    matched = arr * (1.0 - edge_alpha_3) + corrected * edge_alpha_3
    return Image.fromarray(np.clip(matched, 0, 255).astype(np.uint8), mode="RGB")


def tight_person_edge_alpha(person_mask):
    mask_l = person_mask.convert("L")
    mask_arr = np.asarray(mask_l, dtype=np.float32) / 255.0
    hard = mask_l.point(lambda p: 255 if p >= PERSON_PASTE_HARD_THRESHOLD else 0)
    filter_size = max(3, int(EDGE_HALO_WIDTH) * 2 + 1)
    dilated = np.asarray(hard.filter(ImageFilter.MaxFilter(filter_size)), dtype=np.float32) / 255.0
    inner_filter_size = max(3, 2 * max(1, int(EDGE_HALO_WIDTH) + 1) + 1)
    eroded = np.asarray(hard.filter(ImageFilter.MinFilter(inner_filter_size)), dtype=np.float32) / 255.0
    hard_arr = np.asarray(hard, dtype=np.float32) / 255.0

    outside_ring = np.clip(dilated - hard_arr, 0.0, 1.0)
    inner_ring = np.clip(hard_arr - eroded, 0.0, 1.0)
    soft_transition = ((mask_arr >= EDGE_HALO_MIN_ALPHA) & (mask_arr < PERSON_PASTE_HARD_THRESHOLD / 255.0)).astype(np.float32)
    edge = np.maximum(outside_ring, soft_transition) * (1.0 - hard_arr)
    edge = np.maximum(edge, inner_ring * 0.72)
    return np.clip(edge * EDGE_HALO_COLOR_MATCH_STRENGTH, 0.0, 1.0)


def local_background_mean_map(image, person_mask, radius=EDGE_LOCAL_BG_RADIUS, exclude_threshold=0.04):
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    mask_arr = np.asarray(person_mask.convert("L"), dtype=np.float32) / 255.0
    bg_weight = (mask_arr <= exclude_threshold).astype(np.float32)
    if not np.any(bg_weight > 0):
        fallback = arr.reshape(-1, 3).mean(axis=0)
        return np.zeros_like(arr) + fallback.reshape(1, 1, 3)

    bg_weight_u8 = np.clip(bg_weight * 255.0, 0, 255).astype(np.uint8)
    denom = np.asarray(
        Image.fromarray(bg_weight_u8).filter(ImageFilter.GaussianBlur(radius=radius)),
        dtype=np.float32,
    ) / 255.0
    local_mean = np.zeros_like(arr)
    for channel in range(3):
        weighted = np.clip(arr[..., channel] * bg_weight, 0, 255).astype(np.uint8)
        numer = np.asarray(
            Image.fromarray(weighted).filter(ImageFilter.GaussianBlur(radius=radius)),
            dtype=np.float32,
        )
        local_mean[..., channel] = numer / np.maximum(denom, 1e-6)

    fallback_pixels = arr[bg_weight > 0]
    fallback = fallback_pixels.mean(axis=0) if fallback_pixels.size else arr.reshape(-1, 3).mean(axis=0)
    weak = denom < 0.015
    if np.any(weak):
        local_mean[weak] = fallback
    return local_mean


def bbox_background_fallback_mean(image, person_mask, exclude_threshold=0.04):
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    mask_arr = np.asarray(person_mask.convert("L"), dtype=np.float32) / 255.0
    bbox = person_mask.getbbox()
    if bbox is None:
        fallback = arr.reshape(-1, 3).mean(axis=0)
        return np.zeros_like(arr) + fallback.reshape(1, 1, 3)
    x1, y1, x2, y2 = bbox
    pad = int(EDGE_BG_CONTEXT_PAD)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(arr.shape[1], x2 + pad)
    y2 = min(arr.shape[0], y2 + pad)
    crop = arr[y1:y2, x1:x2]
    crop_mask = mask_arr[y1:y2, x1:x2]
    bg_pixels = crop[crop_mask <= exclude_threshold]
    if bg_pixels.size == 0:
        bg_pixels = crop.reshape(-1, 3)
    fallback = bg_pixels.mean(axis=0) if bg_pixels.size else arr.reshape(-1, 3).mean(axis=0)
    return np.zeros_like(arr) + fallback.reshape(1, 1, 3)


def horizontal_background_mean_map(image, person_mask, band_ratio=EDGE_HORIZON_BAND_RATIO, exclude_threshold=0.04):
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    mask_arr = np.asarray(person_mask.convert("L"), dtype=np.float32) / 255.0
    h, w = mask_arr.shape
    bbox = person_mask.getbbox()
    bg_weight = (mask_arr <= exclude_threshold).astype(np.float32)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        pad = int(EDGE_BG_CONTEXT_PAD)
        band = max(8, min(20, int(h * band_ratio)))
        band_mask = np.zeros_like(bg_weight)
        x1 = max(0, x1 - pad)
        x2 = min(w, x2 + pad)
        y1 = max(0, y1 - band)
        y2 = min(h, y2 + max(pad, band // 2))
        band_mask[y1:y2, x1:x2] = 1.0
        bg_weight *= band_mask
    if not np.any(bg_weight > 0):
        return bbox_background_fallback_mean(image, person_mask, exclude_threshold=exclude_threshold)
    fallback = arr[bg_weight > 0].mean(axis=0)
    row_weight = bg_weight.sum(axis=1, keepdims=True)
    row_sum = (arr * bg_weight[..., None]).sum(axis=1)
    row_mean = row_sum / np.maximum(row_weight, 1e-6)
    row_mean[row_weight[:, 0] <= 0] = fallback
    horizon = np.repeat(row_mean[:, None, :], w, axis=1)
    return horizon


def blended_background_mean_map(image, person_mask):
    return horizontal_background_mean_map(image, person_mask)

def match_pasted_edge_to_composite_mean(result_crop, person_mask):
    if not EDGE_HALO_NEUTRALIZE:
        return result_crop
    edge_alpha = tight_person_edge_alpha(person_mask)
    edge_active = edge_alpha > 0.02
    if not np.any(edge_active):
        return result_crop

    arr = np.asarray(result_crop.convert("RGB"), dtype=np.float32)
    blurred = np.asarray(result_crop.convert("RGB").filter(ImageFilter.GaussianBlur(radius=0.45)), dtype=np.float32)
    local_mean = blended_background_mean_map(result_crop, person_mask)
    softened = arr * 0.38 + blurred * 0.62
    mean_matched = softened * 0.42 + local_mean * 0.58
    edge_alpha_3 = np.expand_dims(np.clip(edge_alpha, 0.0, 1.0), axis=2)
    matched = arr * (1.0 - edge_alpha_3) + mean_matched * edge_alpha_3
    return Image.fromarray(np.clip(matched, 0, 255).astype(np.uint8), mode="RGB")


def addit_subject_guided_blend_mask(person_mask, variant=None):
    """Outside-only Add-it proxy mask.

    Keep the accepted person core fully target-side. Blur/blend only the narrow
    outside transition ring and the foot-contact shadow region, so background
    cannot eat into the person silhouette.
    """
    if not ADDIT_SUBJECT_GUIDED_BLEND_PROXY:
        return person_mask.convert("L")
    mask = person_mask.convert("L")
    bbox = mask.getbbox()
    if bbox is None:
        return mask
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    hard = mask.point(lambda p: 255 if p >= PERSON_PASTE_HARD_THRESHOLD else 0)
    dilate_px = max(0, int(round(ADDIT_BLEND_CONTEXT_DILATE)))
    if dilate_px > 0:
        dilated = hard.filter(ImageFilter.MaxFilter(dilate_px * 2 + 1))
    else:
        dilated = hard

    outside_ring = ImageChops.subtract(dilated, hard)
    outside_soft = outside_ring.filter(ImageFilter.GaussianBlur(radius=max(0.0, ADDIT_BLEND_EDGE_RADIUS)))
    hard_arr = np.asarray(hard, dtype=np.float32) / 255.0
    outside_arr = np.asarray(outside_soft, dtype=np.float32) / 255.0
    outside_arr = outside_arr * (1.0 - hard_arr)
    outside_soft = Image.fromarray(np.clip(outside_arr * 255.0, 0, 255).astype(np.uint8), mode="L")

    shadow = Image.new("L", mask.size, 0)
    draw = ImageDraw.Draw(shadow)
    shadow_h = max(3, int(round(h * ADDIT_BLEND_SHADOW_EXTENSION)))
    sx1 = max(0, int(round(x1 - w * 0.18)))
    sx2 = min(mask.size[0], int(round(x2 + w * 0.18)))
    sy1 = max(0, int(round(y2 - h * 0.02)))
    sy2 = min(mask.size[1], int(round(y2 + shadow_h)))
    if sx2 > sx1 and sy2 > sy1:
        draw.ellipse((sx1, sy1, sx2, sy2), fill=int(255 * max(0.0, min(1.0, ADDIT_BLEND_SHADOW_ALPHA))))
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=max(0.0, ADDIT_BLEND_SHADOW_BLUR)))
        shadow_arr = np.asarray(shadow, dtype=np.float32) / 255.0
        shadow_arr = shadow_arr * (1.0 - hard_arr)
        shadow = Image.fromarray(np.clip(shadow_arr * 255.0, 0, 255).astype(np.uint8), mode="L")

    return ImageChops.lighter(ImageChops.lighter(hard, outside_soft), shadow)


def apply_addit_subject_guided_blend(source_crop, target_crop, person_mask, variant=None):
    if not ADDIT_CONCEPT_ENABLED or not ADDIT_SUBJECT_GUIDED_BLEND_PROXY:
        return target_crop
    blend_mask = addit_subject_guided_blend_mask(person_mask, variant=variant)
    return Image.composite(target_crop.convert("RGB"), source_crop.convert("RGB"), blend_mask)


def seamless_clone_person_crop(source_crop, person_rgb, person_mask):
    if not USE_SEAMLESS_CLONE or cv2 is None:
        return None, False
    mask = clean_binary_person_mask(person_mask, keep_components=ACCESSORY_KEEP_COMPONENTS)
    if mask_area_ratio(mask) < SEAMLESS_CLONE_MIN_MASK_AREA_RATIO:
        return None, False
    bbox = mask.getbbox()
    if bbox is None:
        return None, False
    x1, y1, x2, y2 = bbox
    center = (int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0)))
    if center[0] <= 0 or center[0] >= source_crop.size[0] or center[1] <= 0 or center[1] >= source_crop.size[1]:
        return None, False
    try:
        src = cv2.cvtColor(np.asarray(source_crop.convert("RGB")), cv2.COLOR_RGB2BGR)
        obj = cv2.cvtColor(np.asarray(person_rgb.convert("RGB")), cv2.COLOR_RGB2BGR)
        m = np.asarray(mask, dtype=np.uint8)
        mode = cv2.NORMAL_CLONE if SEAMLESS_CLONE_MODE == "normal" else cv2.MIXED_CLONE
        cloned = cv2.seamlessClone(obj, src, m, center, mode)
        return Image.fromarray(cv2.cvtColor(cloned, cv2.COLOR_BGR2RGB), mode="RGB"), True
    except Exception as exc:
        print("seamlessClone failed; falling back to alpha paste:", type(exc).__name__, exc)
        return None, False


def preserve_foreground_after_seamless(cloned_crop, person_rgb, person_mask):
    if not SEAMLESS_EDGE_ONLY_BLEND:
        return cloned_crop
    mask = person_mask.convert("L")
    if mask.getbbox() is None:
        return cloned_crop
    hard = mask.point(lambda p: 255 if p >= PERSON_PASTE_HARD_THRESHOLD else 0)
    core = hard
    if SEAMLESS_FOREGROUND_CORE_ERODE > 0:
        core = core.filter(ImageFilter.MinFilter(SEAMLESS_FOREGROUND_CORE_ERODE * 2 + 1))
    core = core.filter(ImageFilter.GaussianBlur(radius=0.65))
    alpha = np.asarray(core, dtype=np.float32) / 255.0
    alpha = np.expand_dims(np.clip(alpha * SEAMLESS_FOREGROUND_PRESERVE_STRENGTH, 0.0, 1.0), axis=2)
    cloned_arr = np.asarray(cloned_crop.convert("RGB"), dtype=np.float32)
    person_arr = np.asarray(person_rgb.convert("RGB"), dtype=np.float32)
    preserved = cloned_arr * (1.0 - alpha) + person_arr * alpha
    return Image.fromarray(np.clip(preserved, 0, 255).astype(np.uint8), mode="RGB")


def bbox_mask_from_bboxes(image_size, bboxes, padding=0, blur=0):
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    width, height = image_size
    for bbox in bboxes:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        if x2 > x1 and y2 > y1:
            draw.rectangle((x1, y1, x2, y2), fill=255)
    if blur and blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur))
    return mask


def vehicle_is_foreground_occluder(person_bbox, vehicle_bbox):
    inter = bbox_intersection_area(person_bbox, vehicle_bbox)
    if inter <= 0:
        return False, 0.0
    person_area = max(1.0, bbox_area(person_bbox))
    vehicle_area = max(1.0, bbox_area(vehicle_bbox))
    overlap_ratio = inter / min(person_area, vehicle_area)
    if overlap_ratio < MIN_OCCLUDER_OVERLAP_RATIO:
        return False, overlap_ratio
    person_h = max(1.0, person_bbox[3] - person_bbox[1])
    vehicle_h = max(1.0, vehicle_bbox[3] - vehicle_bbox[1])
    vehicle_close_enough = vehicle_bbox[3] >= person_bbox[3] - VEHICLE_OCCLUDER_MAX_FOOT_Y_DELTA
    vehicle_large_enough = vehicle_h >= person_h * OCCLUDER_MIN_HEIGHT_RATIO
    return bool(vehicle_close_enough or vehicle_large_enough), overlap_ratio


def build_foreground_occluder_mask(image_size, person_bbox, existing_person_bboxes=None, existing_vehicle_bboxes=None, variant=None):
    if not OCCLUSION_AWARE_COMPOSITE or person_bbox is None:
        return None, {"foreground_occlusion_used": False, "foreground_occluder_count": 0, "foreground_occlusion_overlap_ratio": 0.0}
    occluders = []
    max_overlap = 0.0
    for old_bbox in existing_person_bboxes or []:
        inter = bbox_intersection_area(person_bbox, old_bbox)
        if inter <= 0:
            continue
        depth_ok, overlap_ratio = person_overlap_depth_ok(person_bbox, old_bbox)
        if not depth_ok:
            occluders.append(old_bbox)
            max_overlap = max(max_overlap, overlap_ratio)
    for vehicle_bbox in existing_vehicle_bboxes or []:
        is_occluder, overlap_ratio = vehicle_is_foreground_occluder(person_bbox, vehicle_bbox)
        if is_occluder:
            occluders.append(vehicle_bbox)
            max_overlap = max(max_overlap, overlap_ratio)
    if not occluders:
        return None, {"foreground_occlusion_used": False, "foreground_occluder_count": 0, "foreground_occlusion_overlap_ratio": 0.0}
    mask = bbox_mask_from_bboxes(
        image_size,
        occluders,
        padding=OCCLUSION_MASK_BBOX_PADDING,
        blur=OCCLUSION_MASK_BLUR_RADIUS,
    )
    return mask, {
        "foreground_occlusion_used": True,
        "foreground_occluder_count": len(occluders),
        "foreground_occlusion_overlap_ratio": round(float(max_overlap), 4),
    }


def apply_foreground_occlusion_mask(person_mask, occlusion_mask, crop_bbox):
    if occlusion_mask is None:
        return person_mask, None, {"foreground_occlusion_removed_ratio": 0.0}
    cx1, cy1, cx2, cy2 = crop_bbox
    occlusion_crop = occlusion_mask.crop((cx1, cy1, cx2, cy2)).resize(person_mask.size, Image.BILINEAR)
    person_arr = np.asarray(person_mask.convert("L"), dtype=np.float32)
    occ_arr = np.asarray(occlusion_crop.convert("L"), dtype=np.float32)
    active = person_arr > 8
    if not np.any(active):
        return person_mask, occlusion_crop, {"foreground_occlusion_removed_ratio": 0.0}
    removed_ratio = float(np.mean((occ_arr > 16) & active) / max(1e-6, np.mean(active)))
    if removed_ratio > MAX_OCCLUSION_REMOVED_MASK_RATIO:
        return person_mask, None, {"foreground_occlusion_removed_ratio": round(removed_ratio, 4), "foreground_occlusion_skipped": True}
    kept = np.clip(person_arr * (1.0 - occ_arr / 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(kept, mode="L"), occlusion_crop, {"foreground_occlusion_removed_ratio": round(removed_ratio, 4)}


def paste_crop_person_to_original(source, generated_crop, person_mask_crop, crop_bbox, occlusion_mask=None):
    cx1, cy1, cx2, cy2 = crop_bbox
    crop_w = cx2 - cx1
    crop_h = cy2 - cy1
    person_rgb = generated_crop.resize((crop_w, crop_h), Image.BICUBIC)
    person_mask = prepare_person_paste_mask(person_mask_crop, (crop_w, crop_h))
    person_mask, occlusion_crop, occlusion_meta = apply_foreground_occlusion_mask(person_mask, occlusion_mask, crop_bbox)
    source_crop = source.crop(crop_bbox).resize((crop_w, crop_h), Image.BICUBIC)
    person_rgb = harmonize_person_to_scene(source_crop, person_rgb, person_mask)
    cloned_crop, seamless_used = seamless_clone_person_crop(source_crop, person_rgb, person_mask)
    result = source.copy()
    blend_meta = {
        "seamless_clone_used": bool(seamless_used),
        "fallback_alpha_paste": not bool(seamless_used),
        "fallback_alpha_used": not bool(seamless_used),
        "foreground_preserved_after_seamless": False,
        "edge_local_bg_match_used": False,
        **occlusion_meta,
    }
    if seamless_used and cloned_crop is not None:
        cloned_crop = preserve_foreground_after_seamless(cloned_crop, person_rgb, person_mask)
        blend_meta["foreground_preserved_after_seamless"] = True
        result.paste(cloned_crop, (cx1, cy1))
    else:
        result.paste(person_rgb, (cx1, cy1), person_mask)

    result_crop = result.crop(crop_bbox)
    result_crop = match_pasted_edge_to_composite_mean(result_crop, person_mask)
    blend_meta["edge_local_bg_match_used"] = True
    result_crop = apply_addit_subject_guided_blend(source_crop, result_crop, person_mask)
    if occlusion_crop is not None:
        result_crop = Image.composite(source_crop, result_crop, occlusion_crop)
    result.paste(result_crop, (cx1, cy1))
    return result, person_mask, blend_meta


def generate_context_person_composite_with_pipe(pipe, source, record, variant, prompt, negative_prompt, seed, device, strength, guidance_scale, num_inference_steps, debug_index=None):
    crop_bbox = (0, 0, source.size[0], source.size[1])
    crop_source = source.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
    original = load_source_image(record.path)
    existing_person_bboxes = load_person_bboxes_for_crop(record, original.size, resolution=source.size[0])
    existing_vehicle_bboxes = load_vehicle_bboxes_for_crop(record, original.size, resolution=source.size[0])
    semantic_masks = semantic_placement_masks(source, record, device=device)
    depth_map = estimate_depth_map(source, device="cpu")
    generation_source = crop_source
    mask_image = Image.new("L", crop_source.size, 0)
    generator_device = device if str(device).startswith("cuda") else "cpu"
    max_retries = variant_retry_budget(variant)
    last_reject_reason = None
    last_reject_meta = default_scale_correction_metadata()
    scale_unrecoverable_streak = 0
    attempt_history = []

    for attempt in range(max_retries + 1):
        attempt_seed = seed + attempt * 9973
        generator = torch.Generator(device=generator_device).manual_seed(attempt_seed)
        attempt_prompt, attempt_negative_prompt, attempt_strength, attempt_guidance, _attempt_margin = build_retry_config(
            prompt,
            negative_prompt,
            last_reject_reason,
            strength,
            guidance_scale,
            CONTEXT_CROP_EXPAND,
            attempt,
        )
        generated_crop = pipe(
            prompt=attempt_prompt,
            negative_prompt=attempt_negative_prompt,
            image=generation_source,
            strength=attempt_strength,
            guidance_scale=attempt_guidance,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0].resize(crop_source.size)

        person_mask_crop, detected_bbox, reject_reason, scale_meta, corrected_generated_crop = select_new_generated_person_mask(
            generated_crop,
            existing_person_bboxes=existing_person_bboxes,
            semantic_masks=semantic_masks,
            variant=variant,
            background_image=crop_source,
            depth_map=depth_map,
        )
        reject_reason = normalize_reject_reason(reject_reason)
        scale_meta = dict(scale_meta or {})
        scale_meta["retry_attempts"] = attempt
        scale_meta["attempt"] = attempt

        if person_mask_crop is None:
            if reject_reason == "scale_unrecoverable":
                scale_unrecoverable_streak += 1
            else:
                scale_unrecoverable_streak = 0
            scale_meta["scale_unrecoverable_streak"] = scale_unrecoverable_streak
            scale_meta["reject_reason"] = reject_reason
            scale_meta["last_reject_reason"] = reject_reason
            attempt_history.append(scale_meta)
            last_reject_reason = reject_reason
            last_reject_meta = scale_meta
            if should_retry(reject_reason, attempt, max_retries, scale_meta):
                print(
                    f"Rejected generated person ({reject_reason}) on attempt {attempt + 1}; "
                    "retrying with adaptive generation params."
                )
                continue
            break

        insert_bbox = tuple(int(round(v)) for v in detected_bbox)
        outside_ratio = mask_outside_bbox_ratio(person_mask_crop, insert_bbox)
        if outside_ratio > MAX_MASK_OUTSIDE_INSERTION_RATIO:
            reject_reason = "partial_or_cropped"
            scale_unrecoverable_streak = 0
            scale_meta.update({
                "reject_reason": reject_reason,
                "last_reject_reason": reject_reason,
                "mask_outside_bbox_ratio": round(outside_ratio, 4),
                "scale_unrecoverable_streak": scale_unrecoverable_streak,
            })
            attempt_history.append(scale_meta)
            last_reject_reason = reject_reason
            last_reject_meta = scale_meta
            print(f"Generated person mask is unstable around detected bbox (outside_ratio={outside_ratio:.2f}).")
            if should_retry(reject_reason, attempt, max_retries, scale_meta):
                print(f"Retry attempt {attempt + 1} because: {reject_reason}")
                continue
            break

        person_mask_crop = constrain_mask_to_bbox(person_mask_crop, insert_bbox)
        if corrected_generated_crop is not None:
            generated_crop = corrected_generated_crop
        occlusion_mask, occlusion_meta = build_foreground_occluder_mask(
            source.size,
            detected_bbox,
            existing_person_bboxes=existing_person_bboxes,
            existing_vehicle_bboxes=existing_vehicle_bboxes,
            variant=variant,
        )
        insert_meta = {
            "expected_person_height": insert_bbox[3] - insert_bbox[1],
            "expected_person_width": insert_bbox[2] - insert_bbox[0],
            "ground_y": insert_bbox[3],
            "img2img_first": True,
            "retry_attempts": attempt,
            **occlusion_meta,
            **scale_meta,
        }
        result, pasted_mask, blend_meta = paste_crop_person_to_original(source, generated_crop, person_mask_crop, crop_bbox, occlusion_mask=occlusion_mask)
        insert_meta.update(blend_meta)
        is_valid, final_reason, final_meta = validate_composite_result(
            source,
            result,
            pasted_mask,
            variant,
            insert_bbox,
            insert_meta,
        )
        final_meta["attempt"] = attempt
        final_meta["scale_unrecoverable_streak"] = scale_unrecoverable_streak
        attempt_history.append(final_meta)

        if is_valid:
            if attempt > 0:
                print(
                    f"Recovered accepted composite after retry {attempt} for {record.path.name} "
                    f"using strength={attempt_strength:.2f}, guidance={attempt_guidance:.2f}."
                )
            debug_mask = Image.new("L", source.size, 0)
            debug_mask.paste(pasted_mask, (crop_bbox[0], crop_bbox[1]))
            result = add_contact_shadow(result, insert_bbox, variant)
            debug_generated = source.copy()
            debug_generated.paste(
                generated_crop.resize((crop_bbox[2] - crop_bbox[0], crop_bbox[3] - crop_bbox[1]), Image.LANCZOS),
                (crop_bbox[0], crop_bbox[1]),
            )
            debug_path = save_inpaint_debug_strip(
                record, variant, seed, source, debug_mask, source, debug_generated, result,
                insert_bbox, debug_index=debug_index,
            )
            final_meta["attempt_history"] = json.dumps(attempt_history)
            final_meta["reject_reason"] = ""
            return result, insert_bbox, crop_bbox, debug_path, final_meta

        final_reason = normalize_reject_reason(final_reason)
        last_reject_reason = final_reason
        last_reject_meta = final_meta
        if should_retry(final_reason, attempt, max_retries, final_meta):
            print(f"Retry attempt {attempt + 1} because final validation failed: {final_reason}")
            continue
        break

    if CONTEXT_PERSON_FALLBACK_TO_BBOX_INPAINT:
        print("No accepted generated person after retries; falling back to bbox_inpaint composite for this sample.")
        generated_full = source.copy()
        generated_full_crop = generated_crop.resize((crop_bbox[2] - crop_bbox[0], crop_bbox[3] - crop_bbox[1]), Image.LANCZOS)
        full_mask_crop = mask_image.resize(generated_full_crop.size, Image.BILINEAR)
        generated_full.paste(generated_full_crop, (crop_bbox[0], crop_bbox[1]), full_mask_crop)
        result = add_contact_shadow(generated_full, insert_bbox if "insert_bbox" in locals() else (0, 0, source.size[0], source.size[1]), variant)
        debug_path = save_inpaint_debug_strip(
            record, variant, seed, source, mask_image.resize(source.size), source, result, result,
            insert_bbox if "insert_bbox" in locals() else (0, 0, source.size[0], source.size[1]), debug_index=debug_index,
        )
        fallback_meta = dict(last_reject_meta or default_scale_correction_metadata())
        fallback_meta["attempt_history"] = json.dumps(attempt_history)
        return result, insert_bbox if "insert_bbox" in locals() else None, crop_bbox, debug_path, fallback_meta

    final_reason = normalize_reject_reason(last_reject_reason or "bad_composite_quality")
    raise RuntimeError(f"No accepted generated person after adaptive retries (last_reason={final_reason}).")


def generate_human_mask_inpaint_with_pipe(pipe, source, record, variant, prompt, negative_prompt, seed, device, strength, guidance_scale, num_inference_steps, debug_index=None):
    depth_map = estimate_depth_map(source, device="cpu")
    rng = random.Random(seed)
    insert_bbox, insert_meta = find_insertion_region(record, source, variant, rng, device=device, return_metadata=True, depth_map=depth_map)
    if insert_bbox is None:
        raise RuntimeError(f"Could not find insertion region for {record.path.name}")
    if BACKGROUND_PRESERVATION_MODE == "bbox_inpaint":
        mask_image = bbox_mask_for_bbox(source.size, insert_bbox, variant=variant)
    else:
        mask_image = human_mask_for_bbox(source.size, insert_bbox, variant)
    inpaint_source = prepare_inpaint_source(source, mask_image, insert_bbox)
    generator_device = device if str(device).startswith("cuda") else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    generated = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=inpaint_source,
        mask_image=mask_image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0].resize(source.size)
    result = source.copy()
    result.paste(generated, (0, 0), mask_image)
    result = add_contact_shadow(result, insert_bbox, variant)
    patch_bbox = expand_bbox_with_context(insert_bbox, resolution=source.width)
    debug_path = save_inpaint_debug_strip(
        record, variant, seed, source, mask_image, inpaint_source, generated, result,
        insert_bbox, debug_index=debug_index,
    )
    return result, insert_bbox, patch_bbox, debug_path


def generate_patch_blend_with_pipe(pipe, source, record, variant, prompt, negative_prompt, seed, device, strength, guidance_scale, num_inference_steps, debug_index=None):
    depth_map = estimate_depth_map(source, device="cpu")
    rng = random.Random(seed)
    insert_bbox, insert_meta = find_insertion_region(record, source, variant, rng, device=device, return_metadata=True, depth_map=depth_map)
    if insert_bbox is None:
        raise RuntimeError(f"Could not find insertion region for {record.path.name}")
    patch_bbox = expand_bbox_with_context(insert_bbox, resolution=source.width)
    source_patch = source.crop(patch_bbox)
    guided_patch = draw_person_guide_on_patch(source_patch, patch_bbox, insert_bbox, variant)
    generator_device = device if str(device).startswith("cuda") else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    aug_patch = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=guided_patch,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0].resize(source_patch.size)
    ix1, iy1, ix2, iy2 = insert_bbox
    px1, py1, _, _ = patch_bbox
    rel_insert_bbox = (ix1 - px1, iy1 - py1, ix2 - px1, iy2 - py1)
    generated_insert = aug_patch.crop(rel_insert_bbox)
    result = source.copy()
    result.paste(generated_insert, (ix1, iy1), feather_mask(generated_insert.size))
    final_patch = result.crop(patch_bbox)
    debug_path = save_patch_debug_strip(
        record, variant, seed, source_patch, guided_patch, aug_patch, final_patch,
        patch_bbox, insert_bbox, debug_index=debug_index,
    )
    return result, insert_bbox, patch_bbox, debug_path


def generate_variant_with_pipe(pipe, record, variant, output_path, seed, device=TRAIN_DEVICE, strength=None, debug_index=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source = resize_center_crop(load_source_image(record.path), resolution=RESOLUTION)
    prompt = build_generation_prompt(record, variant)
    negative_prompt = build_variant_negative_prompt(variant)
    strength = VARIANT_STRENGTHS.get(variant, AUGMENTATION_STRENGTH) if strength is None else strength
    guidance_scale = VARIANT_GUIDANCE_SCALES.get(variant, GUIDANCE_SCALE)
    num_inference_steps = VARIANT_NUM_INFERENCE_STEPS.get(variant, NUM_INFERENCE_STEPS)
    if seed == SEED:
        print("Prompt sample:", prompt)
        print("Negative sample:", negative_prompt)
        print(f"Generation config: variant={variant}, mode={BACKGROUND_PRESERVATION_MODE}, strength={strength}, guidance_scale={guidance_scale}, steps={num_inference_steps}, resolution={RESOLUTION}")
    clear_cuda()
    scale_meta = default_scale_correction_metadata()
    if BACKGROUND_PRESERVATION_MODE == "context_person_composite":
        image, insert_bbox, patch_bbox, debug_path, scale_meta = generate_context_person_composite_with_pipe(
            pipe, source, record, variant, prompt, negative_prompt, seed, device,
            strength, guidance_scale, num_inference_steps, debug_index=debug_index,
        )
    elif BACKGROUND_PRESERVATION_MODE in {"human_mask_inpaint", "bbox_inpaint"}:
        image, insert_bbox, patch_bbox, debug_path = generate_human_mask_inpaint_with_pipe(
            pipe, source, record, variant, prompt, negative_prompt, seed, device,
            strength, guidance_scale, num_inference_steps, debug_index=debug_index,
        )
    elif BACKGROUND_PRESERVATION_MODE == "patch_blend":
        image, insert_bbox, patch_bbox, debug_path = generate_patch_blend_with_pipe(
            pipe, source, record, variant, prompt, negative_prompt, seed, device,
            strength, guidance_scale, num_inference_steps, debug_index=debug_index,
        )
    else:
        generator_device = device if str(device).startswith("cuda") else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=source,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]
        insert_bbox = None
        patch_bbox = None
        debug_path = ""
    image.save(output_path)
    clear_cuda()
    return output_path, {
        "strength": strength,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "generation_mode": BACKGROUND_PRESERVATION_MODE,
        "insert_bbox": insert_bbox,
        "patch_bbox": patch_bbox,
        "patch_debug_path": debug_path,
        "expected_new_person_count": expected_new_person_count(variant),
        "detected_new_person_count": expected_new_person_count(variant) if insert_bbox is not None else 0,
        **scale_meta,
    }
