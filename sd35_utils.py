"""SD3.5 CityPersons augmentation: shared preprocessing, placement, masks, and scale helpers."""

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
from sd35_data import ImageRecord

SEMANTIC_SEGMENTER = None
SEMANTIC_MASK_CACHE = {}

def load_source_image(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def resize_center_crop(image, resolution=RESOLUTION):
    width, height = image.size
    scale = resolution / min(width, height)
    new_size = (round(width * scale), round(height * scale))
    image = image.resize(new_size, Image.BICUBIC)
    left = (image.width - resolution) // 2
    top = (image.height - resolution) // 2
    return image.crop((left, top, left + resolution, top + resolution))


def image_to_tensor(image, resolution=RESOLUTION, device="cuda", dtype=torch.float16):
    image = resize_center_crop(image, resolution)
    pixel_values = torch.tensor(list(image.getdata()), dtype=torch.float32).view(resolution, resolution, 3)
    pixel_values = pixel_values.permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    return pixel_values.to(device=device, dtype=dtype)


def clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
def records_by_split_and_bucket(records):
    grouped = {}
    for record in records:
        grouped.setdefault(record.split, {}).setdefault(record.bucket, []).append(record)
    return grouped


def choose_records_for_bucket(bucket_records, target_count, rng):
    if not bucket_records:
        return []
    if len(bucket_records) >= target_count:
        return rng.sample(bucket_records, target_count)
    return [rng.choice(bucket_records) for _ in range(target_count)]


def choose_record_for_variant(bucket_records, variant, rng):
    return rng.choice(bucket_records)


def generated_image_path(output_dir, record, variant, index):
    safe_variant = variant.replace("/", "_")
    file_name = f"{record.path.stem}_aug_{index:04d}_{safe_variant}.png"
    return Path(output_dir) / record.split / record.bucket / IMAGE_SUBDIR / file_name


def comparison_image_path(output_dir, record, variant, index):
    safe_variant = variant.replace("/", "_")
    file_name = f"{record.path.stem}_pair_{index:04d}_{safe_variant}.png"
    return Path(output_dir) / "comparison_pairs" / record.split / record.bucket / file_name


def save_comparison_pair(original, augmented, comparison_path, title):
    comparison_path = Path(comparison_path)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    original = original.convert("RGB")
    augmented = augmented.convert("RGB").resize(original.size)
    title_h = 34
    label_h = 28
    width = original.width * 2
    height = original.height + title_h + label_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill=(0, 0, 0))
    draw.text((10, title_h + 6), "original", fill=(0, 0, 0))
    draw.text((original.width + 10, title_h + 6), "augmented", fill=(0, 0, 0))
    canvas.paste(original, (0, title_h + label_h))
    canvas.paste(augmented, (original.width, title_h + label_h))
    canvas.save(comparison_path)
    return comparison_path


def clamp_bbox(bbox, width, height):
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(x1 + 1, min(width, int(round(x2))))
    y2 = max(y1 + 1, min(height, int(round(y2))))
    return (x1, y1, x2, y2)


def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))


def union_bboxes(bboxes):
    if not bboxes:
        return None
    return (
        min(float(bbox[0]) for bbox in bboxes),
        min(float(bbox[1]) for bbox in bboxes),
        max(float(bbox[2]) for bbox in bboxes),
        max(float(bbox[3]) for bbox in bboxes),
    )


def mask_bbox_touches_border(mask_bbox, size, margin=PERSON_BORDER_REJECT_PIXELS):
    if mask_bbox is None:
        return True
    width, height = size
    x1, y1, x2, y2 = mask_bbox
    return x1 <= margin or y1 <= margin or x2 >= width - margin or y2 >= height - margin


def person_overlap_depth_ok(front_bbox, occluded_bbox):
    """Allow overlap only when the occluded person is plausibly behind the pasted one."""
    inter = bbox_intersection_area(front_bbox, occluded_bbox)
    if inter <= 0:
        return True, 0.0
    front_area = max(1.0, bbox_area(front_bbox))
    occluded_area = max(1.0, bbox_area(occluded_bbox))
    overlap_ratio = inter / min(front_area, occluded_area)
    if not ALLOW_PERSON_PERSON_OVERLAP:
        return False, overlap_ratio
    if overlap_ratio > MAX_PERSON_PERSON_OVERLAP_RATIO:
        return False, overlap_ratio
    front_h = max(1.0, front_bbox[3] - front_bbox[1])
    occluded_h = max(1.0, occluded_bbox[3] - occluded_bbox[1])
    front_foot_y = float(front_bbox[3])
    occluded_foot_y = float(occluded_bbox[3])
    if front_h <= occluded_h:
        return False, overlap_ratio
    if occluded_h > front_h * OCCLUDED_PERSON_MAX_HEIGHT_RATIO:
        return False, overlap_ratio
    if front_foot_y < occluded_foot_y - OCCLUDED_PERSON_MAX_FOOT_Y_DELTA:
        return False, overlap_ratio
    if occluded_foot_y > front_foot_y + OCCLUDED_PERSON_MAX_FOOT_Y_DELTA:
        return False, overlap_ratio
    return True, overlap_ratio


def center_crop_geometry(original_size, resolution=RESOLUTION):
    width, height = original_size
    scale = resolution / min(width, height)
    resized_w = round(width * scale)
    resized_h = round(height * scale)
    crop_left = (resized_w - resolution) // 2
    crop_top = (resized_h - resolution) // 2
    return scale, crop_left, crop_top


def yolo_bbox_to_crop_bbox(parts, original_size, resolution=RESOLUTION, class_ids=None):
    class_id = int(float(parts[0]))
    if class_ids is not None and class_id not in class_ids:
        return None
    width, height = original_size
    xc, yc, bw, bh = [float(value) for value in parts[1:5]]
    x1 = (xc - bw / 2) * width
    y1 = (yc - bh / 2) * height
    x2 = (xc + bw / 2) * width
    y2 = (yc + bh / 2) * height
    scale, crop_left, crop_top = center_crop_geometry(original_size, resolution)
    crop_bbox = (
        x1 * scale - crop_left,
        y1 * scale - crop_top,
        x2 * scale - crop_left,
        y2 * scale - crop_top,
    )
    if crop_bbox[2] <= 0 or crop_bbox[0] >= resolution or crop_bbox[3] <= 0 or crop_bbox[1] >= resolution:
        return None
    clamped = clamp_bbox(crop_bbox, resolution, resolution)
    return clamped if bbox_area(clamped) > 0 else None


def load_yolo_bboxes_for_crop(record, original_size, class_ids, resolution=RESOLUTION):
    if not record.label_path or not Path(record.label_path).exists() or Path(record.label_path).suffix.lower() != ".txt":
        return []
    bboxes = []
    with Path(record.label_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                bbox = yolo_bbox_to_crop_bbox(parts, original_size, resolution, class_ids=class_ids)
            except ValueError:
                continue
            if bbox:
                bboxes.append(bbox)
    return bboxes


def load_person_bboxes_for_crop(record, original_size, resolution=RESOLUTION):
    return load_yolo_bboxes_for_crop(record, original_size, PATCH_PERSON_CLASS_IDS, resolution=resolution)


def load_vehicle_bboxes_for_crop(record, original_size, resolution=RESOLUTION):
    return load_yolo_bboxes_for_crop(record, original_size, PATCH_VEHICLE_CLASS_IDS, resolution=resolution)


def load_semantic_segmenter(device=TRAIN_DEVICE):
    global SEMANTIC_SEGMENTER
    if not USE_SEMANTIC_PLACEMENT or SMART_PLACEMENT_VERSION != "v2":
        return None
    if SEMANTIC_SEGMENTER is False:
        return None
    if SEMANTIC_SEGMENTER is not None:
        return SEMANTIC_SEGMENTER
    try:
        import warnings
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The following named arguments are not valid for `SegformerImageProcessor.__init__`.*",
                category=UserWarning,
                module="transformers.image_processing_base",
            )
            image_processor = AutoImageProcessor.from_pretrained(
                SEMANTIC_SEGMENTATION_MODEL_ID,
                use_fast=False,
            )
        segmentation_model = AutoModelForSemanticSegmentation.from_pretrained(
            SEMANTIC_SEGMENTATION_MODEL_ID,
            low_cpu_mem_usage=False,
        ).to("cpu")
        segmentation_model.eval()
        SEMANTIC_SEGMENTER = {
            "processor": image_processor,
            "model": segmentation_model,
            "device": "cpu",  # keep SegFormer off GPU; SD3.5 already owns the GPUs/offload state
        }
        print(f"Loaded SegFormer semantic placement model on CPU: {SEMANTIC_SEGMENTATION_MODEL_ID}")
    except Exception as exc:
        SEMANTIC_SEGMENTER = False
        print("Semantic placement disabled; falling back to Smart Placement V1 rules.")
        print(type(exc).__name__, exc)
    return None if SEMANTIC_SEGMENTER is False else SEMANTIC_SEGMENTER


def label_matches(label, label_set):
    label = str(label).lower().replace("_", " ")
    return any(target in label for target in label_set)


def semantic_placement_masks(source, record, device=TRAIN_DEVICE):
    cache_key = (str(record.path), source.size)
    if cache_key in SEMANTIC_MASK_CACHE:
        return SEMANTIC_MASK_CACHE[cache_key]
    segmenter = load_semantic_segmenter(device=device)
    if segmenter is None:
        SEMANTIC_MASK_CACHE[cache_key] = None
        return None
    try:
        processor = segmenter["processor"]
        model = segmenter["model"]
        image = source.convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to("cpu") for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = torch.nn.functional.interpolate(
            outputs.logits,
            size=(source.height, source.width),
            mode="bilinear",
            align_corners=False,
        )
        semantic_ids = logits.argmax(dim=1)[0].detach().cpu().numpy()
    except Exception as exc:
        print("Semantic segmentation failed for this image; using V1 placement fallback for it.")
        print(type(exc).__name__, exc)
        SEMANTIC_MASK_CACHE[cache_key] = None
        return None

    id2label = getattr(model.config, "id2label", {}) or {}
    valid_arr = np.zeros((source.height, source.width), dtype=np.uint8)
    avoid_arr = np.zeros((source.height, source.width), dtype=np.uint8)
    valid_hits = 0
    for class_id in np.unique(semantic_ids):
        label = id2label.get(int(class_id), id2label.get(str(int(class_id)), str(class_id)))
        class_mask = semantic_ids == class_id
        if label_matches(label, VALID_PLACEMENT_LABELS):
            valid_arr[class_mask] = 255
            valid_hits += 1
        if label_matches(label, AVOID_PLACEMENT_LABELS):
            avoid_arr[class_mask] = 255
    if not valid_hits:
        print("SegFormer produced no valid road/sidewalk/terrain labels; using V1 placement fallback for this image.")
        masks = None
    else:
        masks = {
            "valid": Image.fromarray(valid_arr, mode="L"),
            "avoid": Image.fromarray(avoid_arr, mode="L"),
        }
    SEMANTIC_MASK_CACHE[cache_key] = masks
    return masks


def mask_coverage(mask, bbox):
    if mask is None:
        return 0.0
    x1, y1, x2, y2 = clamp_bbox(bbox, mask.width, mask.height)
    crop = np.asarray(mask.crop((x1, y1, x2, y2)), dtype=np.float32) / 255.0
    if crop.size == 0:
        return 0.0
    return float(crop.mean())


def foot_support_bbox(candidate):
    x1, y1, x2, y2 = candidate
    h = y2 - y1
    band_h = max(4, int(h * 0.16))
    return (x1, max(y1, y2 - band_h), x2, y2)


def perspective_scale_for_ground_y(ground_y, resolution=RESOLUTION):
    y_far = resolution * PATCH_ROAD_Y_RANGE[0]
    y_near = resolution * PATCH_ROAD_Y_RANGE[1]
    t = (ground_y - y_far) / max(1.0, y_near - y_far)
    t = max(0.0, min(1.0, t))
    return PERSPECTIVE_SCALE_FAR + t * (PERSPECTIVE_SCALE_NEAR - PERSPECTIVE_SCALE_FAR)


def reference_person_samples(existing_person_bboxes, resolution=RESOLUTION):
    samples = []
    for bbox in existing_person_bboxes:
        x1, y1, x2, y2 = bbox
        person_h = y2 - y1
        person_w = x2 - x1
        if person_h < REFERENCE_SCALE_MIN_PERSON_HEIGHT or person_h > REFERENCE_SCALE_MAX_PERSON_HEIGHT:
            continue
        if person_w <= 0 or person_h <= 0:
            continue
        aspect = person_w / max(1, person_h)
        if aspect < 0.12 or aspect > 0.85:
            continue
        samples.append({"ground_y": y2, "height": person_h, "width": person_w})
    return samples


def median(values):
    values = sorted(values)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def fitted_reference_height_at_y(ground_y, samples, resolution=RESOLUTION):
    if len(samples) < REFERENCE_SCALE_MIN_SAMPLES:
        return None
    slopes = []
    for i, a in enumerate(samples):
        for b in samples[i + 1:]:
            dy = b["ground_y"] - a["ground_y"]
            if abs(dy) < REFERENCE_SCALE_MIN_Y_GAP:
                continue
            slope = (b["height"] - a["height"]) / dy
            if REFERENCE_SCALE_MIN_SLOPE <= slope <= REFERENCE_SCALE_MAX_SLOPE:
                slopes.append(slope)
    slope = median(slopes)
    if slope is None:
        return None
    intercepts = [sample["height"] - slope * sample["ground_y"] for sample in samples]
    intercept = median(intercepts)
    predicted = slope * ground_y + intercept
    return predicted if predicted > 0 else None


def single_reference_height_at_y(ground_y, samples, resolution=RESOLUTION):
    if not samples:
        return None
    nearest = min(samples, key=lambda sample: abs(sample["ground_y"] - ground_y))
    if abs(nearest["ground_y"] - ground_y) > REFERENCE_SCALE_MAX_Y_DISTANCE:
        return None
    ref_scale = perspective_scale_for_ground_y(nearest["ground_y"], resolution=resolution)
    target_scale = perspective_scale_for_ground_y(ground_y, resolution=resolution)
    return nearest["height"] * target_scale / max(1e-6, ref_scale)


def robust_reference_height_at_y(ground_y, existing_person_bboxes, resolution=RESOLUTION):
    samples = reference_person_samples(existing_person_bboxes, resolution=resolution)
    if not samples:
        return None
    fitted = fitted_reference_height_at_y(ground_y, samples, resolution=resolution)
    if fitted is not None:
        return fitted
    return single_reference_height_at_y(ground_y, samples, resolution=resolution)


def reference_vehicle_samples(existing_vehicle_bboxes, resolution=RESOLUTION):
    samples = []
    for bbox in existing_vehicle_bboxes:
        x1, y1, x2, y2 = bbox
        vehicle_h = y2 - y1
        vehicle_w = x2 - x1
        if vehicle_h < CAR_REFERENCE_MIN_HEIGHT or vehicle_h > CAR_REFERENCE_MAX_HEIGHT:
            continue
        if vehicle_w <= 0 or vehicle_h <= 0:
            continue
        aspect = vehicle_w / max(1, vehicle_h)
        if aspect < 1.0 or aspect > 4.8:
            continue
        same_depth_ratio = max(CAR_HEIGHT_TO_PERSON_HEIGHT_MIN_RATIO, min(CAR_HEIGHT_TO_PERSON_HEIGHT_MAX_RATIO, CAR_HEIGHT_TO_PERSON_HEIGHT_RATIO))
        person_equivalent_h = vehicle_h * same_depth_ratio
        samples.append({"ground_y": y2, "height": person_equivalent_h, "width": vehicle_w})
    return samples


def single_vehicle_reference_height_at_y(ground_y, samples):
    close = [sample for sample in samples if abs(sample["ground_y"] - ground_y) <= CAR_REFERENCE_MAX_Y_DISTANCE]
    if not close:
        return None
    weights = []
    heights = []
    for sample in close:
        distance = abs(sample["ground_y"] - ground_y)
        weights.append(1.0 / (1.0 + distance))
        heights.append(sample["height"])
    total_weight = sum(weights)
    if total_weight <= 0:
        return None
    return sum(height * weight for height, weight in zip(heights, weights)) / total_weight


def robust_vehicle_reference_height_at_y(ground_y, existing_vehicle_bboxes, resolution=RESOLUTION):
    samples = reference_vehicle_samples(existing_vehicle_bboxes, resolution=resolution)
    if not samples:
        return None
    fitted = fitted_reference_height_at_y(ground_y, samples, resolution=resolution)
    if fitted is not None:
        return fitted
    return single_vehicle_reference_height_at_y(ground_y, samples)


def combine_reference_heights(person_reference_h, vehicle_reference_h):
    if person_reference_h is not None and vehicle_reference_h is not None:
        return (1.0 - CAR_REFERENCE_SCALE_BLEND) * person_reference_h + CAR_REFERENCE_SCALE_BLEND * vehicle_reference_h
    if person_reference_h is not None:
        return person_reference_h
    return vehicle_reference_h


def fallback_person_height_for_variant(variant, resolution=RESOLUTION, ground_y=None):
    base_heights = {
        "add_single_pedestrian": 96,
        "add_two_pedestrians": 98,
        "add_small_group": 102,
        "add_occluded_pedestrian": 90,
        "add_distant_pedestrian": 68,
        "add_near_pedestrian": 150,
    }
    height = base_heights.get(variant, 96)
    if ground_y is not None:
        height *= perspective_scale_for_ground_y(ground_y, resolution=resolution)
        y_norm = max(0.0, min(1.0, ground_y / max(1, resolution)))
        max_ratio = 0.62 if variant == "add_near_pedestrian" else 0.50
        if variant == "add_distant_pedestrian":
            max_ratio = 0.26
        # Keep the envelope plausible for CityPersons perspective; SD3.5 still has slack inside it.
        height = min(height, resolution * max_ratio)
        if y_norm < 0.74:
            height = min(height, resolution * 0.34)
    return height


def min_target_height_ratio_for_variant(variant):
    if variant == "add_near_pedestrian":
        return MIN_ACCEPTED_NEAR_HEIGHT_RATIO
    if variant == "add_distant_pedestrian":
        return MIN_ACCEPTED_DISTANT_HEIGHT_RATIO
    if variant == "add_occluded_pedestrian":
        return 0.075
    if variant in {"add_two_pedestrians", "add_small_group"}:
        return 0.085
    return MIN_ACCEPTED_SINGLE_HEIGHT_RATIO


def variant_scale_multiplier(variant):
    if variant == "add_distant_pedestrian":
        return 0.82
    if variant == "add_near_pedestrian":
        return NEAR_PERSON_SCALE_MULTIPLIER
    if variant == "add_occluded_pedestrian":
        return 0.92
    return 1.0


def variant_insert_size(variant, resolution=RESOLUTION, ground_x=None, ground_y=None, existing_person_bboxes=None, existing_vehicle_bboxes=None, depth_map=None):
    if depth_map is not None and ground_x is not None and ground_y is not None:
        fallback_h = expected_person_height_from_depth(ground_x, ground_y, depth_map, resolution, variant)
    else:
        fallback_h = fallback_person_height_for_variant(variant, resolution=resolution, ground_y=ground_y)
    person_reference_h = None
    vehicle_reference_h = None
    if USE_REFERENCE_PERSON_SCALE and existing_person_bboxes and ground_y is not None:
        person_reference_h = robust_reference_height_at_y(ground_y, existing_person_bboxes, resolution=resolution)
    if existing_vehicle_bboxes and ground_y is not None:
        vehicle_reference_h = robust_vehicle_reference_height_at_y(ground_y, existing_vehicle_bboxes, resolution=resolution)
    reference_h = combine_reference_heights(person_reference_h, vehicle_reference_h)
    if reference_h is not None:
        min_h = fallback_h * REFERENCE_SCALE_MIN_FACTOR
        max_h = fallback_h * REFERENCE_SCALE_MAX_FACTOR
        reference_h = max(min_h, min(reference_h, max_h))
        target_h = REFERENCE_SCALE_BLEND * reference_h + (1.0 - REFERENCE_SCALE_BLEND) * fallback_h
    else:
        target_h = fallback_h
    target_h *= max(variant_scale_multiplier(variant), PERSON_TARGET_SCALE_MULTIPLIER)
    if ground_y is not None:
        y_norm = max(0.0, min(1.0, ground_y / max(1, resolution)))
        max_h = resolution * (0.28 + 0.34 * y_norm)
        if variant == "add_distant_pedestrian":
            max_h = min(max_h, resolution * 0.25)
        elif variant == "add_near_pedestrian":
            max_h = min(max_h, resolution * 0.72)
        min_h = resolution * min_target_height_ratio_for_variant(variant)
        target_h = max(min_h, min(target_h, max_h))
    if variant in {"add_two_pedestrians", "add_small_group"}:
        count = 2 if variant == "add_two_pedestrians" else 3
        target_w = target_h * PERSON_ASPECT_RATIO * count * 0.86
    else:
        target_w = target_h * PERSON_ASPECT_RATIO
    if FLEXIBLE_SCALE_INPAINT:
        width = target_w * SCALE_ENVELOPE_WIDTH_MULT
        height = target_h * SCALE_ENVELOPE_HEIGHT_MULT
    else:
        width = target_w
        height = target_h
    width = round(max(28, min(width, resolution - 2 * INSERTION_EDGE_MARGIN)))
    height = round(max(46, min(height, resolution - 2 * INSERTION_EDGE_MARGIN)))
    return width, height


def expand_bbox_with_context(bbox, resolution=RESOLUTION, context_ratio=PATCH_CONTEXT_RATIO):
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    pad = int(max(bw, bh) * context_ratio)
    x1, y1, x2, y2 = clamp_bbox((x1 - pad, y1 - pad, x2 + pad, y2 + pad), resolution, resolution)
    patch_w = x2 - x1
    patch_h = y2 - y1
    if patch_w < PATCH_MIN_SIZE:
        extra = (PATCH_MIN_SIZE - patch_w) // 2
        x1, y1, x2, y2 = clamp_bbox((x1 - extra, y1, x2 + extra, y2), resolution, resolution)
    if patch_h < PATCH_MIN_SIZE:
        extra = (PATCH_MIN_SIZE - patch_h) // 2
        x1, y1, x2, y2 = clamp_bbox((x1, y1 - extra, x2, y2 + extra), resolution, resolution)
    return (x1, y1, x2, y2)


def candidate_insertion_score(candidate, existing_person_bboxes, existing_vehicle_bboxes=None, semantic_masks=None, resolution=RESOLUTION, placement_target=None):
    x1, y1, x2, y2 = candidate
    width = x2 - x1
    height = y2 - y1
    area = max(1, width * height)
    cx = (x1 + x2) / 2
    ground_y = y2
    y_min = resolution * PATCH_ROAD_Y_RANGE[0]
    y_max = resolution * PATCH_ROAD_Y_RANGE[1]
    y_mid = y_min + 0.68 * (y_max - y_min)
    road_score = 1.0 - min(1.0, abs(ground_y - y_mid) / max(1.0, y_max - y_min))
    center_score = 1.0 - min(1.0, abs(cx - resolution / 2) / (resolution / 2))
    margin = min(x1, y1, resolution - x2, resolution - y2)
    margin_score = min(1.0, max(0.0, margin / INSERTION_EDGE_MARGIN))
    person_overlap_ratio = 0.0
    for bbox in existing_person_bboxes:
        depth_ok, overlap = person_overlap_depth_ok(candidate, bbox)
        person_overlap_ratio = max(person_overlap_ratio, overlap)
        if not depth_ok:
            return -1e9
    vehicle_overlap_ratio = 0.0
    if existing_vehicle_bboxes:
        for bbox in existing_vehicle_bboxes:
            vehicle_overlap_ratio += bbox_intersection_area(candidate, bbox) / area
        vehicle_overlap_ratio = min(1.0, vehicle_overlap_ratio)
    size_ratio = height / resolution
    if size_ratio < 0.10:
        size_score = size_ratio / 0.10
    elif size_ratio > 0.42:
        size_score = max(0.0, 1.0 - (size_ratio - 0.42) / 0.20)
    else:
        size_score = 1.0
    slot_score = 0.0
    if placement_target is not None:
        target_x, target_y = placement_target
        dx = abs(cx / resolution - target_x)
        dy = abs(ground_y / resolution - target_y)
        slot_score = max(0.0, 1.0 - (dx / 0.24 + dy / 0.18) / 2.0)
    score = (
        1.10 * road_score
        + INSERTION_CENTER_BIAS * center_score
        + PLACEMENT_SLOT_BIAS * slot_score
        + 0.35 * margin_score
        + 0.45 * size_score
    )
    if person_overlap_ratio > 0:
        score -= INSERTION_OVERLAP_PENALTY * 0.35 * person_overlap_ratio
    if ALLOW_PERSON_VEHICLE_OVERLAP and vehicle_overlap_ratio > 0:
        score += VEHICLE_OVERLAP_FRONT_LAYER_BONUS * min(vehicle_overlap_ratio, MAX_VEHICLE_OVERLAP_RATIO)
        if vehicle_overlap_ratio > MAX_VEHICLE_OVERLAP_RATIO:
            score -= INSERTION_OVERLAP_PENALTY * (vehicle_overlap_ratio - MAX_VEHICLE_OVERLAP_RATIO)
    elif vehicle_overlap_ratio > 0:
        score -= INSERTION_OVERLAP_PENALTY * vehicle_overlap_ratio
    if REQUIRE_SEMANTIC_PLACEMENT and semantic_masks is None:
        return -1e9
    if semantic_masks:
        valid_mask = semantic_masks.get("valid")
        avoid_mask = semantic_masks.get("avoid")
        foot_bbox = foot_support_bbox(candidate)
        foot_score = mask_coverage(valid_mask, foot_bbox)
        foot_avoid_score = mask_coverage(avoid_mask, foot_bbox)
        body_valid_score = mask_coverage(valid_mask, candidate)
        avoid_score = mask_coverage(avoid_mask, candidate)
        if foot_score < MIN_FOOT_SUPPORT:
            return -1e9
        if foot_avoid_score > MAX_FOOT_AVOID_SUPPORT:
            return -1e9
        if REQUIRE_SEMANTIC_PLACEMENT and body_valid_score < MIN_BODY_VALID_SUPPORT:
            return -1e9
        if REQUIRE_SEMANTIC_PLACEMENT and avoid_score > MAX_BODY_AVOID_SUPPORT:
            return -1e9
        score += SEMANTIC_FOOT_WEIGHT * foot_score
        score += 0.45 * body_valid_score
        score -= SEMANTIC_AVOID_PENALTY * avoid_score
    return score


def ground_y_range_for_variant(variant, height):
    y_min = int(height * PATCH_ROAD_Y_RANGE[0])
    y_max = int(height * PATCH_ROAD_Y_RANGE[1])
    span = max(1, y_max - y_min)
    if variant == "add_distant_pedestrian":
        return y_min, y_min + int(span * 0.34)
    if variant == "add_near_pedestrian":
        return y_min + int(span * 0.58), y_max
    if variant == "add_occluded_pedestrian":
        return y_min + int(span * 0.20), y_min + int(span * 0.78)
    return y_min, y_max


def placement_target_for_variant(variant, rng):
    if variant == "add_distant_pedestrian":
        y_choices = PLACEMENT_SLOT_YS[:2]
    elif variant == "add_near_pedestrian":
        y_choices = PLACEMENT_SLOT_YS[-2:]
    else:
        y_choices = PLACEMENT_SLOT_YS
    target_x = rng.choice(PLACEMENT_SLOT_XS) + rng.uniform(-PLACEMENT_SLOT_JITTER, PLACEMENT_SLOT_JITTER)
    target_y = rng.choice(y_choices) + rng.uniform(-PLACEMENT_SLOT_JITTER, PLACEMENT_SLOT_JITTER)
    target_x = max(0.08, min(0.92, target_x))
    target_y = max(PATCH_ROAD_Y_RANGE[0], min(PATCH_ROAD_Y_RANGE[1], target_y))
    return target_x, target_y


def sample_ground_y_for_variant(variant, y_min, y_max, rng):
    if y_max <= y_min:
        return y_min
    u = rng.random()
    if variant == "add_distant_pedestrian":
        u = 0.90 * (u ** 1.35)
    elif variant == "add_near_pedestrian":
        u = 0.12 + 0.76 * (u ** 1.25)
    else:
        u = 0.06 + 0.84 * (u ** 1.15)
    return int(round(y_min + max(0.0, min(0.92, u)) * (y_max - y_min)))


def find_insertion_region(record, source, variant, rng, device=TRAIN_DEVICE, return_metadata=False, depth_map=None):
    width, height = source.size
    original = load_source_image(record.path)
    existing_person_bboxes = load_person_bboxes_for_crop(record, original.size, resolution=width)
    existing_vehicle_bboxes = load_vehicle_bboxes_for_crop(record, original.size, resolution=width)
    semantic_masks = semantic_placement_masks(source, record, device=device)
    y_min, y_max = ground_y_range_for_variant(variant, height)
    placement_target = placement_target_for_variant(variant, rng)
    best_bbox = None
    best_score = -1e9
    best_meta = None
    for _ in range(PATCH_MAX_PLACEMENT_TRIES):
        ground_y = sample_ground_y_for_variant(variant, y_min, y_max, rng)
        candidate_x = rng.randint(INSERTION_EDGE_MARGIN, width - INSERTION_EDGE_MARGIN)
        insert_w, insert_h = variant_insert_size(
            variant,
            resolution=width,
            ground_x=candidate_x,
            ground_y=ground_y,
            existing_person_bboxes=existing_person_bboxes,
            existing_vehicle_bboxes=existing_vehicle_bboxes,
            depth_map=depth_map
        )
        if width - insert_w - INSERTION_EDGE_MARGIN <= INSERTION_EDGE_MARGIN:
            continue
        x1 = max(INSERTION_EDGE_MARGIN, min(width - insert_w - INSERTION_EDGE_MARGIN, candidate_x - insert_w // 2))
        y1 = max(INSERTION_EDGE_MARGIN, min(height - insert_h - INSERTION_EDGE_MARGIN, ground_y - insert_h))
        candidate = (x1, y1, x1 + insert_w, y1 + insert_h)
        score = candidate_insertion_score(
            candidate,
            existing_person_bboxes,
            existing_vehicle_bboxes=existing_vehicle_bboxes,
            semantic_masks=semantic_masks,
            resolution=width,
            placement_target=placement_target,
        )
        if score > best_score:
            best_bbox = candidate
            best_score = score
            expected_person_h = insert_h / max(1e-6, SCALE_ENVELOPE_HEIGHT_MULT if FLEXIBLE_SCALE_INPAINT else 1.0)
            expected_person_w = insert_w / max(1e-6, SCALE_ENVELOPE_WIDTH_MULT if FLEXIBLE_SCALE_INPAINT else 1.0)
            best_meta = {
                "expected_person_height": expected_person_h,
                "expected_person_width": expected_person_w,
                "insert_width": insert_w,
                "insert_height": insert_h,
                "ground_y": ground_y,
            }
    if best_score <= MIN_ACCEPTED_PLACEMENT_SCORE:
        best_bbox, best_meta = None, None
    if return_metadata:
        return best_bbox, best_meta
    return best_bbox

def feather_mask(size, radius=PATCH_FEATHER_RADIUS):
    width, height = size
    mask = Image.new("L", (width, height), 0)
    inset = max(1, radius)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((inset, inset, width - inset, height - inset), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(radius=radius))


def guide_bboxes_for_variant(insert_bbox, variant):
    x1, y1, x2, y2 = insert_bbox
    width = x2 - x1
    height = y2 - y1
    if variant == "add_two_pedestrians":
        gap = max(6, int(width * GROUP_PERSON_GAP_RATIO))
        person_w = max(22, int((width - gap) / 2))
        left_h = height
        right_h = int(height * (1.0 - GROUP_PERSON_HEIGHT_JITTER))
        return [
            (x1, y2 - left_h, x1 + person_w, y2),
            (x2 - person_w, y2 - right_h, x2, y2),
        ]
    if variant == "add_small_group":
        gap = max(4, int(width * GROUP_PERSON_GAP_RATIO * 0.75))
        person_w = max(18, int((width - 2 * gap) / 3))
        heights = [int(height * 0.90), height, int(height * 0.82)]
        starts = [x1, x1 + person_w + gap, x2 - person_w]
        return [
            (starts[index], y2 - heights[index], starts[index] + person_w, y2)
            for index in range(3)
        ]
    if variant == "add_occluded_pedestrian":
        inset = max(2, int(width * 0.08))
        return [(x1 + inset, y1 + int(height * 0.08), x2 - inset, y2)]
    return [insert_bbox]


def draw_person_guide_on_patch(patch, patch_bbox, insert_bbox, variant):
    if not DRAW_INSERTION_GUIDE:
        return patch
    px1, py1, _, _ = patch_bbox
    overlay = Image.new("RGBA", patch.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for bbox in guide_bboxes_for_variant(insert_bbox, variant):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 -= px1
        x2 -= px1
        y1 -= py1
        y2 -= py1
        bw = max(8, x2 - x1)
        bh = max(24, y2 - y1)
        cx = x1 + bw // 2
        head_r = max(4, int(bw * 0.16))
        head_y = y1 + max(4, int(bh * 0.10))
        shoulder_y = y1 + int(bh * 0.26)
        hip_y = y1 + int(bh * 0.60)
        foot_y = y2
        alpha = int(255 * INSERTION_GUIDE_ALPHA)
        color = (25, 25, 25, alpha)
        outline = (245, 245, 245, max(40, int(alpha * 0.34)))
        draw.ellipse((cx - head_r, head_y, cx + head_r, head_y + 2 * head_r), fill=color)
        draw.rounded_rectangle((cx - int(bw * 0.18), shoulder_y, cx + int(bw * 0.18), hip_y), radius=4, fill=color)
        draw.line((cx - int(bw * 0.10), hip_y, cx - int(bw * 0.22), foot_y), fill=color, width=max(4, int(bw * 0.09)))
        draw.line((cx + int(bw * 0.10), hip_y, cx + int(bw * 0.22), foot_y), fill=color, width=max(4, int(bw * 0.09)))
        draw.line((cx - int(bw * 0.18), shoulder_y + 6, cx - int(bw * 0.30), hip_y - 4), fill=color, width=max(2, int(bw * 0.04)))
        draw.line((cx + int(bw * 0.18), shoulder_y + 6, cx + int(bw * 0.30), hip_y - 4), fill=color, width=max(2, int(bw * 0.04)))
        draw.line((x1 + int(bw * 0.12), foot_y, x2 - int(bw * 0.12), foot_y), fill=outline, width=max(2, int(bw * 0.05)))
        draw.rounded_rectangle((x1, y1, x2, y2), radius=3, outline=outline, width=max(1, int(bw * 0.025)))
    if INSERTION_GUIDE_BLUR:
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=INSERTION_GUIDE_BLUR))
    return Image.alpha_composite(patch.convert("RGBA"), overlay).convert("RGB")


def save_patch_debug_strip(record, variant, seed, source_patch, guided_patch, generated_patch, final_patch, patch_bbox, insert_bbox, debug_index=None):
    if not SAVE_PATCH_DEBUG:
        return ""
    if debug_index is not None and debug_index >= PATCH_DEBUG_MAX_ITEMS:
        return ""
    debug_dir = PATCH_DEBUG_DIR / record.split / record.bucket
    debug_dir.mkdir(parents=True, exist_ok=True)
    panels = [
        ("source", source_patch.convert("RGB")),
        ("guide", guided_patch.convert("RGB")),
        ("generated", generated_patch.convert("RGB")),
        ("final", final_patch.convert("RGB")),
    ]
    width = max(panel.width for _, panel in panels)
    height = max(panel.height for _, panel in panels)
    label_h = 24
    canvas = Image.new("RGB", (width * len(panels), height + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        image = image.resize((width, height))
        x = index * width
        draw.text((x + 8, 6), label, fill=(0, 0, 0))
        canvas.paste(image, (x, label_h))
    safe_variant = variant.replace("/", "_")
    debug_path = debug_dir / f"{record.path.stem}_debug_{seed}_{safe_variant}.png"
    canvas.save(debug_path)
    return str(debug_path)


def human_mask_for_bbox(image_size, insert_bbox, variant):
    width, height = image_size
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    for bbox in guide_bboxes_for_variant(insert_bbox, variant):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1 - HUMAN_MASK_PADDING)
        y1 = max(0, y1 - HUMAN_MASK_PADDING)
        x2 = min(width, x2 + HUMAN_MASK_PADDING)
        y2 = min(height, y2 + HUMAN_MASK_PADDING)
        bw = max(8, x2 - x1)
        bh = max(24, y2 - y1)
        cx = x1 + bw // 2
        head_r = max(4, int(bw * 0.18))
        head_y = y1 + max(2, int(bh * 0.05))
        shoulder_y = y1 + int(bh * 0.24)
        hip_y = y1 + int(bh * 0.60)
        foot_y = y2
        draw.ellipse((cx - head_r, head_y, cx + head_r, head_y + 2 * head_r), fill=255)
        draw.rounded_rectangle((cx - int(bw * 0.28), shoulder_y, cx + int(bw * 0.28), hip_y), radius=6, fill=255)
        draw.line((cx - int(bw * 0.10), hip_y, cx - int(bw * 0.22), foot_y), fill=255, width=max(6, int(bw * 0.14)))
        draw.line((cx + int(bw * 0.10), hip_y, cx + int(bw * 0.22), foot_y), fill=255, width=max(6, int(bw * 0.14)))
        draw.line((cx - int(bw * 0.18), shoulder_y + 4, cx - int(bw * 0.30), hip_y - 4), fill=255, width=max(4, int(bw * 0.09)))
        draw.line((cx + int(bw * 0.18), shoulder_y + 4, cx + int(bw * 0.30), hip_y - 4), fill=255, width=max(4, int(bw * 0.09)))
    if HUMAN_MASK_BLUR_RADIUS:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=HUMAN_MASK_BLUR_RADIUS))
    return mask


def bbox_mask_for_bbox(image_size, insert_bbox, variant=None, padding=BBOX_MASK_PADDING, blur=BBOX_MASK_BLUR_RADIUS):
    width, height = image_size
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    bboxes = guide_bboxes_for_variant(insert_bbox, variant) if variant else [insert_bbox]
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(x1 - padding))
        y1 = max(0, int(y1 - padding))
        x2 = min(width, int(x2 + padding))
        y2 = min(height, int(y2 + padding))
        radius = min(BBOX_MASK_RADIUS, max(2, (x2 - x1) // 5), max(2, (y2 - y1) // 5))
        draw.rounded_rectangle((x1, y1, x2, y2), radius=radius, fill=255)
    if blur:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur))
    return mask


def prepare_inpaint_source(source, mask_image, insert_bbox):
    prepared = source.copy().convert("RGB")
    if PERSON_GENERATION_ONLY_MODE:
        # Keep local road/sidewalk texture so img2img understands where the feet should land.
        context = Image.blend(
            prepared,
            Image.new("RGB", prepared.size, (118, 118, 118)),
            PERSON_GENERATION_CONTEXT_DARKEN,
        )
        blur_source = prepared.filter(ImageFilter.GaussianBlur(radius=6)).convert("RGB")
        neutral = Image.new("RGB", prepared.size, (132, 132, 132))
        target_canvas = Image.blend(blur_source, neutral, PERSON_GENERATION_NEUTRAL_STRENGTH)
        target_canvas = Image.blend(prepared, target_canvas, 0.72)
        prepared = Image.composite(target_canvas, context, mask_image)
        return prepared

    blur_source = source.filter(ImageFilter.GaussianBlur(radius=20)).convert("RGB")
    mask_bbox = mask_image.getbbox() or insert_bbox
    x1, y1, x2, y2 = mask_bbox
    fill_crop = blur_source.crop((x1, y1, x2, y2))
    neutral = Image.new("RGB", fill_crop.size, (132, 132, 132))
    fill_crop = Image.blend(fill_crop, neutral, PERSON_GENERATION_NEUTRAL_STRENGTH)
    prepared.paste(fill_crop, (x1, y1), mask_image.crop((x1, y1, x2, y2)))
    return prepared


def masked_rgb_mae(image_a, image_b, mask_image):
    a = np.asarray(image_a.convert("RGB"), dtype=np.float32) / 255.0
    b = np.asarray(image_b.convert("RGB"), dtype=np.float32) / 255.0
    mask = np.asarray(mask_image.convert("L"), dtype=np.float32) / 255.0
    active = mask > 0.20
    if not np.any(active):
        return 0.0
    return float(np.mean(np.abs(a[active] - b[active])))


def masked_rgb_mae_255(image_a, image_b, mask_image):
    return 255.0 * masked_rgb_mae(image_a, image_b, mask_image)


def mask_area_ratio(mask_image, threshold=0.20):
    mask = np.asarray(mask_image.convert("L"), dtype=np.float32) / 255.0
    return float(np.mean(mask > threshold))


def fill_binary_mask_holes(arr):
    if cv2 is None or not FILL_PERSON_MASK_HOLES:
        return arr
    ys, xs = np.where(arr > 0)
    if len(xs) == 0 or len(ys) == 0:
        return arr
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    roi = arr[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    if h <= 2 or w <= 2:
        return arr

    # Fill only holes enclosed by the foreground silhouette. Padding prevents
    # floodFill from treating background beside a border-touching person as a hole.
    padded = np.pad(roi, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    flood = padded.copy()
    flood_mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
    filled_roi = cv2.bitwise_or(roi, holes)
    out = arr.copy()
    out[y1:y2, x1:x2] = filled_roi
    return out


def clean_binary_person_mask(mask, keep_components=1):
    mask_l = mask.convert("L")
    arr = (np.asarray(mask_l, dtype=np.uint8) >= PERSON_PASTE_HARD_THRESHOLD).astype(np.uint8) * 255
    if cv2 is not None:
        kernel = np.ones((3, 3), np.uint8)
        arr = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel, iterations=2)
        if PERSON_MASK_ERODE_PIXELS > 0:
            arr = cv2.erode(arr, kernel, iterations=PERSON_MASK_ERODE_PIXELS)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((arr > 0).astype(np.uint8), 8)
        if num_labels > 1:
            component_ids = sorted(range(1, num_labels), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
            largest_area = float(stats[component_ids[0], cv2.CC_STAT_AREA]) if component_ids else 0.0
            min_area = max(8.0, largest_area * ACCESSORY_MIN_COMPONENT_AREA_RATIO)
            keep_limit = max(int(keep_components), int(ACCESSORY_KEEP_COMPONENTS))
            keep = {
                component_id
                for component_id in component_ids[:keep_limit]
                if stats[component_id, cv2.CC_STAT_AREA] >= min_area
            }
            if keep:
                arr = np.where(np.isin(labels, list(keep)), 255, 0).astype(np.uint8)
        arr = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel, iterations=1)
        arr = fill_binary_mask_holes(arr)
        return Image.fromarray(arr, mode="L")
    hard = Image.fromarray(arr, mode="L")
    if PERSON_MASK_ERODE_PIXELS > 0:
        hard = hard.filter(ImageFilter.MinFilter(PERSON_MASK_ERODE_PIXELS * 2 + 1))
    return hard


def prepare_person_paste_mask(person_mask, size):
    mask = person_mask.resize(size, Image.NEAREST).convert("L")
    mask = clean_binary_person_mask(mask, keep_components=ACCESSORY_KEEP_COMPONENTS)
    trim_px = max(0, int(PERSON_MASK_TRIM_FRINGE_PIXELS))
    if trim_px > 0:
        # Remove the generated-background fringe at the silhouette boundary,
        # then restore one soft pixel so accessories are not aggressively cut.
        eroded = mask.filter(ImageFilter.MinFilter(trim_px * 2 + 1))
        mask = eroded.filter(ImageFilter.MaxFilter(trim_px * 2 + 1))
    if PERSON_PASTE_FEATHER_RADIUS and PERSON_PASTE_FEATHER_RADIUS > 0:
        soft = mask.filter(ImageFilter.GaussianBlur(radius=PERSON_PASTE_FEATHER_RADIUS))
        mask = ImageChops.multiply(mask, soft)
    return mask


def constrain_mask_to_bbox(mask, bbox, padding_ratio=0.12, min_padding=6):
    bbox_mask = Image.new("L", mask.size, 0)
    draw = ImageDraw.Draw(bbox_mask)
    x1, y1, x2, y2 = bbox
    pad = max(min_padding, int(round(max(x2 - x1, y2 - y1) * padding_ratio)))
    padded = clamp_bbox((x1 - pad, y1 - pad, x2 + pad, y2 + pad), mask.size[0], mask.size[1])
    draw.rectangle(tuple(int(round(v)) for v in padded), fill=255)
    return ImageChops.multiply(mask.convert("L"), bbox_mask)


def mask_outside_bbox_ratio(mask, bbox, padding_ratio=0.12, min_padding=6):
    mask_l = mask.convert("L")
    mask_arr = np.asarray(mask_l, dtype=np.float32) / 255.0
    active = mask_arr > 0.20
    if not np.any(active):
        return 1.0
    x1, y1, x2, y2 = bbox
    pad = max(min_padding, int(round(max(x2 - x1, y2 - y1) * padding_ratio)))
    px1, py1, px2, py2 = clamp_bbox((x1 - pad, y1 - pad, x2 + pad, y2 + pad), mask_l.size[0], mask_l.size[1])
    keep = np.zeros(active.shape, dtype=bool)
    keep[int(py1):int(py2), int(px1):int(px2)] = True
    return float(np.mean(active & ~keep) / max(1e-6, np.mean(active)))


def expand_bbox_for_detection(bbox, image_size, factor=CONTEXT_TARGET_BBOX_EXPAND_FOR_DETECTION):
    width, height = image_size
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    bw = (x2 - x1) * factor
    bh = (y2 - y1) * factor
    return clamp_bbox((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), width, height)

DEPTH_PIPELINE = None

def load_depth_pipeline(device="cpu"):
    global DEPTH_PIPELINE
    if DEPTH_PIPELINE is not None:
        return DEPTH_PIPELINE
    try:
        from transformers import pipeline
        DEPTH_PIPELINE = pipeline(
            task="depth-estimation",
            model="LiheYoung/depth-anything-small-hf",
            device=device if "cuda" in str(device) else "cpu"
        )
        print("Loaded depth estimator model: LiheYoung/depth-anything-small-hf")
    except Exception as exc:
        print("Failed to load depth estimator:", type(exc).__name__, exc)
        DEPTH_PIPELINE = None
    return DEPTH_PIPELINE

def estimate_depth_map(image, device="cpu"):
    pipe = load_depth_pipeline(device)
    if pipe is None:
        return None
    try:
        result = pipe(image)
        depth_pil = result["depth"]
        depth_pil = depth_pil.resize(image.size, Image.Resampling.BILINEAR)
        depth_arr = np.asarray(depth_pil, dtype=np.float32) / 255.0
        return depth_arr
    except Exception as exc:
        print("Depth estimation failed:", type(exc).__name__, exc)
        return None

def expected_person_height_from_depth(foot_x, foot_y, depth_map=None, image_height=RESOLUTION, variant=None, jitter=0.0):
    if depth_map is None:
        return expected_person_height_from_ground_y(foot_y, image_height, variant, jitter)
    
    h, w = depth_map.shape
    fx = int(np.clip(foot_x, 0, w - 1))
    fy = int(np.clip(foot_y, 0, h - 1))
    
    pad = 2
    x_min = max(0, fx - pad)
    x_max = min(w, fx + pad + 1)
    y_min = max(0, fy - pad)
    y_max = min(h, fy + pad + 1)
    
    disparity = float(np.mean(depth_map[y_min:y_max, x_min:x_max]))
    
    base_ratio = 0.075 + disparity * (0.29 - 0.075)
    base_ratio = float(np.clip(base_ratio, 0.075, 0.29))
    
    variant_name = variant or "add_single_pedestrian"
    variant_multiplier = 1.0
    if "distant" in variant_name or "small_group" in variant_name:
        variant_multiplier = 0.82
    elif "near" in variant_name:
        variant_multiplier = 1.13
    elif "two" in variant_name:
        variant_multiplier = 0.96
    base_ratio *= max(variant_multiplier, PERSON_TARGET_SCALE_MULTIPLIER)
    if variant_name in {"add_two_pedestrians", "add_small_group"}:
        base_ratio *= 1.0 + float(jitter)
        
    return max(28.0, base_ratio * image_height)

def expected_person_height_from_ground_y(foot_y, image_height=RESOLUTION, variant=None, jitter=0.0):
    ground_y_norm = float(foot_y) / max(1.0, float(image_height))
    base_ratio = 0.065 + (ground_y_norm - 0.55) * 0.58
    base_ratio = float(np.clip(base_ratio, 0.075, 0.29))
    variant_name = variant or "add_single_pedestrian"
    variant_multiplier = 1.0
    if "distant" in variant_name or "small_group" in variant_name:
        variant_multiplier = 0.82
    elif "near" in variant_name:
        variant_multiplier = 1.13
    elif "two" in variant_name:
        variant_multiplier = 0.96
    base_ratio *= max(variant_multiplier, PERSON_TARGET_SCALE_MULTIPLIER)
    if variant_name in {"add_two_pedestrians", "add_small_group"}:
        base_ratio *= 1.0 + float(jitter)
    return max(28.0, base_ratio * image_height)


def default_scale_correction_metadata():
    return {
        "expected_person_height": "",
        "detected_person_height": "",
        "scale_ratio_before_correction": "",
        "expected_height": "",
        "detected_height": "",
        "scale_ratio_before": "",
        "scale_corrected": False,
        "resized_person_height": "",
        "resized_person_width": "",
        "scale_correction_status": "none",
        "seamless_clone_used": False,
        "fallback_alpha_paste": False,
        "fallback_alpha_used": False,
        "retry_attempts": 0,
        "last_reject_reason": "",
        "reject_reason": "",
    }


def scale_jitter_for_person(variant, index):
    if not GROUP_SCALE_JITTER_ENABLED:
        return 0.0
    if variant not in {"add_two_pedestrians", "add_small_group"}:
        return 0.0
    return (-0.04, 0.0, 0.04, -0.02, 0.02)[index % 5]


def enforce_monotonic_perspective_heights(items):
    if not PERSPECTIVE_MONOTONIC_SCALE_ENABLED or len(items) < 2:
        return items
    ordered = sorted(items, key=lambda item: item["foot_y"])
    prev_height = None
    for item in ordered:
        if prev_height is not None:
            item["expected_height"] = max(item["expected_height"], prev_height * PERSPECTIVE_MONOTONIC_MIN_RATIO)
        prev_height = item["expected_height"]
    return items


def scale_correction_policy(scale_ratio, person_conf=1.0, mask_area=1.0):
    if SCALE_CORRECTION_SOFT_MIN <= scale_ratio <= SCALE_CORRECTION_SOFT_MAX:
        return "recoverable"
    if scale_ratio < SCALE_CORRECTION_HARD_MIN or scale_ratio > SCALE_CORRECTION_HARD_MAX:
        return "unrecoverable"
    if (
        not SCALE_CORRECTION_BORDERLINE_RETRY
        or (person_conf >= SCALE_CORRECTION_BORDERLINE_MIN_CONF and mask_area >= SCALE_CORRECTION_BORDERLINE_MIN_MASK_AREA_RATIO)
    ):
        return "borderline_accept"
    return "borderline_retry"


def corrected_person_layers(generated_image, detections, variant, depth_map=None):
    width, height = generated_image.size
    margin = int(MIN_BORDER_MARGIN_RATIO * max(width, height))
    corrected_rgb = Image.new("RGB", generated_image.size, (0, 0, 0))
    combined_mask = Image.new("L", generated_image.size, 0)
    corrected_bboxes = []
    meta = default_scale_correction_metadata()
    per_person_meta = []
    keep_components = max(1, len(detections))
    planned_items = []
    for person_order, detection in enumerate(detections):
        det_bbox = detection["bbox"]
        dx1, dy1, dx2, dy2 = [int(round(v)) for v in det_bbox]
        dx1, dy1 = max(0, dx1), max(0, dy1)
        dx2, dy2 = min(width, dx2), min(height, dy2)
        foot_y = float(dy2)
        foot_x = (float(dx1) + float(dx2)) / 2.0
        planned_items.append({
            "detection": detection,
            "order": person_order,
            "foot_y": foot_y,
            "foot_x": foot_x,
            "expected_height": expected_person_height_from_depth(
                foot_x,
                foot_y,
                depth_map,
                image_height=height,
                variant=variant,
                jitter=scale_jitter_for_person(variant, person_order),
            ),
        })
    planned_items = enforce_monotonic_perspective_heights(planned_items)
    for item in planned_items:
        detection = item["detection"]
        person_order = item["order"]
        det_bbox = detection["bbox"]
        x1, y1, x2, y2 = [int(round(v)) for v in det_bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        detected_height = max(1.0, float(y2 - y1))
        detected_width = max(1.0, float(x2 - x1))
        target_foot_y = float(y2)
        target_center_x = (float(x1) + float(x2)) / 2.0
        expected_height = item["expected_height"]
        scale_ratio = expected_height / max(1.0, detected_height)
        raw_mask = detection["mask"].resize(generated_image.size, Image.NEAREST)
        mask_bbox = raw_mask.getbbox()
        if mask_bbox is not None:
            pad = PERSON_CROP_MASK_PADDING
            mx1, my1, mx2, my2 = [int(round(v)) for v in mask_bbox]
            x1 = max(0, min(x1, mx1 - pad))
            y1 = max(0, min(y1, my1 - pad))
            x2 = min(width, max(x2, mx2 + pad))
            y2 = min(height, max(y2, my2 + pad))
        mask_area = mask_area_ratio(raw_mask)
        status = scale_correction_policy(scale_ratio, detection.get("conf", 1.0), mask_area)
        if status == "unrecoverable":
            meta.update({
                "expected_person_height": round(expected_height, 2),
                "detected_person_height": round(detected_height, 2),
                "scale_ratio_before_correction": round(scale_ratio, 4),
                "expected_height": round(expected_height, 2),
                "detected_height": round(detected_height, 2),
                "scale_ratio_before": round(scale_ratio, 4),
                "scale_correction_status": "unrecoverable",
                "last_reject_reason": "scale_unrecoverable",
                "reject_reason": "scale_unrecoverable",
            })
            return None, None, None, meta, "scale_unrecoverable"
        if status == "borderline_retry":
            meta.update({
                "expected_person_height": round(expected_height, 2),
                "detected_person_height": round(detected_height, 2),
                "scale_ratio_before_correction": round(scale_ratio, 4),
                "expected_height": round(expected_height, 2),
                "detected_height": round(detected_height, 2),
                "scale_ratio_before": round(scale_ratio, 4),
                "scale_correction_status": "borderline_retry",
                "last_reject_reason": "scale_unrecoverable",
                "reject_reason": "scale_unrecoverable",
            })
            return None, None, None, meta, "scale_unrecoverable"

        resized_height = max(8, int(round(detected_height * scale_ratio)))
        resized_width = max(4, int(round(detected_width * scale_ratio)))
        new_y2 = int(round(target_foot_y))
        new_y1 = new_y2 - resized_height
        new_x1 = int(round(target_center_x - resized_width / 2.0))
        new_x2 = new_x1 + resized_width
        if new_x1 < margin or new_x2 > width - margin or new_y1 < margin or new_y2 > height - margin:
            meta.update({
                "expected_person_height": round(expected_height, 2),
                "detected_person_height": round(detected_height, 2),
                "scale_ratio_before_correction": round(scale_ratio, 4),
                "expected_height": round(expected_height, 2),
                "detected_height": round(detected_height, 2),
                "scale_ratio_before": round(scale_ratio, 4),
                "resized_person_height": resized_height,
                "resized_person_width": resized_width,
                "scale_correction_status": "border_reject_after_resize",
                "last_reject_reason": "scale_unrecoverable",
                "reject_reason": "scale_unrecoverable",
            })
            return None, None, None, meta, "scale_unrecoverable"
        corrected_aspect = resized_height / max(1.0, resized_width)
        corrected_area_ratio = (resized_height * resized_width) / max(1.0, width * height)
        if (
            corrected_aspect < MIN_CORRECTED_ASPECT_RATIO
            or corrected_aspect > MAX_CORRECTED_ASPECT_RATIO
            or corrected_area_ratio < MIN_CORRECTED_PERSON_AREA_RATIO
        ):
            meta.update({
                "expected_person_height": round(expected_height, 2),
                "detected_person_height": round(detected_height, 2),
                "scale_ratio_before_correction": round(scale_ratio, 4),
                "expected_height": round(expected_height, 2),
                "detected_height": round(detected_height, 2),
                "scale_ratio_before": round(scale_ratio, 4),
                "resized_person_height": resized_height,
                "resized_person_width": resized_width,
                "scale_correction_status": "bad_scale_geometry_after_resize",
                "last_reject_reason": "scale_unrecoverable",
                "reject_reason": "scale_unrecoverable",
            })
            return None, None, None, meta, "scale_unrecoverable"

        crop_w = max(1, x2 - x1)
        crop_h = max(1, y2 - y1)
        scale_x = resized_width / max(1.0, detected_width)
        scale_y = resized_height / max(1.0, detected_height)
        resized_crop_w = max(4, int(round(crop_w * scale_x)))
        resized_crop_h = max(8, int(round(crop_h * scale_y)))
        crop_offset_x = int(round((x1 - int(round(det_bbox[0]))) * scale_x))
        crop_offset_y = int(round((y1 - int(round(det_bbox[1]))) * scale_y))
        paste_x = new_x1 + crop_offset_x
        paste_y = new_y1 + crop_offset_y
        person_crop = generated_image.crop((x1, y1, x2, y2)).resize((resized_crop_w, resized_crop_h), Image.LANCZOS)
        mask_crop = raw_mask.crop((x1, y1, x2, y2)).resize((resized_crop_w, resized_crop_h), Image.NEAREST)
        mask_crop = clean_binary_person_mask(mask_crop, keep_components=ACCESSORY_KEEP_COMPONENTS)
        corrected_rgb.paste(person_crop, (paste_x, paste_y), mask_crop)
        combined_mask.paste(mask_crop, (paste_x, paste_y), mask_crop)
        corrected_bboxes.append((new_x1, new_y1, new_x2, new_y2))
        per_person_meta.append({
            "expected_person_height": round(expected_height, 2),
            "detected_person_height": round(detected_height, 2),
            "scale_ratio_before_correction": round(scale_ratio, 4),
            "expected_height": round(expected_height, 2),
            "detected_height": round(detected_height, 2),
            "scale_ratio_before": round(scale_ratio, 4),
            "scale_corrected": abs(scale_ratio - 1.0) > 0.05,
            "resized_person_height": resized_height,
            "resized_person_width": resized_width,
            "scale_correction_status": "corrected" if abs(scale_ratio - 1.0) > 0.05 else "none",
        })

    if not corrected_bboxes:
        meta["last_reject_reason"] = "no_person_detected"
        return None, None, None, meta, "no_person_detected"
    any_corrected = any(item["scale_corrected"] for item in per_person_meta)
    meta.update({
        "expected_person_height": json.dumps([item["expected_person_height"] for item in per_person_meta]),
        "detected_person_height": json.dumps([item["detected_person_height"] for item in per_person_meta]),
        "scale_ratio_before_correction": json.dumps([item["scale_ratio_before_correction"] for item in per_person_meta]),
        "expected_height": json.dumps([item["expected_height"] for item in per_person_meta]),
        "detected_height": json.dumps([item["detected_height"] for item in per_person_meta]),
        "scale_ratio_before": json.dumps([item["scale_ratio_before"] for item in per_person_meta]),
        "scale_corrected": any_corrected,
        "resized_person_height": json.dumps([item["resized_person_height"] for item in per_person_meta]),
        "resized_person_width": json.dumps([item["resized_person_width"] for item in per_person_meta]),
        "scale_correction_status": "corrected" if any_corrected else "none",
    })
    combined_mask = clean_binary_person_mask(combined_mask, keep_components=max(keep_components, ACCESSORY_KEEP_COMPONENTS))
    return corrected_rgb, combined_mask, union_bboxes(corrected_bboxes), meta, "ok"


def min_accepted_person_height_ratio(variant, ground_y=None, resolution=RESOLUTION):
    if variant == "add_distant_pedestrian":
        return MIN_ACCEPTED_DISTANT_HEIGHT_RATIO
    if variant == "add_near_pedestrian":
        base = MIN_ACCEPTED_NEAR_HEIGHT_RATIO
    else:
        base = MIN_ACCEPTED_SINGLE_HEIGHT_RATIO
    if ground_y is not None:
        y_far = resolution * PATCH_ROAD_Y_RANGE[0]
        y_near = resolution * PATCH_ROAD_Y_RANGE[1]
        t = max(0.0, min(1.0, (ground_y - y_far) / max(1.0, y_near - y_far)))
        perspective_min = (
            MIN_ACCEPTED_PERSPECTIVE_HEIGHT_RATIO_FAR
            + t * (MIN_ACCEPTED_PERSPECTIVE_HEIGHT_RATIO_NEAR - MIN_ACCEPTED_PERSPECTIVE_HEIGHT_RATIO_FAR)
        )
        base = max(base, perspective_min)
        if variant != "add_distant_pedestrian" and ground_y / max(1, resolution) >= 0.72:
            base = max(base, MIN_ACCEPTED_FOREGROUND_HEIGHT_RATIO)
    return base


def max_accepted_person_height_ratio(variant, ground_y=None, resolution=RESOLUTION):
    if variant == "add_distant_pedestrian":
        return 0.27
    if ground_y is None:
        return 0.62 if variant == "add_near_pedestrian" else 0.52
    y_far = resolution * PATCH_ROAD_Y_RANGE[0]
    y_near = resolution * PATCH_ROAD_Y_RANGE[1]
    t = max(0.0, min(1.0, (ground_y - y_far) / max(1.0, y_near - y_far)))
    far_max = 0.36
    near_max = 0.62 if variant == "add_near_pedestrian" else 0.52
    return far_max + t * (near_max - far_max)


def normalize_expected_person_height(expected_person_height):
    if expected_person_height is None or expected_person_height == "":
        return None
    if isinstance(expected_person_height, (int, float)):
        return float(expected_person_height)
    if isinstance(expected_person_height, str):
        try:
            expected_person_height = json.loads(expected_person_height)
        except Exception:
            try:
                return float(expected_person_height)
            except Exception:
                return None
    if isinstance(expected_person_height, (list, tuple)):
        values = []
        for item in expected_person_height:
            try:
                values.append(float(item))
            except Exception:
                continue
        return max(values) if values else None
    return None


def validate_pasted_person_mask(pasted_mask, variant, insert_bbox, resolution=RESOLUTION, expected_person_height=None):
    bbox = pasted_mask.getbbox()
    if bbox is None:
        raise RuntimeError("Accepted mask is empty after paste.")
    x1, y1, x2, y2 = bbox
    if REJECT_IF_MASK_TOUCHES_BORDER and mask_bbox_touches_border(bbox, pasted_mask.size, margin=PERSON_BORDER_REJECT_PIXELS):
        print("Final person mask is close to image border; accepting with relaxed limb-preservation policy.")
    mask_h = y2 - y1
    mask_w = max(1, x2 - x1)
    final_aspect = mask_h / mask_w
    relaxed_max_aspect = max(MAX_PERSON_MASK_ASPECT_RATIO, 6.0)
    relaxed_min_aspect = min(MIN_PERSON_MASK_ASPECT_RATIO, 1.15)
    if final_aspect < relaxed_min_aspect or final_aspect > relaxed_max_aspect:
        raise RuntimeError(f"Accepted person final mask aspect invalid/slim (aspect={final_aspect:.2f}).")
    ground_y = insert_bbox[3] if insert_bbox is not None else y2
    expected_h = normalize_expected_person_height(expected_person_height)
    if FINAL_SCALE_VALIDATION_ENABLED:
        if expected_h is not None and expected_h > 1:
            scale_ratio = mask_h / max(1.0, expected_h)
            if scale_ratio < FINAL_MIN_SCALE_RATIO or scale_ratio > FINAL_MAX_SCALE_RATIO:
                raise RuntimeError(
                    f"Accepted person scale mismatch on full image (scale_ratio={scale_ratio:.2f}, "
                    f"mask_h={mask_h:.1f}, expected={expected_h:.1f})."
                )
        else:
            min_h = resolution * min_accepted_person_height_ratio(variant, ground_y=ground_y, resolution=resolution)
            if mask_h < min_h:
                raise RuntimeError(f"Accepted person is visually too small (mask_h={mask_h:.1f}, min_h={min_h:.1f}).")
    mask_arr = np.asarray(pasted_mask.convert("L"), dtype=np.float32) / 255.0
    active = mask_arr > 0.04
    if np.any(active):
        opaque_ratio = float(np.mean(mask_arr[active] > 0.72))
        if opaque_ratio < MIN_ACCEPTED_MASK_OPAQUE_RATIO:
            raise RuntimeError(f"Accepted person mask is too soft/transparent (opaque_ratio={opaque_ratio:.2f}).")
