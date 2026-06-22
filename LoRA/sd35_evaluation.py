"""SD3.5 CityPersons augmentation: segmentation, validation, and retry policy."""

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
from sd35_utils import *
from sd35_metrics import affordance_reject_reason, compute_affordance_score

PERSON_SEGMENTERS = {}

def _segmenter_device_key(device=None):
    if device is None:
        if torch.cuda.is_available():
            return str(torch.cuda.current_device())
        return "cpu"
    return str(device)


def load_person_segmenter(device=None):
    device_key = _segmenter_device_key(device)
    if PERSON_SEGMENTERS.get(device_key) is False:
        return None
    if device_key in PERSON_SEGMENTERS:
        return PERSON_SEGMENTERS[device_key]
    try:
        from ultralytics import YOLO
        segmenter = YOLO(CONTEXT_PERSON_SEGMENTATION_MODEL)
        PERSON_SEGMENTERS[device_key] = segmenter
        print(f"Loaded person segmentation model on {device_key}: {CONTEXT_PERSON_SEGMENTATION_MODEL}")
        return segmenter
    except Exception as exc:
        PERSON_SEGMENTERS[device_key] = False
        print("Person segmentation unavailable; context_person_composite will use fallback if enabled.")
        print(type(exc).__name__, exc)
        return None


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    union = max(1, bbox_area(a) + bbox_area(b) - inter)
    return inter / union


def mask_bbox_from_array(mask_array, threshold=0.5):
    ys, xs = np.where(mask_array > threshold)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def mask_bbox_touches_border(mask_bbox, size, margin=PERSON_BORDER_REJECT_PIXELS):
    if mask_bbox is None:
        return True
    width, height = size
    x1, y1, x2, y2 = mask_bbox
    return x1 <= margin or y1 <= margin or x2 >= width - margin or y2 >= height - margin


def expected_new_person_count(variant):
    return {
        "add_two_pedestrians": 2,
        "add_small_group": 3,
    }.get(variant, 1)


def union_bboxes(bboxes):
    if not bboxes:
        return None
    return (
        min(float(bbox[0]) for bbox in bboxes),
        min(float(bbox[1]) for bbox in bboxes),
        max(float(bbox[2]) for bbox in bboxes),
        max(float(bbox[3]) for bbox in bboxes),
    )


def person_mask_completeness_ok(raw_mask, raw_mask_bbox, det_bbox, target_bbox):
    if raw_mask_bbox is None:
        return False, "empty mask bbox"
    mask_w = max(1.0, raw_mask_bbox[2] - raw_mask_bbox[0])
    mask_h = max(1.0, raw_mask_bbox[3] - raw_mask_bbox[1])
    det_h = max(1.0, det_bbox[3] - det_bbox[1])
    target_h = max(1.0, target_bbox[3] - target_bbox[1])
    if mask_h / det_h < MIN_PERSON_MASK_DET_HEIGHT_RATIO:
        return False, f"mask covers too little detection height ({mask_h / det_h:.2f})"
    if mask_h / target_h < MIN_PERSON_MASK_TARGET_HEIGHT_RATIO:
        return False, f"mask covers too little target height ({mask_h / target_h:.2f})"
    aspect = mask_h / mask_w
    if aspect < MIN_PERSON_MASK_ASPECT_RATIO or aspect > MAX_PERSON_MASK_ASPECT_RATIO:
        return False, f"partial body mask aspect ({aspect:.2f})"
    x1, y1, x2, y2 = [int(round(v)) for v in det_bbox]
    height, width = raw_mask.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    crop = raw_mask[y1:y2, x1:x2] > CONTEXT_PERSON_MASK_THRESHOLD
    if crop.size == 0:
        return False, "empty detection crop"
    bands = np.array_split(crop, 3, axis=0)
    row_coverages = [float(np.mean(np.any(band, axis=1))) if band.size else 0.0 for band in bands]
    top_mid_ok = min(row_coverages[:2]) >= MIN_PERSON_MASK_VERTICAL_BAND_COVERAGE
    lower_leg_floor = max(0.06, MIN_PERSON_MASK_VERTICAL_BAND_COVERAGE * 0.45)
    lower_ok = row_coverages[2] >= lower_leg_floor
    if not (top_mid_ok and lower_ok):
        return False, f"partial body vertical coverage {row_coverages}"
    return True, "ok"


def select_generated_person_mask(generated_crop, target_bbox, device=None):
    segmenter = load_person_segmenter(device=device)
    if segmenter is None:
        return None, None
    try:
        results = segmenter.predict(generated_crop, imgsz=RESOLUTION, conf=CONTEXT_PERSON_MIN_CONFIDENCE, device=device, verbose=False)
    except Exception as exc:
        print("Person segmentation failed; using fallback if enabled.")
        print(type(exc).__name__, exc)
        return None, None
    if not results:
        return None, None
    result = results[0]
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or masks is None or boxes.xyxy is None or masks.data is None:
        return None, None
    detection_target_bbox = expand_bbox_for_detection(target_bbox, generated_crop.size)
    target_cx = (target_bbox[0] + target_bbox[2]) / 2
    target_cy = (target_bbox[1] + target_bbox[3]) / 2
    target_h = max(1.0, target_bbox[3] - target_bbox[1])
    best_index = None
    best_score = -1e9
    xyxy = boxes.xyxy.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxy))
    conf = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
    for index, box in enumerate(xyxy):
        if int(cls[index]) != 0:
            continue
        raw_mask = masks.data[index].detach().cpu().numpy()
        raw_mask_bbox = mask_bbox_from_array(raw_mask, threshold=CONTEXT_PERSON_MASK_THRESHOLD)
        if REJECT_IF_MASK_TOUCHES_BORDER and mask_bbox_touches_border(raw_mask_bbox, generated_crop.size, margin=max(1, PERSON_BORDER_REJECT_PIXELS // 2)):
            print("Detected person mask is near crop border; keeping candidate for relaxed full-body validation.")
        det_bbox = tuple(float(v) for v in box)
        complete, reason = person_mask_completeness_ok(raw_mask, raw_mask_bbox, det_bbox, target_bbox)
        if not complete:
            print(f"Detected person mask looks partial ({reason}); retrying.")
            continue
        det_h = max(1.0, det_bbox[3] - det_bbox[1])
        height_ratio = det_h / target_h
        if STRICT_EARLY_PERSON_SCALE_FILTER and (height_ratio < MIN_GENERATED_HEIGHT_RATIO or height_ratio > MAX_GENERATED_HEIGHT_RATIO):
            print(f"Detected person scale mismatch (height_ratio={height_ratio:.2f}); retrying.")
            continue
        mask_h = 0.0 if raw_mask_bbox is None else raw_mask_bbox[3] - raw_mask_bbox[1]
        mask_height_ratio = mask_h / target_h
        if STRICT_EARLY_PERSON_SCALE_FILTER and mask_height_ratio < MIN_MASK_BBOX_HEIGHT_RATIO:
            print(f"Detected person mask height too small (ratio={mask_height_ratio:.2f}); retrying.")
            continue
        det_cx = (det_bbox[0] + det_bbox[2]) / 2
        det_cy = (det_bbox[1] + det_bbox[3]) / 2
        overlap = bbox_iou(det_bbox, detection_target_bbox)
        target_overlap = bbox_intersection_area(det_bbox, detection_target_bbox) / max(1, bbox_area(det_bbox))
        if overlap <= 0 and target_overlap < CONTEXT_MIN_PERSON_TARGET_OVERLAP:
            continue
        distance = math.hypot(det_cx - target_cx, det_cy - target_cy) / RESOLUTION
        score = 3.0 * overlap + 1.5 * target_overlap + float(conf[index]) - distance
        if score > best_score:
            best_index = index
            best_score = score
    if best_index is None:
        return None, None
    mask_array = masks.data[best_index].detach().cpu().numpy()
    mask = Image.fromarray((mask_array > CONTEXT_PERSON_MASK_THRESHOLD).astype(np.uint8) * 255, mode="L")
    mask = mask.resize(generated_crop.size, Image.NEAREST)
    area_ratio = mask_area_ratio(mask)
    if area_ratio < CONTEXT_MIN_PERSON_MASK_AREA_RATIO:
        print(f"Detected person mask is too small (area_ratio={area_ratio:.5f}); rejecting as ghost/unchanged.")
        return None, None
    return mask, tuple(float(v) for v in xyxy[best_index])


def select_new_generated_person_mask(generated_image, existing_person_bboxes=None, semantic_masks=None, variant=None, background_image=None, depth_map=None, device=None):
    segmenter = load_person_segmenter(device=device)
    if segmenter is None:
        return None, None, "segmenter_unavailable", default_scale_correction_metadata(), None
    try:
        results = segmenter.predict(generated_image, imgsz=RESOLUTION, conf=CONTEXT_PERSON_MIN_CONFIDENCE, device=device, verbose=False)
    except Exception as exc:
        print("Person segmentation failed; retrying if possible.")
        print(type(exc).__name__, exc)
        meta = default_scale_correction_metadata()
        meta["last_reject_reason"] = "segmenter_failed"
        return None, None, "segmenter_failed", meta, None
    if not results:
        meta = default_scale_correction_metadata()
        meta["last_reject_reason"] = "no_person_detected"
        return None, None, "no_person_detected", meta, None
    result = results[0]
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or masks is None or boxes.xyxy is None or masks.data is None:
        meta = default_scale_correction_metadata()
        meta["last_reject_reason"] = "no_person_detected"
        return None, None, "no_person_detected", meta, None

    existing_person_bboxes = existing_person_bboxes or []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxy))
    conf = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
    min_person_conf = MIN_PERSON_CONF_BY_VARIANT.get(variant, MIN_RETRY_PERSON_CONFIDENCE)
    expected_count = expected_new_person_count(variant or "add_single_pedestrian")
    candidates = []
    best_reject_reason = "no_person_detected"
    for index, box in enumerate(xyxy):
        if int(cls[index]) != 0:
            continue
        person_conf = float(conf[index])
        if person_conf < min_person_conf:
            best_reject_reason = "low_person_conf"
            print(f"Detected new person confidence too low (conf={person_conf:.2f}, min={min_person_conf:.2f}); retrying.")
            continue
        det_bbox = tuple(float(v) for v in box)
        det_area = max(1, bbox_area(det_bbox))
        old_overlap = 0.0
        old_min_overlap = 0.0
        old_iou = 0.0
        bad_person_depth_overlap = False
        for old_bbox in existing_person_bboxes:
            overlap = bbox_overlap_ratios(det_bbox, old_bbox)
            old_overlap = max(old_overlap, overlap["a"])
            old_min_overlap = max(old_min_overlap, overlap["min"])
            old_iou = max(old_iou, bbox_iou(det_bbox, old_bbox))
            depth_ok, _overlap = person_overlap_depth_ok(det_bbox, old_bbox)
            if not depth_ok:
                bad_person_depth_overlap = True
        if bad_person_depth_overlap:
            best_reject_reason = "bad_person_depth_overlap"
            continue
        if old_min_overlap > MAX_PERSON_PERSON_OVERLAP_RATIO or old_iou > 0.06:
            best_reject_reason = "bad_person_depth_overlap"
            continue
        if (not ALLOW_PERSON_PERSON_OVERLAP) and (old_overlap > 0.18 or old_iou > 0.08):
            continue

        det_h = max(1.0, det_bbox[3] - det_bbox[1])
        det_w = max(1.0, det_bbox[2] - det_bbox[0])
        if det_h < 18 or det_w < 5:
            best_reject_reason = "too_small_or_ghost_person"
            continue

        foot_score = 0.0
        body_valid_score = 0.0
        avoid_score = 0.0
        if semantic_masks:
            valid_mask = semantic_masks.get("valid")
            avoid_mask = semantic_masks.get("avoid")
            foot_bbox = foot_support_bbox(det_bbox)
            foot_score = mask_coverage(valid_mask, foot_bbox)
            foot_avoid_score = mask_coverage(avoid_mask, foot_bbox)
            body_valid_score = mask_coverage(valid_mask, det_bbox)
            avoid_score = mask_coverage(avoid_mask, det_bbox)
            if foot_score < MIN_FOOT_SUPPORT:
                best_reject_reason = "floating_or_bad_ground"
                continue
            if foot_avoid_score > MAX_FOOT_AVOID_SUPPORT:
                best_reject_reason = "floating_or_bad_ground"
                continue
        else:
            ground_y_ratio = det_bbox[3] / max(1, generated_image.size[1])
            if ground_y_ratio < PATCH_ROAD_Y_RANGE[0] - 0.08 or ground_y_ratio > PATCH_ROAD_Y_RANGE[1] + 0.06:
                best_reject_reason = "floating_or_bad_ground"
                continue

        raw_mask = masks.data[index].detach().cpu().numpy()
        raw_mask_bbox = mask_bbox_from_array(raw_mask, threshold=CONTEXT_PERSON_MASK_THRESHOLD)
        if raw_mask_bbox is None:
            best_reject_reason = "too_small_or_ghost_person"
            continue
        if REJECT_IF_MASK_TOUCHES_BORDER and mask_bbox_touches_border(raw_mask_bbox, generated_image.size):
            best_reject_reason = "partial_or_cropped_body"
            print("Detected new person mask touches image border; rejecting likely cropped/oversized body.")
            continue
        mask_h = raw_mask_bbox[3] - raw_mask_bbox[1]
        if mask_h < 0.45 * det_h:
            best_reject_reason = "partial_or_cropped_body"
            continue

        mask_array = raw_mask > CONTEXT_PERSON_MASK_THRESHOLD
        mask = Image.fromarray(mask_array.astype(np.uint8) * 255, mode="L").resize(generated_image.size, Image.NEAREST)
        det_cx = (det_bbox[0] + det_bbox[2]) / 2.0
        expected_h = expected_person_height_from_depth(det_cx, det_bbox[3], depth_map, generated_image.size[1], variant=variant)
        scale_ratio = expected_h / max(1.0, det_h)
        policy = scale_correction_policy(scale_ratio, person_conf, mask_area_ratio(mask))
        if policy == "unrecoverable":
            best_reject_reason = "scale_unrecoverable"
            continue
        if policy == "borderline_retry":
            best_reject_reason = "scale_unrecoverable"
            continue

        distance_center = abs(((det_bbox[0] + det_bbox[2]) / 2) / max(1, generated_image.size[0]) - 0.5)
        scale_score = 1.0 - min(1.0, abs(math.log(max(scale_ratio, 1e-6))))
        score = person_conf + 2.2 * foot_score + 0.8 * body_valid_score + 0.45 * scale_score - 2.8 * avoid_score - 0.2 * distance_center
        candidates.append((score, index, det_bbox, mask, person_conf))
    if not candidates:
        meta = default_scale_correction_metadata()
        meta["last_reject_reason"] = best_reject_reason
        return None, None, best_reject_reason, meta, None
    sorted_candidates = sorted(candidates, key=lambda item: item[0], reverse=True)
    selected = sorted_candidates[:expected_count]
    if KEEP_EXTRA_GENERATED_PEOPLE and len(sorted_candidates) > expected_count:
        best_score = sorted_candidates[0][0]
        extra_limit = expected_count + MAX_EXTRA_GENERATED_PEOPLE
        for candidate in sorted_candidates[expected_count:extra_limit]:
            if best_score - candidate[0] <= EXTRA_PERSON_MIN_SCORE_DELTA:
                selected.append(candidate)
    if len(selected) < expected_count:
        if variant == "add_small_group" and ALLOW_PARTIAL_SMALL_GROUP and len(selected) >= 2:
            print(f"Accepting partial small group with {len(selected)}/{expected_count} pedestrians.")
        else:
            print(f"Only detected {len(selected)}/{expected_count} new pedestrians for {variant}; retrying.")
            meta = default_scale_correction_metadata()
            meta["last_reject_reason"] = "not_enough_new_people"
            return None, None, "not_enough_new_people", meta, None
    detections = [
        {"bbox": bbox, "mask": mask, "conf": person_conf, "index": index}
        for _score, index, bbox, mask, person_conf in selected
    ]
    scale_meta = default_scale_correction_metadata()
    if detections:
        scale_meta["person_confidence"] = round(
            sum(float(detection.get("conf", 0.0)) for detection in detections) / max(1, len(detections)),
            4,
        )
    corrected_image, corrected_mask, corrected_bbox, scale_meta_update, correction_reason = corrected_person_layers(
        generated_image,
        detections,
        variant or "add_single_pedestrian",
        depth_map=depth_map,
    )
    scale_meta.update(scale_meta_update or {})
    if corrected_mask is None:
        return None, None, correction_reason, scale_meta, None
    area_ratio = mask_area_ratio(corrected_mask)
    if area_ratio < max(CONTEXT_MIN_PERSON_MASK_AREA_RATIO, MIN_GHOST_PERSON_MASK_AREA_RATIO):
        print(f"Detected new person mask is too small (area_ratio={area_ratio:.5f}); rejecting as ghost/unchanged.")
        scale_meta["last_reject_reason"] = "too_small_or_ghost_person"
        return None, None, "too_small_or_ghost_person", scale_meta, None
    if background_image is not None:
        person_diff = masked_rgb_mae_255(background_image, corrected_image, corrected_mask)
        if person_diff < MIN_GHOST_PERSON_CONTRAST_255:
            print(f"Detected new person is ghost-like / low contrast (person_diff={person_diff:.2f}); retrying.")
            scale_meta["last_reject_reason"] = "ghost_person_low_contrast"
            return None, None, "ghost_person_low_contrast", scale_meta, None
    return corrected_mask, corrected_bbox, "ok", scale_meta, corrected_image

def adaptive_retry_params(base_strength, base_guidance, reject_reason, attempt):
    if attempt <= 0:
        return base_strength, base_guidance
    reason = reject_reason or "no_person_mask"
    if reason in {"ghost_person", "ghost_person_low_contrast", "too_small_or_ghost_person"}:
        return min(0.88, base_strength + 0.05 * attempt), min(9.0, base_guidance + 1.00 * attempt)
    if reason in {"low_person_conf", "no_person_mask"}:
        return min(0.84, base_strength + 0.04 * attempt), min(8.2, base_guidance + 0.60 * attempt)
    if reason == "not_enough_new_people":
        return min(0.86, base_strength + 0.03 * attempt), min(8.6, base_guidance + 0.55 * attempt)
    if reason == "too_large_for_perspective":
        return max(0.60, base_strength - 0.04 * attempt), max(6.0, base_guidance - 0.60 * attempt)
    if reason == "bad_scale_too_large":
        return max(0.58, base_strength - 0.04 * attempt), max(5.8, base_guidance - 0.45 * attempt)
    if reason == "bad_scale_too_small":
        return min(0.86, base_strength + 0.04 * attempt), min(8.4, base_guidance + 0.35 * attempt)
    if reason == "low_affordance_score":
        return min(0.82, base_strength + 0.02 * attempt), min(8.0, base_guidance + 0.20 * attempt)
    if reason == "partial_or_cropped":
        return max(0.62, base_strength - 0.03 * attempt), max(6.0, base_guidance - 0.20 * attempt)
    return min(0.80, base_strength + 0.02 * attempt), min(7.8, base_guidance + 0.25 * attempt)


def adaptive_context_expand(base_expand, reject_reason, attempt):
    if attempt <= 0 or reject_reason != "partial_or_cropped":
        return base_expand
    return base_expand * (1.0 + 0.14 * attempt)


def build_retry_config(base_prompt, base_negative, reject_reason, strength, guidance, margin, attempt):
    attempt_prompt = base_prompt
    attempt_negative = base_negative
    attempt_strength, attempt_guidance = adaptive_retry_params(strength, guidance, reject_reason, attempt)
    attempt_margin = adaptive_context_expand(margin, reject_reason, attempt)
    if attempt <= 0 or not reject_reason:
        return attempt_prompt, attempt_negative, attempt_strength, attempt_guidance, attempt_margin
    if reject_reason in {"ghost_person", "ghost_person_low_contrast", "too_small_or_ghost_person"}:
        attempt_strength = min(0.88, attempt_strength + 0.05)
        attempt_guidance = min(9.0, attempt_guidance + 1.00)
        attempt_prompt += ", solid opaque pedestrian, realistic clothing texture, street photography, realistic human body"
        attempt_negative += ", transparent, faded, ghost person, silhouette, translucent body"
    elif reject_reason in {"low_person_conf", "no_person_mask", "no_person_detected"}:
        attempt_strength = min(0.86, attempt_strength + 0.04)
        attempt_guidance = min(8.6, attempt_guidance + 0.35)
        attempt_prompt += ", clear solid person"
        attempt_negative += ", transparent, faded"
    elif reject_reason == "too_large_for_perspective":
        attempt_prompt += ", realistic scale"
        attempt_negative += ", wrong scale"
    elif reject_reason in {"scale_unrecoverable", "final_scale_mismatch"}:
        attempt_strength = min(0.86, attempt_strength + 0.03)
        attempt_guidance = min(8.4, attempt_guidance + 0.25)
        attempt_prompt += ", visible grounded person"
        attempt_negative += ", tiny, barely visible"
    elif reject_reason == "bad_scale_too_large":
        attempt_strength = max(0.58, attempt_strength - 0.03)
        attempt_guidance = max(5.8, attempt_guidance - 0.30)
        attempt_prompt += ", smaller realistic pedestrian farther in perspective"
        attempt_negative += ", giant, closeup, oversized person"
    elif reject_reason == "bad_scale_too_small":
        attempt_strength = min(0.86, attempt_strength + 0.03)
        attempt_guidance = min(8.4, attempt_guidance + 0.25)
        attempt_prompt += ", clear full-size pedestrian at the marked depth"
        attempt_negative += ", tiny, miniature, barely visible"
    elif reject_reason in {"partial_or_cropped", "partial_or_cropped_body", "accepted_mask_empty", "mask_too_soft"}:
        attempt_prompt += ", complete body visible"
        attempt_negative += ", cropped head, cropped feet, half body"
    elif reject_reason == "not_enough_new_people":
        attempt_strength = min(0.86, attempt_strength + 0.03)
        attempt_guidance = min(8.4, attempt_guidance + 0.30)
        attempt_prompt += ", all people separate, visible gaps, distinct bodies"
        attempt_negative += ", missing person, merged bodies, fused bodies, overlapping people"
    elif reject_reason == "bad_person_depth_overlap":
        attempt_strength = max(0.62, attempt_strength - 0.02)
        attempt_prompt += ", separated depth, no body overlap, distinct silhouettes"
        attempt_negative += ", overlap, merged people, fused bodies, person on person"
    elif reject_reason == "bad_occlusion":
        attempt_strength = max(0.62, attempt_strength - 0.02)
        attempt_prompt += ", plausible partial occlusion, visible body silhouette"
        attempt_negative += ", excessive overlap, cut off body, merged people"
    elif reject_reason == "low_affordance_score":
        attempt_prompt += ", feet on ground, realistic perspective scale, clean visible silhouette"
        attempt_negative += ", floating, wrong scale, impossible overlap"
    elif reject_reason == "bad_placement":
        attempt_strength = min(0.84, attempt_strength + 0.02)
        attempt_guidance = min(8.2, attempt_guidance + 0.25)
        attempt_prompt += ", exactly inside the marked placement area, centered on the guide"
        attempt_negative += ", person outside marked area, wrong location"
    elif reject_reason == "floating_or_bad_ground":
        attempt_prompt += ", feet on road"
        attempt_negative += ", floating, on vehicle"
    elif reject_reason == "bad_composite_quality":
        attempt_prompt += ", clean edges"
        attempt_negative += ", sticker, halo, blurry"
    return attempt_prompt, attempt_negative, attempt_strength, attempt_guidance, attempt_margin


def variant_retry_budget(variant, base_retries=CONTEXT_GENERATION_RETRIES):
    if variant == "add_distant_pedestrian":
        return base_retries + DISTANT_EXTRA_RETRIES
    if variant == "add_small_group":
        return base_retries + SMALL_GROUP_EXTRA_RETRIES
    if variant == "add_near_pedestrian":
        return min(base_retries, NEAR_MAX_RETRIES)
    return base_retries


def should_retry(reason, attempt, max_retries, metadata=None):
    metadata = metadata or {}
    reason = normalize_reject_reason(reason or metadata.get("reject_reason") or "unknown")
    if attempt >= max_retries:
        return False
    if metadata.get("scale_unrecoverable_streak", 0) >= EARLY_STOP_SCALE_UNRECOVERABLE_STREAK:
        return False

    early_reasons = {
        "no_person_detected",
        "no_person_mask",
        "segmenter_failed",
        "segmenter_unavailable",
        "low_person_conf",
        "partial_or_cropped_body",
        "partial_or_cropped",
        "not_enough_new_people",
        "bad_person_depth_overlap",
        "bad_placement",
        "bad_scale",
        "bad_scale_too_large",
        "bad_scale_too_small",
        "bad_occlusion",
        "low_affordance_score",
        "scale_unrecoverable",
        "floating_or_bad_ground",
        "too_small_or_ghost_person",
    }
    if reason in early_reasons:
        if "scale" in reason and metadata.get("scale_ratio_before_correction"):
            return attempt < 1
        return True

    post_paste_reasons = {
        "ghost_person_low_contrast",
        "ghost_person",
        "bad_composite_quality",
        "final_scale_mismatch",
        "accepted_mask_empty",
        "mask_too_soft",
    }
    if reason in post_paste_reasons:
        mask_area = float(metadata.get("mask_area_ratio") or 0.0)
        if mask_area >= POST_PASTE_RETRY_MIN_MASK_AREA_RATIO:
            return True
        if metadata.get("seamless_clone_used", False) and attempt == 0:
            return True
        if not metadata.get("seamless_clone_used", False):
            return True
    return False


def clamp01(value):
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _first_numeric(value, default=None):
    if value in ("", None):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list) and parsed:
                values = [float(item) for item in parsed if item not in ("", None)]
                return sum(values) / len(values) if values else default
            return float(parsed)
        except Exception:
            try:
                return float(value)
            except Exception:
                return default
    return default


def compute_quality_scores(meta):
    person_conf = _first_numeric(meta.get("person_confidence"), default=None)
    if person_conf is None:
        person_conf = _first_numeric(meta.get("person_score"), default=1.0)
    person_score = clamp01(person_conf)

    scale_ratio = _first_numeric(
        meta.get("final_scale_ratio"),
        default=_first_numeric(meta.get("scale_ratio_before"), default=_first_numeric(meta.get("scale_ratio_before_correction"), default=1.0)),
    )
    scale_score = clamp01(1.0 - min(1.0, abs(math.log(max(scale_ratio or 1.0, 1e-6)))))

    final_diff = _first_numeric(meta.get("final_person_diff_mae255"), default=0.0)
    diff_threshold = _first_numeric(meta.get("final_person_diff_threshold"), default=FINAL_COMPOSITE_MIN_MAE_255)
    background_score = clamp01(final_diff / max(1e-6, diff_threshold * 4.0))

    edge_removed = _first_numeric(meta.get("foreground_occlusion_removed_ratio"), default=0.0)
    seamless_bonus = 0.12 if meta.get("seamless_clone_used", False) else 0.0
    fallback_penalty = 0.08 if meta.get("fallback_alpha_used", False) else 0.0
    edge_score = clamp01(1.0 - edge_removed + seamless_bonus - fallback_penalty)

    quality_score = (
        0.45 * person_score
        + 0.25 * scale_score
        + 0.20 * background_score
        + 0.10 * edge_score
    )
    return {
        "person_score": round(person_score, 4),
        "scale_score": round(scale_score, 4),
        "background_score": round(background_score, 4),
        "edge_score": round(edge_score, 4),
        "quality_score": round(clamp01(quality_score), 4),
    }


def validate_composite_result(source, result, pasted_mask, variant, insert_bbox, insert_meta):
    meta = dict(insert_meta or {})
    meta["mask_area_ratio"] = mask_area_ratio(pasted_mask)
    try:
        validate_pasted_person_mask(
            pasted_mask,
            variant,
            insert_bbox,
            resolution=source.size[0],
            expected_person_height=meta.get("expected_person_height"),
        )
    except RuntimeError as exc:
        reason = normalize_reject_reason(exc)
        if "scale mismatch" in str(exc):
            reason = "final_scale_mismatch"
        elif "empty" in str(exc):
            reason = "accepted_mask_empty"
        elif "soft" in str(exc) or "transparent" in str(exc):
            reason = "mask_too_soft"
        meta["reject_reason"] = reason
        meta["last_reject_reason"] = reason
        return False, reason, meta

    debug_mask = Image.new("L", source.size, 0)
    debug_mask.paste(pasted_mask, (0, 0))
    final_person_diff = masked_rgb_mae_255(source, result, debug_mask)
    meta["final_person_diff_mae255"] = round(final_person_diff, 4)
    final_diff_threshold = FINAL_COMPOSITE_MIN_MAE_255_SEAMLESS if meta.get("seamless_clone_used") else FINAL_COMPOSITE_MIN_MAE_255
    meta["final_person_diff_threshold"] = final_diff_threshold
    bbox = pasted_mask.getbbox()
    if bbox is not None:
        mask_h = bbox[3] - bbox[1]
        expected_h = normalize_expected_person_height(meta.get("expected_person_height"))
        if expected_h:
            meta["final_scale_ratio"] = round(mask_h / max(1.0, expected_h), 4)
        bad_overlap = False
        max_overlap = 0.0
        for old_bbox in meta.get("existing_person_bboxes") or []:
            depth_ok, overlap_ratio = person_overlap_depth_ok(bbox, old_bbox)
            max_overlap = max(max_overlap, overlap_ratio)
            if not depth_ok:
                bad_overlap = True
        meta["final_person_person_overlap_ratio"] = round(max_overlap, 4)
        if bad_overlap:
            meta["reject_reason"] = "bad_person_depth_overlap"
            meta["last_reject_reason"] = "bad_person_depth_overlap"
            return False, "bad_person_depth_overlap", meta
    meta.update(compute_quality_scores(meta))
    affordance_metrics = compute_affordance_score(
        source.size,
        pasted_mask=pasted_mask,
        insert_bbox=insert_bbox,
        metadata=meta,
        variant=variant,
    )
    meta.update(affordance_metrics)
    if not affordance_metrics.get("affordance_valid", False):
        reason = affordance_reject_reason(affordance_metrics)
        meta["reject_reason"] = reason
        meta["last_reject_reason"] = reason
        return False, reason, meta
    if final_person_diff < final_diff_threshold:
        meta["reject_reason"] = "ghost_person_low_contrast"
        meta["last_reject_reason"] = "ghost_person_low_contrast"
        return False, "ghost_person_low_contrast", meta
    meta["reject_reason"] = ""
    return True, "ok", meta


def normalize_reject_reason(reason_text):
    text = str(reason_text)
    patterns = [
        r"last_reason=([A-Za-z0-9_]+)",
        r"Composite rejected as ([A-Za-z0-9_]+)",
        r"rejected as ([A-Za-z0-9_]+)",
        r"\\(([A-Za-z0-9_]+)\\)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip(")., ")
    known_reasons = [
        "scale_unrecoverable",
        "ghost_person",
        "ghost_person_low_contrast",
        "floating_or_bad_ground",
        "not_enough_new_people",
        "bad_person_depth_overlap",
        "bad_scale_too_large",
        "bad_scale_too_small",
        "bad_scale",
        "bad_occlusion",
        "low_affordance_score",
        "low_person_conf",
        "partial_or_cropped_body",
        "partial_or_cropped",
        "bad_mask_quality",
        "bad_placement",
        "no_person_detected",
    ]
    for reason in known_reasons:
        if reason in text:
            return reason
    if "expected scalar type Half" in text or "mixed dtype" in text:
        return "pipeline_dtype_mismatch"
    return text.split()[0].strip(").,") if text.split() else "unknown"


def _safe_metric(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _load_dataset_yaml(data_yaml):
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to resolve data_yaml for detection utility evaluation.") from exc
    data_yaml = Path(data_yaml)
    with data_yaml.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    return data_yaml, root, data


def _resolve_split_images(data_yaml, split="val"):
    data_yaml, root, data = _load_dataset_yaml(data_yaml)
    split_value = data.get(split) or data.get("valid" if split == "val" else split)
    if split_value is None:
        raise FileNotFoundError(f"Split {split!r} was not found in {data_yaml}.")
    split_path = Path(split_value)
    if not split_path.is_absolute():
        split_path = root / split_path
    if split_path.is_file():
        return [Path(line.strip()) for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not split_path.exists():
        raise FileNotFoundError(f"Split image path does not exist: {split_path}")
    return [path for path in sorted(split_path.rglob("*")) if path.suffix.lower() in IMAGE_EXTS]


def _label_path_for_image(image_path):
    image_path = Path(image_path)
    parts = list(image_path.parts)
    for index, part in enumerate(parts):
        if part == "images":
            label_parts = parts[:index] + ["labels"] + parts[index + 1:]
            return Path(*label_parts).with_suffix(".txt")
    return image_path.parent.parent / "labels" / image_path.with_suffix(".txt").name


def _load_yolo_label_bboxes(label_path, image_size, class_id=0):
    label_path = Path(label_path)
    if not label_path.exists():
        return []
    width, height = image_size
    bboxes = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                current_class = int(float(parts[0]))
                if current_class != class_id:
                    continue
                xc, yc, bw, bh = [float(value) for value in parts[1:5]]
            except ValueError:
                continue
            x1 = (xc - bw / 2.0) * width
            y1 = (yc - bh / 2.0) * height
            x2 = (xc + bw / 2.0) * width
            y2 = (yc + bh / 2.0) * height
            bboxes.append((x1, y1, x2, y2))
    return bboxes


def _prediction_bboxes(model, image_path, conf_threshold=0.25, class_id=0):
    try:
        results = model.predict(str(image_path), conf=conf_threshold, verbose=False)
    except Exception:
        return []
    if not results:
        return []
    boxes = getattr(results[0], "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxy))
    conf = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
    detections = []
    for index, box in enumerate(xyxy):
        if int(cls[index]) != class_id or float(conf[index]) < conf_threshold:
            continue
        detections.append(tuple(float(value) for value in box))
    return detections


def _match_detection_counts(gt_bboxes, pred_bboxes, iou_threshold=0.5):
    matched_predictions = set()
    true_positive = 0
    for gt_bbox in gt_bboxes:
        best_index = None
        best_iou = 0.0
        for index, pred_bbox in enumerate(pred_bboxes):
            if index in matched_predictions:
                continue
            iou = bbox_iou(gt_bbox, pred_bbox)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_index is not None and best_iou >= iou_threshold:
            matched_predictions.add(best_index)
            true_positive += 1
    false_negative = max(0, len(gt_bboxes) - true_positive)
    false_positive = max(0, len(pred_bboxes) - len(matched_predictions))
    return true_positive, false_negative, false_positive


def approximate_miss_rate_from_yolo(model, data_yaml, split="val", iou_threshold=0.5, conf_threshold=0.25, class_id=0):
    """Compute person-level approximate MR = FN / (TP + FN) from YOLO labels and predictions."""
    image_paths = _resolve_split_images(data_yaml, split=split)
    num_gt = 0
    num_tp = 0
    num_fn = 0
    num_fp = 0
    for image_path in image_paths:
        try:
            with Image.open(image_path) as image:
                image_size = image.size
        except Exception:
            continue
        gt_bboxes = _load_yolo_label_bboxes(_label_path_for_image(image_path), image_size, class_id=class_id)
        pred_bboxes = _prediction_bboxes(model, image_path, conf_threshold=conf_threshold, class_id=class_id)
        tp, fn, fp = _match_detection_counts(gt_bboxes, pred_bboxes, iou_threshold=iou_threshold)
        num_gt += len(gt_bboxes)
        num_tp += tp
        num_fn += fn
        num_fp += fp
    miss_rate = num_fn / max(1, num_tp + num_fn)
    return {
        "miss_rate": round(float(miss_rate), 6),
        "num_gt": int(num_gt),
        "num_tp": int(num_tp),
        "num_fn": int(num_fn),
        "num_fp": int(num_fp),
    }


def evaluate_detection_utility(
    model_path,
    data_yaml,
    split="val",
    iou_threshold=0.5,
    conf_threshold=0.25,
    save_json=True,
    output_dir=METRICS_DIR,
):
    """Evaluate downstream detector utility with YOLO AP metrics and approximate miss rate."""
    from ultralytics import YOLO

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    val_result = model.val(
        data=str(data_yaml),
        split=split,
        iou=iou_threshold,
        conf=conf_threshold,
        verbose=False,
    )
    box = getattr(val_result, "box", None)
    metrics = {
        "ap50": round(_safe_metric(getattr(box, "map50", 0.0)), 6),
        "ap75": round(_safe_metric(getattr(box, "map75", 0.0)), 6),
        "map50_95": round(_safe_metric(getattr(box, "map", 0.0)), 6),
        "miss_rate": None,
        "num_gt": 0,
        "num_tp": 0,
        "num_fn": 0,
        "num_fp": 0,
        "model_path": str(model_path),
        "data_yaml": str(data_yaml),
        "split": split,
        "iou_threshold": iou_threshold,
        "conf_threshold": conf_threshold,
    }
    try:
        metrics.update(
            approximate_miss_rate_from_yolo(
                model,
                data_yaml,
                split=split,
                iou_threshold=iou_threshold,
                conf_threshold=conf_threshold,
                class_id=0,
            )
        )
    except Exception as exc:
        metrics["miss_rate_warning"] = f"{type(exc).__name__}: {exc}"

    if save_json:
        json_path = output_dir / f"detection_utility_{split}.json"
        csv_path = output_dir / f"detection_utility_{split}.csv"
        json_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
            writer.writeheader()
            writer.writerow(metrics)
        metrics["json_path"] = str(json_path)
        metrics["csv_path"] = str(csv_path)
    return metrics


def compare_detection_utility(real_only_result, synthetic_result):
    """Compare real-only and real+synthetic detector results with direction-aware deltas."""
    real_only_result = real_only_result or {}
    synthetic_result = synthetic_result or {}

    def delta(key):
        if real_only_result.get(key) is None or synthetic_result.get(key) is None:
            return None
        return round(float(synthetic_result.get(key)) - float(real_only_result.get(key)), 6)

    return {
        "real_only_ap50": real_only_result.get("ap50"),
        "synthetic_ap50": synthetic_result.get("ap50"),
        "delta_ap50": delta("ap50"),
        "real_only_ap75": real_only_result.get("ap75"),
        "synthetic_ap75": synthetic_result.get("ap75"),
        "delta_ap75": delta("ap75"),
        "real_only_map50_95": real_only_result.get("map50_95"),
        "synthetic_map50_95": synthetic_result.get("map50_95"),
        "delta_map50_95": delta("map50_95"),
        "real_only_mr": real_only_result.get("miss_rate"),
        "synthetic_mr": synthetic_result.get("miss_rate"),
        "delta_mr": delta("miss_rate"),
        "ap_delta_direction": "higher_is_better",
        "mr_delta_direction": "lower_is_better",
    }
