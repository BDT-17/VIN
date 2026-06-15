"""Add-it reference hints for SD3.5 CityPersons augmentation.

Add-it outputs created here are teacher/reference candidates only.  They are
saved for inspection and never returned as final augmented dataset images.
"""

import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageDraw

from sd35_config import *
from sd35_utils import *
from sd35_evaluation import (
    bbox_iou,
    load_person_segmenter,
    mask_bbox_from_array,
    mask_bbox_touches_border,
)


ADDIT_NUM_CANDIDATES = int(globals().get("ADDIT_NUM_CANDIDATES", 3))
ADDIT_MIN_PERSON_CONF = float(globals().get("ADDIT_MIN_PERSON_CONF", 0.35))
ADDIT_MAX_EXISTING_IOU = float(globals().get("ADDIT_MAX_EXISTING_IOU", 0.25))
ADDIT_SAVE_REFERENCES = bool(globals().get("ADDIT_SAVE_REFERENCES", True))
ADDIT_RETRY_SEED_STEP = int(globals().get("ADDIT_RETRY_SEED_STEP", 9973))
ADDIT_REFERENCE_DIR = Path(globals().get("ADDIT_REFERENCE_DIR", OUTPUT_DIR / "addit_references"))
ADDIT_REFERENCE_DEBUG_DIR = Path(globals().get("ADDIT_REFERENCE_DEBUG_DIR", OUTPUT_DIR / "debug_addit_reference"))
globals().setdefault("ADDIT_REFERENCE_ENABLED", True)


def addit_reference_flag_enabled():
    config = globals().get("EFFECTIVE_CONFIG", {})
    if isinstance(config, dict) and "ADDIT_REFERENCE_ENABLED" in config:
        return bool(config["ADDIT_REFERENCE_ENABLED"])
    return bool(globals().get("ADDIT_CONFIG", {}).get("ADDIT_REFERENCE_ENABLED", True))


@dataclass
class AddItReferenceHint:
    valid: bool = False
    insert_bbox: Optional[Tuple[int, int, int, int]] = None
    mask: Optional[Image.Image] = None
    estimated_scale: Optional[float] = None
    center: Optional[Tuple[float, float]] = None
    confidence: float = 0.0
    candidate_id: int = -1
    candidate_path: str = ""
    existing_iou: float = 0.0
    reject_reason: str = ""
    debug_info: dict = field(default_factory=dict)

    def manifest_fields(self):
        return {
            "placement_source": "addit_reference" if self.valid else "heuristic",
            "addit_candidate_id": self.candidate_id if self.candidate_id >= 0 else "",
            "addit_person_conf": round(float(self.confidence), 4) if self.confidence else "",
            "addit_bbox": json.dumps(self.insert_bbox) if self.insert_bbox else "",
            "addit_existing_iou": round(float(self.existing_iou), 4) if self.existing_iou else "",
            "addit_valid": bool(self.valid),
            "addit_reject_reason": self.reject_reason,
            "addit_reference_path": self.candidate_path,
        }


def empty_addit_reference_fields(reason=""):
    hint = AddItReferenceHint(valid=False, reject_reason=reason)
    return hint.manifest_fields()


def addit_reference_meta_from_hint(hint: Optional[AddItReferenceHint]):
    if hint is None:
        return empty_addit_reference_fields()
    return hint.manifest_fields()


def _safe_bbox(bbox, image_size):
    width, height = image_size
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1, x2, y2 = clamp_bbox((x1, y1, x2, y2), width, height)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _detect_persons(image, min_conf=0.0):
    segmenter = load_person_segmenter()
    if segmenter is None:
        return [], "segmenter_unavailable"
    try:
        results = segmenter.predict(image, imgsz=RESOLUTION, conf=min_conf, verbose=False)
    except Exception as exc:
        return [], f"segmenter_failed:{type(exc).__name__}"
    if not results:
        return [], "no_yolo_results"
    result = results[0]
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or boxes.xyxy is None:
        return [], "no_boxes"
    xyxy = boxes.xyxy.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxy))
    conf = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
    detections = []
    for index, box in enumerate(xyxy):
        if int(cls[index]) != 0 or float(conf[index]) < min_conf:
            continue
        mask = None
        raw_mask_bbox = None
        if masks is not None and getattr(masks, "data", None) is not None and index < len(masks.data):
            raw_mask = masks.data[index].detach().cpu().numpy()
            raw_mask_bbox = mask_bbox_from_array(raw_mask, threshold=CONTEXT_PERSON_MASK_THRESHOLD)
            mask = Image.fromarray((raw_mask > CONTEXT_PERSON_MASK_THRESHOLD).astype(np.uint8) * 255, mode="L")
            mask = mask.resize(image.size, Image.NEAREST)
        detections.append({
            "bbox": tuple(float(v) for v in box),
            "conf": float(conf[index]),
            "mask": mask,
            "mask_bbox": raw_mask_bbox,
        })
    return detections, ""


def _change_score(source, candidate, bbox):
    crop_a = source.crop(bbox).convert("RGB")
    crop_b = candidate.crop(bbox).convert("RGB")
    diff = ImageChops.difference(crop_a, crop_b).convert("L")
    arr = np.asarray(diff, dtype=np.float32)
    return float(arr.mean() / 255.0)


def _validate_candidate_bbox(detection, source_persons, semantic_masks, depth_map, image_size, variant):
    bbox = _safe_bbox(detection["bbox"], image_size)
    if bbox is None:
        return False, "bbox_outside_image", 0.0, None
    width, height = image_size
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        return False, "bbox_outside_image", 0.0, bbox
    h_ratio = bh / max(1, height)
    if h_ratio < MIN_PERSON_HEIGHT_RATIO or h_ratio > MAX_PERSON_HEIGHT_RATIO:
        return False, "bad_bbox_size", 0.0, bbox
    aspect = bh / max(1, bw)
    if aspect < MIN_PERSON_ASPECT_RATIO or aspect > MAX_PERSON_ASPECT_RATIO:
        return False, "bad_pedestrian_aspect", 0.0, bbox
    max_existing_iou = 0.0
    for old_bbox in source_persons:
        max_existing_iou = max(max_existing_iou, bbox_iou(bbox, old_bbox))
        depth_ok, _overlap = person_overlap_depth_ok(bbox, old_bbox)
        if not depth_ok:
            return False, "overlaps_existing_person_depth", max_existing_iou, bbox
    if max_existing_iou > ADDIT_MAX_EXISTING_IOU:
        return False, "overlaps_existing_person", max_existing_iou, bbox
    if detection["conf"] < ADDIT_MIN_PERSON_CONF:
        return False, "low_person_conf", max_existing_iou, bbox
    if detection.get("mask_bbox") is not None and mask_bbox_touches_border(detection["mask_bbox"], image_size):
        return False, "partial_or_cropped_body", max_existing_iou, bbox
    if semantic_masks:
        foot_score = mask_coverage(semantic_masks.get("valid"), foot_support_bbox(bbox))
        foot_avoid = mask_coverage(semantic_masks.get("avoid"), foot_support_bbox(bbox))
        body_avoid = mask_coverage(semantic_masks.get("avoid"), bbox)
        if foot_score < MIN_FOOT_SUPPORT:
            return False, "floating_or_bad_ground", max_existing_iou, bbox
        if foot_avoid > MAX_FOOT_AVOID_SUPPORT or body_avoid > MAX_BODY_AVOID_SUPPORT:
            return False, "bad_semantic_area", max_existing_iou, bbox
    else:
        ground_ratio = y2 / max(1, height)
        if ground_ratio < PATCH_ROAD_Y_RANGE[0] - 0.08 or ground_ratio > PATCH_ROAD_Y_RANGE[1] + 0.06:
            return False, "implausible_ground_y", max_existing_iou, bbox
    expected_h = expected_person_height_from_depth((x1 + x2) / 2, y2, depth_map, height, variant=variant)
    scale = float(expected_h / max(1, bh))
    return True, "", max_existing_iou, bbox, scale


def _save_candidate_debug(record, source, candidate, hint):
    debug_dir = ADDIT_REFERENCE_DEBUG_DIR / record.split / record.bucket
    debug_dir.mkdir(parents=True, exist_ok=True)
    overlay = candidate.copy()
    draw = ImageDraw.Draw(overlay)
    if hint.insert_bbox:
        draw.rectangle(hint.insert_bbox, outline="lime" if hint.valid else "red", width=3)
        draw.text((hint.insert_bbox[0] + 4, hint.insert_bbox[1] + 4), hint.reject_reason or "addit", fill="lime")
    w, h = source.size
    strip = Image.new("RGB", (w * 3, h + 28), "white")
    strip_draw = ImageDraw.Draw(strip)
    strip_draw.text((8, 6), f"{record.path.name} | candidate={hint.candidate_id} | valid={hint.valid}", fill="black")
    strip.paste(source, (0, 28))
    strip.paste(candidate, (w, 28))
    strip.paste(overlay, (w * 2, 28))
    out_path = debug_dir / f"{record.path.stem}_addit_ref_{hint.candidate_id:02d}.png"
    strip.save(out_path)
    return str(out_path)


def _save_reference_candidate(record, candidate, variant, seed, candidate_id):
    out_dir = ADDIT_REFERENCE_DIR / record.split / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{record.path.stem}_addit_ref_{candidate_id:02d}_{seed}.png"
    candidate.save(out_path)
    return str(out_path)


def extract_addit_reference_hint(source, candidate, record, variant, candidate_id=0, candidate_path="", device=TRAIN_DEVICE, depth_map=None):
    original = load_source_image(record.path)
    existing_person_bboxes = load_person_bboxes_for_crop(record, original.size, resolution=source.size[0])
    semantic_masks = semantic_placement_masks(source, record, device=device)
    depth_map = depth_map if depth_map is not None else estimate_depth_map(source, device="cpu")
    source_persons, source_reason = _detect_persons(source, min_conf=CONTEXT_PERSON_MIN_CONFIDENCE)
    yolo_source_bboxes = [det["bbox"] for det in source_persons]
    source_bboxes = yolo_source_bboxes or existing_person_bboxes
    candidate_persons, detect_reason = _detect_persons(candidate, min_conf=CONTEXT_PERSON_MIN_CONFIDENCE)
    if not candidate_persons:
        return AddItReferenceHint(
            valid=False,
            candidate_id=candidate_id,
            candidate_path=candidate_path,
            reject_reason=detect_reason or "no_person_detected",
            debug_info={"source_detect_reason": source_reason},
        )

    best = None
    rejected = []
    for det in candidate_persons:
        validation = _validate_candidate_bbox(det, source_bboxes, semantic_masks, depth_map, candidate.size, variant)
        valid = bool(validation[0])
        reason = validation[1]
        existing_iou = validation[2]
        bbox = validation[3]
        scale = validation[4] if valid and len(validation) > 4 else None
        if bbox is None:
            rejected.append(reason)
            continue
        novelty = 1.0 - min(1.0, existing_iou / max(1e-6, ADDIT_MAX_EXISTING_IOU))
        change = _change_score(source, candidate, bbox)
        score = float(det["conf"]) + 0.35 * novelty + 0.25 * change
        item = (score, valid, reason, bbox, scale, existing_iou, det, change)
        if best is None or score > best[0]:
            best = item
        if not valid:
            rejected.append(reason)

    if best is None:
        return AddItReferenceHint(
            valid=False,
            candidate_id=candidate_id,
            candidate_path=candidate_path,
            reject_reason=rejected[-1] if rejected else "no_new_person",
        )

    _score, valid, reason, bbox, scale, existing_iou, det, change = best
    hint = AddItReferenceHint(
        valid=valid,
        insert_bbox=bbox,
        mask=det.get("mask"),
        estimated_scale=scale,
        center=((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
        confidence=float(det["conf"]),
        candidate_id=candidate_id,
        candidate_path=candidate_path,
        existing_iou=float(existing_iou),
        reject_reason="" if valid else reason,
        debug_info={
            "change_score": round(change, 4),
            "candidate_person_count": len(candidate_persons),
            "source_person_count": len(source_bboxes),
            "rejected": rejected,
        },
    )
    debug_path = _save_candidate_debug(record, source, candidate, hint)
    hint.debug_info["debug_path"] = debug_path
    return hint


def generate_addit_reference_hints(pipe, source, record, variant, seed, device=TRAIN_DEVICE):
    if not addit_reference_flag_enabled():
        return []
    try:
        from addit.addit_pipeline import AddItCityPersonsPipeline
    except Exception as exc:
        print("Add-it reference pipeline unavailable:", type(exc).__name__, exc)
        return [AddItReferenceHint(valid=False, reject_reason="addit_import_failed")]

    hints: List[AddItReferenceHint] = []
    addit_pipe = AddItCityPersonsPipeline(pipe, yolo_model_path=CONTEXT_PERSON_SEGMENTATION_MODEL, device=device)
    depth_map = estimate_depth_map(source, device="cpu")
    for candidate_id in range(max(1, ADDIT_NUM_CANDIDATES)):
        candidate_seed = seed + candidate_id * ADDIT_RETRY_SEED_STEP
        try:
            result = addit_pipe.run_single(record, variant, seed=candidate_seed, device=device)
            candidate = result.result_image.resize(source.size) if result and result.result_image is not None else None
            if candidate is None:
                hints.append(AddItReferenceHint(
                    valid=False,
                    candidate_id=candidate_id,
                    reject_reason=getattr(result, "reject_reason", "addit_no_candidate"),
                ))
                continue
            candidate_path = ""
            if ADDIT_SAVE_REFERENCES:
                candidate_path = _save_reference_candidate(record, candidate, variant, candidate_seed, candidate_id)
            hint = extract_addit_reference_hint(
                source,
                candidate,
                record,
                variant,
                candidate_id=candidate_id,
                candidate_path=candidate_path,
                device=device,
                depth_map=depth_map,
            )
            hints.append(hint)
        except RuntimeError as exc:
            message = str(exc).lower()
            reason = "addit_oom" if "out of memory" in message or "cuda" in message else "addit_runtime_failed"
            hints.append(AddItReferenceHint(valid=False, candidate_id=candidate_id, reject_reason=reason))
            print(f"Add-it reference candidate {candidate_id} failed: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=4)
            clear_cuda()
        except Exception as exc:
            hints.append(AddItReferenceHint(valid=False, candidate_id=candidate_id, reject_reason="addit_failed"))
            print(f"Add-it reference candidate {candidate_id} failed: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=4)
    return hints


def first_valid_addit_hint(hints):
    for hint in hints or []:
        if hint.valid and hint.insert_bbox is not None:
            return hint
    return None


def save_addit_final_debug(record, final_image, insert_bbox, metadata, seed, variant):
    if not addit_reference_flag_enabled():
        return ""
    debug_dir = ADDIT_REFERENCE_DEBUG_DIR / record.split / record.bucket
    debug_dir.mkdir(parents=True, exist_ok=True)
    source = resize_center_crop(load_source_image(record.path), resolution=final_image.size[0])
    overlay = final_image.copy()
    draw = ImageDraw.Draw(overlay)
    if insert_bbox:
        draw.rectangle(tuple(int(v) for v in insert_bbox), outline="cyan", width=3)
    label = f"{metadata.get('placement_source', 'heuristic')} | {metadata.get('reject_reason', '')}"
    draw.text((8, 8), label, fill="cyan")
    w, h = final_image.size
    strip = Image.new("RGB", (w * 3, h + 28), "white")
    strip_draw = ImageDraw.Draw(strip)
    strip_draw.text((8, 6), f"{record.path.name} | {variant} | seed={seed}", fill="black")
    strip.paste(source, (0, 28))
    strip.paste(final_image, (w, 28))
    strip.paste(overlay, (w * 2, 28))
    out_path = debug_dir / f"{record.path.stem}_final_sd35_{variant}_{seed}.png"
    strip.save(out_path)
    return str(out_path)
