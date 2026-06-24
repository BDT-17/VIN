"""Lightweight evaluation metrics for SD3.5 object insertion outputs."""

import csv
import json
import math
import statistics
from pathlib import Path

from sd35_config import *


def clamp01(value):
    """Clamp numeric values into the inclusive [0, 1] range."""
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def json_safe_value(value):
    """Convert common Python, NumPy and Path values into JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe_value(item) for item in value]
    return str(value)


def parse_numeric_values(value):
    """Parse a scalar or JSON/list-like metric field into a list of floats."""
    if value in (None, ""):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            try:
                return [float(value)]
            except Exception:
                return []
        return parse_numeric_values(parsed)
    if isinstance(value, (list, tuple, set)):
        values = []
        for item in value:
            values.extend(parse_numeric_values(item))
        return values
    return []


def first_numeric(value, default=None):
    """Return the first numeric value from a scalar/list-like field."""
    values = parse_numeric_values(value)
    return values[0] if values else default


def representative_numeric(value, default=None):
    """Return a conservative representative numeric value for scalar/list fields."""
    values = parse_numeric_values(value)
    if not values:
        return default
    return max(values)


def bbox_area(bbox):
    """Return positive area for an xyxy bbox."""
    if not bbox:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_intersection_area(a, b):
    """Return intersection area for two xyxy bboxes."""
    if not a or not b:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    return max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))


def bbox_iou(a, b):
    """Return IoU for two xyxy bboxes."""
    inter = bbox_intersection_area(a, b)
    union = max(1.0, bbox_area(a) + bbox_area(b) - inter)
    return inter / union


def _mask_bbox_and_area(mask):
    if mask is None:
        return None, 0.0
    bbox = mask.getbbox()
    histogram = mask.convert("L").histogram()
    active_pixels = sum(histogram[11:])
    return bbox, float(active_pixels)


def compute_placement_score(mask_bbox, image_size, insert_bbox=None, metadata=None):
    """Score whether the inserted object's foot point lands in a plausible placement band."""
    metadata = metadata or {}
    if mask_bbox is None:
        return {
            "placement_score": 0.0,
            "placement_available": False,
            "placement_warning": "missing_mask_bbox",
            "valid_grounding": False,
            "foot_y_error": "",
            "placement_band_valid": False,
        }

    width, height = image_size
    x1, _y1, x2, y2 = [float(v) for v in mask_bbox]
    foot_y = y2
    expected_ground_y = first_numeric(metadata.get("ground_y"), None)
    if expected_ground_y is None and insert_bbox is not None:
        expected_ground_y = float(insert_bbox[3])
    if expected_ground_y is None:
        expected_ground_y = foot_y

    tolerance = max(
        float(AFFORDANCE_PLACEMENT_TOLERANCE_PIXELS),
        float(AFFORDANCE_PLACEMENT_TOLERANCE_RATIO) * max(1.0, float(height)),
    )
    foot_y_error = abs(float(foot_y) - float(expected_ground_y))
    center_x = (x1 + x2) / 2.0
    boundary_valid = 0 <= center_x <= width and 0 <= foot_y <= height
    lower, upper = PATCH_ROAD_Y_RANGE
    foot_y_norm = foot_y / max(1.0, float(height))
    band_valid = (float(lower) - AFFORDANCE_PLACEMENT_BAND_MARGIN) <= foot_y_norm <= (
        float(upper) + AFFORDANCE_PLACEMENT_BAND_MARGIN
    )
    score = clamp01(1.0 - foot_y_error / max(1.0, tolerance))
    if not boundary_valid:
        score *= 0.25
    if not band_valid:
        score *= 0.55

    return {
        "placement_score": round(score, 4),
        "placement_available": True,
        "placement_warning": "",
        "valid_grounding": bool(score >= MIN_PLACEMENT_SCORE and band_valid and boundary_valid),
        "foot_y_error": round(float(foot_y_error), 4),
        "placement_band_valid": bool(band_valid),
    }


def compute_scale_score(mask_bbox, metadata=None):
    """Score whether object height matches the perspective/scale policy output."""
    metadata = metadata or {}
    if mask_bbox is None:
        return {
            "scale_score": 0.0,
            "scale_available": False,
            "scale_warning": "missing_mask_bbox",
            "expected_height": "",
            "actual_height": "",
            "scale_relative_error": "",
            "scale_valid": False,
        }

    actual_height = max(1.0, float(mask_bbox[3] - mask_bbox[1]))
    expected_height = representative_numeric(
        metadata.get("expected_height", metadata.get("expected_person_height")),
        None,
    )
    if expected_height is None or expected_height <= 0:
        return {
            "scale_score": None,
            "scale_available": False,
            "scale_warning": "missing_expected_height",
            "expected_height": "",
            "actual_height": round(actual_height, 4),
            "scale_relative_error": "",
            "scale_valid": False,
        }

    relative_error = abs(actual_height - expected_height) / max(1.0, expected_height)
    score = math.exp(-relative_error)
    return {
        "scale_score": round(clamp01(score), 4),
        "scale_available": True,
        "scale_warning": "",
        "expected_height": round(float(expected_height), 4),
        "actual_height": round(float(actual_height), 4),
        "scale_relative_error": round(float(relative_error), 4),
        "scale_valid": bool(score >= MIN_SCALE_SCORE),
    }


def compute_occlusion_score(mask_bbox, mask_area, image_size, metadata=None):
    """Score visible area, overlap and truncation quality for an inserted object."""
    metadata = metadata or {}
    if mask_bbox is None:
        return {
            "occlusion_score": 0.0,
            "occlusion_available": False,
            "occlusion_warning": "missing_mask_bbox",
            "visible_ratio": "",
            "overlap_ratio": "",
            "truncation_ratio": "",
            "occlusion_valid": False,
        }

    width, height = image_size
    box_area = max(1.0, bbox_area(mask_bbox))
    visible_ratio = clamp01(mask_area / box_area)
    overlap_ratio = max(
        first_numeric(metadata.get("final_person_person_overlap_ratio"), 0.0) or 0.0,
        first_numeric(metadata.get("foreground_occlusion_overlap_ratio"), 0.0) or 0.0,
    )
    x1, y1, x2, y2 = [float(v) for v in mask_bbox]
    clipped_w = max(0.0, min(width, x2) - max(0.0, x1))
    clipped_h = max(0.0, min(height, y2) - max(0.0, y1))
    clipped_area = clipped_w * clipped_h
    truncation_ratio = clamp01(1.0 - clipped_area / box_area)
    if x1 <= PERSON_BORDER_REJECT_PIXELS or y1 <= PERSON_BORDER_REJECT_PIXELS:
        truncation_ratio = max(truncation_ratio, AFFORDANCE_BORDER_TRUNCATION_PENALTY)
    if x2 >= width - PERSON_BORDER_REJECT_PIXELS or y2 >= height - PERSON_BORDER_REJECT_PIXELS:
        truncation_ratio = max(truncation_ratio, AFFORDANCE_BORDER_TRUNCATION_PENALTY)

    overlap_penalty = clamp01(overlap_ratio / max(1e-6, AFFORDANCE_MAX_REASONABLE_OVERLAP))
    visibility_penalty = clamp01((AFFORDANCE_MIN_VISIBLE_RATIO - visible_ratio) / max(1e-6, AFFORDANCE_MIN_VISIBLE_RATIO))
    score = clamp01(1.0 - 0.50 * overlap_penalty - 0.35 * truncation_ratio - 0.35 * visibility_penalty)
    return {
        "occlusion_score": round(score, 4),
        "occlusion_available": True,
        "occlusion_warning": "",
        "visible_ratio": round(float(visible_ratio), 4),
        "overlap_ratio": round(float(overlap_ratio), 4),
        "truncation_ratio": round(float(truncation_ratio), 4),
        "occlusion_valid": bool(score >= MIN_OCCLUSION_SCORE),
    }


def compute_affordance_score(image_size, pasted_mask=None, insert_bbox=None, metadata=None, variant=None):
    """Compute PlacementScore, ScaleScore, OcclusionScore and final AffordanceScore."""
    metadata = metadata or {}
    mask_bbox, mask_area = _mask_bbox_and_area(pasted_mask)
    placement = compute_placement_score(mask_bbox, image_size, insert_bbox=insert_bbox, metadata=metadata)
    scale = compute_scale_score(mask_bbox, metadata=metadata)
    occlusion = compute_occlusion_score(mask_bbox, mask_area, image_size, metadata=metadata)
    weights = AFFORDANCE_SCORE_WEIGHTS
    _scale_val = scale.get("scale_score")
    if _scale_val is None:
        # Scale unavailable: reweight placement and occlusion proportionally
        _p_w = float(weights.get("placement", 0.4))
        _o_w = float(weights.get("occlusion", 0.2))
        _sum_w = _p_w + _o_w
        affordance_score = (
            _p_w / _sum_w * float(placement["placement_score"])
            + _o_w / _sum_w * float(occlusion["occlusion_score"])
        )
    else:
        affordance_score = (
            float(weights.get("placement", 0.4)) * float(placement["placement_score"])
            + float(weights.get("scale", 0.4)) * float(_scale_val)
            + float(weights.get("occlusion", 0.2)) * float(occlusion["occlusion_score"])
        )
    metrics = {
        **placement,
        **scale,
        **occlusion,
        "affordance_score": round(clamp01(affordance_score), 4),
        "affordance_available": bool(
            placement.get("placement_available")
            or scale.get("scale_available")
            or occlusion.get("occlusion_available")
        ),
        "affordance_warning": ";".join(
            warning
            for warning in (
                placement.get("placement_warning"),
                scale.get("scale_warning"),
                occlusion.get("occlusion_warning"),
            )
            if warning
        ),
    }
    metrics["affordance_valid"] = bool(
        (not placement.get("placement_available") or metrics["placement_score"] >= MIN_PLACEMENT_SCORE)
        and (not scale.get("scale_available") or metrics["scale_score"] >= MIN_SCALE_SCORE)
        and (not occlusion.get("occlusion_available") or metrics["occlusion_score"] >= MIN_OCCLUSION_SCORE)
        and (
            not AFFORDANCE_REJECT_ON_TOTAL_SCORE
            or metrics["affordance_score"] >= MIN_AFFORDANCE_SCORE
        )
    )
    return metrics


def affordance_reject_reason(metrics):
    """Map failed affordance components to a retry-friendly reject reason."""
    if not metrics:
        return "low_affordance_score"
    if metrics.get("placement_available") and float(metrics.get("placement_score", 0.0)) < MIN_PLACEMENT_SCORE:
        return "bad_placement"
    if metrics.get("scale_available") and float(metrics.get("scale_score", 0.0)) < MIN_SCALE_SCORE:
        actual = first_numeric(metrics.get("actual_height"), None)
        expected = first_numeric(metrics.get("expected_height"), None)
        if actual is not None and expected is not None:
            return "bad_scale_too_large" if actual > expected else "bad_scale_too_small"
        return "bad_scale"
    if metrics.get("occlusion_available") and float(metrics.get("occlusion_score", 0.0)) < MIN_OCCLUSION_SCORE:
        return "bad_occlusion"
    if AFFORDANCE_REJECT_ON_TOTAL_SCORE and float(metrics.get("affordance_score", 0.0)) < MIN_AFFORDANCE_SCORE:
        return "low_affordance_score"
    return ""


def summarize_manifest_metrics(rows, reject_histogram=None, detection_comparison=None):
    """Build a compact JSON/CSV-friendly summary from manifest rows."""
    rows = rows or []
    reject_histogram = reject_histogram or {}
    has_accept_flag = any("accepted" in row for row in rows)
    accepted_rows = [
        row for row in rows
        if not has_accept_flag or str(row.get("accepted", True)).lower() == "true"
    ]
    rejected_rows = [
        row for row in rows
        if has_accept_flag and str(row.get("accepted", True)).lower() != "true"
    ]
    num_accepted = len(accepted_rows)
    num_rejected = len(rejected_rows) if has_accept_flag else int(sum(reject_histogram.values()))
    num_generated = len(rows) if has_accept_flag else num_accepted + num_rejected

    def mean_metric(key):
        values = [float(row.get(key, 0.0) or 0.0) for row in accepted_rows if row.get(key, "") not in ("", None)]
        return round(float(statistics.mean(values)), 4) if values else 0.0

    summary = {
        "num_generated": num_generated,
        "num_accepted": num_accepted,
        "num_rejected": num_rejected,
        "accept_rate": round(num_accepted / max(1, num_generated), 4),
        "mean_placement_score": mean_metric("placement_score"),
        "mean_scale_score": mean_metric("scale_score"),
        "mean_occlusion_score": mean_metric("occlusion_score"),
        "mean_affordance_score": mean_metric("affordance_score"),
        "reject_reason_counts": dict(reject_histogram),
    }
    if detection_comparison:
        summary.update(json_safe_value(detection_comparison))
    return summary


def write_metrics_summary(rows, reject_histogram=None, output_dir=METRICS_DIR, detection_comparison=None):
    """Write metrics_summary.json and metrics_summary.csv for a batch run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_manifest_metrics(rows, reject_histogram=reject_histogram, detection_comparison=detection_comparison)
    json_path = output_dir / "metrics_summary.json"
    csv_path = output_dir / "metrics_summary.csv"
    json_path.write_text(json.dumps(json_safe_value(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(json_safe_value(summary))
    return json_path, csv_path, summary
