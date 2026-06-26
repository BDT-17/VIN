"""Per-case metrics for the SD3.5 inpaint evaluation.

No V5 metrics here (no placement_score / affordance_score / scale_correction /
harmonization_score). Component metrics only — never a single fused score.

With hard-restore on (default), the result's background outside the mask equals
the input (source_img). PIPE's source and target differ only inside the object
region, so `outside_mask_mae` vs the target reference stays ~0 — i.e. it verifies
both that hard-restore held and that the derived mask covers the object. The
informative signals are then the person_* and edge_seam metrics inside the mask.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np

# direction for paired deltas (LoRA - baseline)
METRIC_DIRECTIONS = {
    "outside_mask_mae": "lower_is_better",
    "outside_mask_ssim": "higher_is_better",
    "person_detected": "higher_is_better",
    "person_confidence": "higher_is_better",
    "person_inside_mask_ratio": "higher_is_better",
    "scale_ratio": "neutral",
    "scale_error": "lower_is_better",
    "edge_seam_score": "lower_is_better",
    "runtime_seconds": "lower_is_better",
    "cuda_peak_mb": "lower_is_better",
}


def _load_gray(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def _load_rgb(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask > 127
    from PIL import Image, ImageFilter
    m = Image.fromarray((mask > 127).astype(np.uint8) * 255)
    m = m.filter(ImageFilter.MaxFilter(px * 2 + 1))
    return np.asarray(m, dtype=np.float32) > 127


def background_metrics(reference_path, result_path, mask_path, dilate_px=8) -> Dict:
    ref = _load_rgb(reference_path)
    out = _load_rgb(result_path)
    mask = _load_gray(mask_path)
    if ref.shape != out.shape:
        from PIL import Image
        out_img = Image.fromarray(out.astype(np.uint8)).resize((ref.shape[1], ref.shape[0]))
        out = np.asarray(out_img, dtype=np.float32)
    inside = _dilate(mask, dilate_px)
    outside = ~inside
    if outside.sum() == 0:
        return {"outside_mask_mae": 0.0, "outside_mask_ssim": 1.0}
    diff = np.abs(ref - out).mean(axis=2)
    mae = float(diff[outside].mean())
    ssim = _ssim(ref.mean(axis=2), out.mean(axis=2), outside)
    return {"outside_mask_mae": round(mae, 4), "outside_mask_ssim": round(ssim, 4)}


def _ssim(a, b, region) -> float:
    a, b = a[region], b[region]
    if a.size == 0:
        return 1.0
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
                 ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2) + 1e-9))


def person_metrics(result_path, mask_path, expected_bbox_xyxy, detector=None,
                   conf_thr=0.25) -> Dict:
    """If a YOLO detector callable is provided, use it; else return not-evaluable."""
    out = {"person_detected": 0, "person_confidence": 0.0,
           "person_inside_mask_ratio": 0.0, "detected_height": 0.0,
           "expected_height": 0.0, "scale_ratio": None, "scale_error": None}
    exp_h = (expected_bbox_xyxy[3] - expected_bbox_xyxy[1]) if expected_bbox_xyxy else 0.0
    out["expected_height"] = float(exp_h)
    if detector is None:
        return out
    dets = detector(result_path)  # list of {bbox_xyxy, conf}
    persons = [d for d in dets if d.get("conf", 0) >= conf_thr]
    if not persons:
        return out
    best = max(persons, key=lambda d: d["conf"])
    bx1, by1, bx2, by2 = best["bbox_xyxy"]
    out["person_detected"] = 1
    out["person_confidence"] = round(float(best["conf"]), 4)
    out["detected_height"] = float(by2 - by1)
    out["person_inside_mask_ratio"] = round(_inside_ratio(best["bbox_xyxy"], mask_path), 4)
    if exp_h > 0:
        out["scale_ratio"] = round((by2 - by1) / exp_h, 4)
        out["scale_error"] = round(abs((by2 - by1) - exp_h) / exp_h, 4)
    return out


def _inside_ratio(bbox_xyxy, mask_path) -> float:
    mask = _load_gray(mask_path) > 127
    x1, y1, x2, y2 = (int(v) for v in bbox_xyxy)
    H, W = mask.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    area = max(1, (x2 - x1) * (y2 - y1))
    inside = mask[y1:y2, x1:x2].sum()
    return float(inside) / area


def edge_seam_score(reference_path, result_path, mask_path) -> Dict:
    ref = _load_gray(reference_path)
    out = _load_gray(result_path)
    mask = _load_gray(mask_path) > 127
    if ref.shape != out.shape:
        return {"edge_seam_score": None}
    from PIL import Image, ImageFilter
    edge = (np.asarray(Image.fromarray((mask * 255).astype(np.uint8))
                       .filter(ImageFilter.FIND_EDGES), dtype=np.float32) > 0)
    if edge.sum() == 0:
        return {"edge_seam_score": 0.0}
    gy_r, gx_r = np.gradient(ref)
    gy_o, gx_o = np.gradient(out)
    grad_diff = np.abs((gx_r + gy_r) - (gx_o + gy_o))
    return {"edge_seam_score": round(float(grad_diff[edge].mean()), 4)}


def compute_case_metrics(reference_path, result_path, mask_path, expected_bbox_xyxy,
                         detector=None, dilate_px=8, conf_thr=0.25) -> Dict:
    m = {}
    m.update(background_metrics(reference_path, result_path, mask_path, dilate_px))
    m.update(person_metrics(result_path, mask_path, expected_bbox_xyxy, detector, conf_thr))
    m.update(edge_seam_score(reference_path, result_path, mask_path))
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Cross-flow shared metrics (V5 / LoRA / ADD-IT share this schema).
# These are ADDITIVE — the LoRA-native metrics above are unchanged.  The shared
# core lives at the repo root in ``shared_metrics.py``; we expose a thin bridge
# so a LoRA eval can also emit the comparable schema (person/inclusion/scale/bg).
# ─────────────────────────────────────────────────────────────────────────────

def compute_shared_case_metrics(reference_path, result_path, source_path=None,
                                 mask_path=None, expected_bbox_xyxy=None,
                                 detector=None, dilate_px=8, conf_thr=0.25) -> Dict:
    """Emit the cross-flow shared metric schema for one LoRA case.

    For the PIPE inpaint flow the background reference is the PIPE *target*
    (its background equals the source), and the object region is the inpaint
    ``mask_path``.  ``source_path`` (object-erased input) enables Inclusion.
    """
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repo root
    from shared_metrics import compute_shared_metrics  # noqa: E402

    exp_h = None
    if expected_bbox_xyxy:
        exp_h = float(expected_bbox_xyxy[3] - expected_bbox_xyxy[1])

    def _det(img):
        return detector(img) if detector is not None else []

    return compute_shared_metrics(
        result_image=str(result_path),
        source_image=str(source_path) if source_path is not None else None,
        reference_image=str(reference_path),
        detector=(_det if detector is not None else None),
        expected_height=exp_h,
        object_mask=str(mask_path) if mask_path is not None else None,
        conf_thr=conf_thr,
        person_class=0,
        dilate_px=dilate_px,
    )
