"""Shared evaluation metrics — common core for ALL THREE insertion flows.

The repo has three object-insertion flows that previously each measured quality
differently and could not be compared head-to-head:

  * V5 augmentation  (``sd35_metrics.py``)        — fused affordance score + gates
  * LoRA inpaint     (``LoRA/inference/inpaint_metrics.py``) — component-only
  * ADD-IT           (``addit(experimental)/``)   — (had no quantitative metric)

This module is the **shared intersection** every flow can compute: it depends on
nothing but ``numpy`` + ``Pillow`` (and an *injected* detector callable), so the
root V5 flow, the ``LoRA/`` package and the ``addit(experimental)/`` package can
all ``import shared_metrics`` once the repo root is on ``sys.path`` (which all
three notebooks already do).

Scope (the agreed common set — no fixed mask / no fixed bbox required):
  * **Person detection**     — ``person_detected``, ``person_confidence``
  * **Inclusion**            — was a NEW object actually added vs the source?
                               (detector count delta, ADD-IT-paper "Inclusion")
  * **Scale**                — ``scale_ratio``, ``scale_error`` vs expected height
  * **Background preservation** — ``bg_mae`` / ``bg_ssim`` of pixels OUTSIDE the
                               object region (an explicit mask if a flow has one,
                               else the detected-object bbox).

Each flow keeps its own extra metrics (V5 placement/affordance, LoRA
edge_seam/outside_mask, ADD-IT γ-trace) as *extensions* layered on top of this.

Detector contract (dependency-injected so this module imports anywhere):
    detector(image_path_or_pil) -> list of {"bbox_xyxy": (x1,y1,x2,y2), "conf": float, "cls": int}
``cls`` is optional; if present, ``person_class`` filters to it (COCO person=0).
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np

ImageLike = Union[str, Path, "object"]  # str/Path or PIL.Image

# Direction each metric should move for "better" — used by paired comparisons.
SHARED_METRIC_DIRECTIONS = {
    "person_detected": "higher_is_better",
    "person_confidence": "higher_is_better",
    "person_count": "neutral",
    "object_added": "higher_is_better",     # inclusion (boolean as 0/1)
    "inclusion_count_delta": "higher_is_better",
    "scale_ratio": "neutral",
    "scale_error": "lower_is_better",
    "bg_mae": "lower_is_better",
    "bg_ssim": "higher_is_better",
}


# ═══════════════════════════════════════════════════════════════════════════
# Image loading (lazy PIL import so the module is importable without PIL too)
# ═══════════════════════════════════════════════════════════════════════════

def _to_pil(image: ImageLike):
    from PIL import Image
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    return image.convert("RGB")


def _load_rgb(image: ImageLike) -> np.ndarray:
    return np.asarray(_to_pil(image), dtype=np.float32)


def _load_gray(image: ImageLike) -> np.ndarray:
    from PIL import Image
    if isinstance(image, (str, Path)):
        return np.asarray(Image.open(image).convert("L"), dtype=np.float32)
    return np.asarray(image.convert("L"), dtype=np.float32)


def _match_size(arr: np.ndarray, ref_hw) -> np.ndarray:
    """Resize an HxWxC (or HxW) array to ref (H, W) if needed."""
    h, w = ref_hw
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr
    from PIL import Image
    mode = "RGB" if arr.ndim == 3 else "L"
    img = Image.fromarray(arr.astype(np.uint8), mode=mode).resize((w, h))
    return np.asarray(img, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Detection helpers
# ═══════════════════════════════════════════════════════════════════════════

def _run_detector(detector: Callable, image: ImageLike, conf_thr: float,
                  person_class: Optional[int]) -> List[Dict]:
    """Run an injected detector and filter to confident (person-class) boxes."""
    if detector is None:
        return []
    dets = detector(image) or []
    out = []
    for d in dets:
        if d.get("conf", 0.0) < conf_thr:
            continue
        if person_class is not None and d.get("cls") is not None and int(d["cls"]) != int(person_class):
            continue
        out.append(d)
    return out


def _bbox_height(bbox) -> float:
    return float(bbox[3] - bbox[1])


# ═══════════════════════════════════════════════════════════════════════════
# 1. Person detection
# ═══════════════════════════════════════════════════════════════════════════

def person_detection_metrics(result_image: ImageLike, detector: Callable,
                             conf_thr: float = 0.25,
                             person_class: Optional[int] = 0) -> Dict:
    persons = _run_detector(detector, result_image, conf_thr, person_class)
    if not persons:
        return {"person_detected": 0, "person_confidence": 0.0, "person_count": 0}
    best = max(persons, key=lambda d: d["conf"])
    return {
        "person_detected": 1,
        "person_confidence": round(float(best["conf"]), 4),
        "person_count": len(persons),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. Inclusion — did a NEW object actually appear vs the source?
#    (ADD-IT paper's "Inclusion": fraction of cases where the object was added.)
# ═══════════════════════════════════════════════════════════════════════════

def inclusion_metrics(source_image: ImageLike, result_image: ImageLike,
                      detector: Callable, conf_thr: float = 0.25,
                      person_class: Optional[int] = 0) -> Dict:
    """Compare detected object count in result vs source.

    A positive ``inclusion_count_delta`` (and ``object_added=1``) means the edit
    introduced at least one more detected object than the source had — the core
    "did Add-it actually add something" signal, also valid for V5 and LoRA.
    """
    src_n = len(_run_detector(detector, source_image, conf_thr, person_class)) if source_image is not None else 0
    res_n = len(_run_detector(detector, result_image, conf_thr, person_class))
    delta = res_n - src_n
    return {
        "inclusion_count_delta": int(delta),
        "object_added": 1 if delta >= 1 else 0,
        "source_person_count": int(src_n),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. Scale — detected object height vs an expected height
# ═══════════════════════════════════════════════════════════════════════════

def scale_metrics(result_image: ImageLike, expected_height: Optional[float],
                  detector: Callable, conf_thr: float = 0.25,
                  person_class: Optional[int] = 0) -> Dict:
    """``scale_ratio`` = detected/expected, ``scale_error`` = |detected-expected|/expected.

    Returns ``None`` for both when no expected height is supplied (honest: do not
    fabricate a perfect 1.0), matching the V5 flow's "scale_score is None when no
    prior" convention.
    """
    out = {"detected_height": 0.0, "expected_height": float(expected_height or 0.0),
           "scale_ratio": None, "scale_error": None}
    persons = _run_detector(detector, result_image, conf_thr, person_class)
    if not persons:
        return out
    best = max(persons, key=lambda d: d["conf"])
    h = _bbox_height(best["bbox_xyxy"])
    out["detected_height"] = round(float(h), 4)
    if expected_height and expected_height > 0:
        out["scale_ratio"] = round(h / expected_height, 4)
        out["scale_error"] = round(abs(h - expected_height) / expected_height, 4)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 4. Background preservation — pixels OUTSIDE the object region
# ═══════════════════════════════════════════════════════════════════════════

def _dilate(mask_bool: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask_bool
    from PIL import Image, ImageFilter
    m = Image.fromarray(mask_bool.astype(np.uint8) * 255)
    m = m.filter(ImageFilter.MaxFilter(px * 2 + 1))
    return np.asarray(m, dtype=np.float32) > 127


def _ssim_region(a: np.ndarray, b: np.ndarray, region: np.ndarray) -> float:
    """Global (single-window) SSIM over a boolean region — matches the LoRA flow."""
    aa, bb = a[region], b[region]
    if aa.size == 0:
        return 1.0
    mu_a, mu_b = aa.mean(), bb.mean()
    va, vb = aa.var(), bb.var()
    cov = ((aa - mu_a) * (bb - mu_b)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
                 ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2) + 1e-9))


def _object_region(result_image: ImageLike, ref_hw, object_mask, object_bbox,
                   detector, conf_thr, person_class, dilate_px) -> np.ndarray:
    """Boolean mask of the OBJECT region (True = inside object) at ref resolution.

    Priority: explicit mask → explicit bbox → detected-object bbox(es) → empty.
    """
    h, w = ref_hw
    region = np.zeros((h, w), dtype=bool)
    if object_mask is not None:
        m = _load_gray(object_mask)
        m = _match_size(m, ref_hw)
        region = m > 127
    elif object_bbox is not None:
        x1, y1, x2, y2 = (int(v) for v in object_bbox)
        region[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
    else:
        for d in _run_detector(detector, result_image, conf_thr, person_class):
            x1, y1, x2, y2 = (int(v) for v in d["bbox_xyxy"])
            region[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
    return _dilate(region, dilate_px)


def background_metrics(reference_image: ImageLike, result_image: ImageLike,
                       object_mask: Optional[ImageLike] = None,
                       object_bbox: Optional[Sequence[float]] = None,
                       detector: Optional[Callable] = None,
                       conf_thr: float = 0.25, person_class: Optional[int] = 0,
                       dilate_px: int = 8) -> Dict:
    """MAE / SSIM of pixels OUTSIDE the object region, result vs reference.

    The "object region" is taken from (in priority order) an explicit
    ``object_mask`` (LoRA's mask), an explicit ``object_bbox`` (V5's insert
    bbox), or the detector's boxes (ADD-IT, which has no fixed region).  Lower
    ``bg_mae`` / higher ``bg_ssim`` ⇒ background better preserved.
    """
    ref = _load_rgb(reference_image)
    out = _load_rgb(result_image)
    out = _match_size(out, ref.shape[:2])
    region = _object_region(result_image, ref.shape[:2], object_mask, object_bbox,
                            detector, conf_thr, person_class, dilate_px)
    outside = ~region
    if outside.sum() == 0:
        return {"bg_mae": 0.0, "bg_ssim": 1.0}
    diff = np.abs(ref - out).mean(axis=2)
    mae = float(diff[outside].mean())
    ssim = _ssim_region(ref.mean(axis=2), out.mean(axis=2), outside)
    return {"bg_mae": round(mae, 4), "bg_ssim": round(ssim, 4)}


# ═══════════════════════════════════════════════════════════════════════════
# Aggregate — the single call all three flows use
# ═══════════════════════════════════════════════════════════════════════════

def compute_shared_metrics(
    result_image: ImageLike,
    source_image: Optional[ImageLike] = None,
    reference_image: Optional[ImageLike] = None,
    detector: Optional[Callable] = None,
    expected_height: Optional[float] = None,
    object_mask: Optional[ImageLike] = None,
    object_bbox: Optional[Sequence[float]] = None,
    conf_thr: float = 0.25,
    person_class: Optional[int] = 0,
    dilate_px: int = 8,
) -> Dict:
    """Compute the common metric set for one insertion result.

    Parameters
    ----------
    result_image     : the edited / augmented output (required).
    source_image     : the pre-edit image — enables Inclusion (object added?).
    reference_image  : the background-truth image for bg preservation.  For V5
                       and ADD-IT this is the source; for LoRA-on-PIPE it is the
                       PIPE target (whose background equals the source).
    detector         : injected ``detector(img)->[{bbox_xyxy,conf,cls}]``.  All
                       detection-based metrics are skipped (None/0) without it.
    expected_height  : expected object pixel height for the scale metrics.
    object_mask/bbox : explicit object region for bg preservation (optional).
    """
    m: Dict = {}
    m.update(person_detection_metrics(result_image, detector, conf_thr, person_class))
    if source_image is not None:
        m.update(inclusion_metrics(source_image, result_image, detector, conf_thr, person_class))
    m.update(scale_metrics(result_image, expected_height, detector, conf_thr, person_class))
    if reference_image is not None:
        m.update(background_metrics(reference_image, result_image, object_mask,
                                    object_bbox, detector, conf_thr, person_class, dilate_px))
    return m


# ═══════════════════════════════════════════════════════════════════════════
# Summary + paired comparison (works across flows since the schema is shared)
# ═══════════════════════════════════════════════════════════════════════════

def _mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 4) if vals else None


def summarize_shared(rows: List[Dict]) -> Dict:
    """Mean of every shared metric over a list of per-case metric dicts."""
    keys = [k for k in SHARED_METRIC_DIRECTIONS if any(k in r for r in rows)]
    return {
        "num_cases": len(rows),
        "means": {k: _mean([r.get(k) for r in rows]) for k in keys},
    }


def paired_shared(rows_a: List[Dict], rows_b: List[Dict],
                  key_fields=("case_id", "seed"), label_a="A", label_b="B") -> Dict:
    """Per-metric paired comparison between two flows / conditions.

    Each row should carry the ``key_fields`` so pairs can be matched.  Returns
    per-metric means for each side, the delta (B−A), and whether B improved
    given each metric's direction.
    """
    def keyf(r):
        return tuple(r.get(k) for k in key_fields)

    a = {keyf(r): r for r in rows_a}
    b = {keyf(r): r for r in rows_b}
    common = sorted(set(a) & set(b))
    metrics = {}
    for name, direction in SHARED_METRIC_DIRECTIONS.items():
        ma = _mean([a[k].get(name) for k in common])
        mb = _mean([b[k].get(name) for k in common])
        delta = round(mb - ma, 4) if (ma is not None and mb is not None) else None
        improved = None
        if delta is not None and direction == "higher_is_better":
            improved = delta >= 0
        elif delta is not None and direction == "lower_is_better":
            improved = delta <= 0
        metrics[name] = {f"{label_a}_mean": ma, f"{label_b}_mean": mb,
                         "delta": delta, "direction": direction, "improved": improved}
    return {"matched_pairs": len(common), "metrics": metrics}
