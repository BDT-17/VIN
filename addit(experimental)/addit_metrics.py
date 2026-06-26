"""ADD-IT quantitative metrics — built on the shared cross-flow metric core.

ADD-IT decides placement itself and has no fixed insertion mask/bbox, so the
"object region" for background preservation is the model-derived subject mask
(``AddItResult.mask_image``) when present, else the detector's boxes.

This wraps ``shared_metrics.compute_shared_metrics`` so ADD-IT reports the SAME
schema (person / inclusion / scale / background) as the V5 and LoRA flows and
can be compared head-to-head with them.  The optional YOLO detector is loaded
lazily and injected — keeping ``addit_core`` free of detector dependencies.

The ADD-IT paper's own automated metrics (Inclusion via Grounding-DINO,
Affordance vs annotated regions, CLIP_dir/out/im) are a *superset*; the shared
"object_added/inclusion_count_delta" here is the detector-based Inclusion
analogue.  Porting CLIP_* / annotated-Affordance is a separate task.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional

# Repo root is on sys.path (the notebook adds it); import the shared core.
try:
    from shared_metrics import (
        SHARED_METRIC_DIRECTIONS, compute_shared_metrics, paired_shared,
        summarize_shared,
    )
except ImportError:  # pragma: no cover - path fallback
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from shared_metrics import (
        SHARED_METRIC_DIRECTIONS, compute_shared_metrics, paired_shared,
        summarize_shared,
    )


def make_yolo_detector(model_path: str = "yolov8m-seg.pt", device=None) -> Callable:
    """Build a detector callable matching the shared-metrics contract.

    Returns ``detector(image) -> [{"bbox_xyxy", "conf", "cls"}]`` using
    Ultralytics YOLO.  Loaded once and closed over.
    """
    from ultralytics import YOLO
    import numpy as np
    from PIL import Image

    model = YOLO(model_path)

    def detector(image) -> List[Dict]:
        if isinstance(image, (str, Path)):
            arr = np.array(Image.open(image).convert("RGB"))
        else:
            arr = np.array(image.convert("RGB"))
        kwargs = {"verbose": False, "conf": 0.10}
        if device is not None:
            kwargs["device"] = device
        res = model(arr, **kwargs)
        dets = []
        if res and res[0].boxes is not None:
            boxes = res[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            for b, c, k in zip(xyxy, confs, clss):
                dets.append({"bbox_xyxy": tuple(float(v) for v in b),
                             "conf": float(c), "cls": int(k)})
        return dets

    return detector


def metrics_for_result(result, detector: Optional[Callable] = None,
                       expected_height: Optional[float] = None,
                       conf_thr: float = 0.25, dilate_px: int = 8) -> Dict:
    """Compute the shared metric set for one ``AddItResult``.

    Background preservation uses the model-derived subject mask as the object
    region; the source image is both the inclusion baseline and the background
    reference (ADD-IT preserves the source background by design).
    """
    if not getattr(result, "success", False) or result.result_image is None:
        return {"person_detected": 0, "person_confidence": 0.0, "object_added": 0}
    m = compute_shared_metrics(
        result_image=result.result_image,
        source_image=result.source_image,
        reference_image=result.source_image,     # source is the background truth
        detector=detector,
        expected_height=expected_height,
        object_mask=result.mask_image,           # model-derived subject mask
        conf_thr=conf_thr,
        person_class=0,                           # COCO person
        dilate_px=dilate_px,
    )
    # ADD-IT-specific extension: did the auto-γ solver actually vary γ?
    if getattr(result, "gamma_trace", None):
        gt = result.gamma_trace
        m["gamma_mean"] = round(sum(gt) / len(gt), 4)
        m["gamma_min"] = round(min(gt), 4)
        m["gamma_max"] = round(max(gt), 4)
    m["used_sam2"] = bool(getattr(result, "used_sam2", False))
    return m


def metrics_for_batch(results, detector: Optional[Callable] = None,
                      expected_height: Optional[float] = None,
                      conf_thr: float = 0.25) -> Dict:
    """Per-case rows + a shared summary for a list of ``AddItResult``."""
    rows = []
    for i, r in enumerate(results):
        row = {"case_id": getattr(r, "subject_token", "") or f"case{i}",
               "seed": getattr(r, "seed", i)}
        row.update(metrics_for_result(r, detector, expected_height, conf_thr))
        rows.append(row)
    return {"rows": rows, "summary": summarize_shared(rows)}
