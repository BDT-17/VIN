"""Person detector for the inpaint/edit eval.

The eval metric contract (see ``inpaint_metrics.person_metrics``) expects a
callable::

    detector(image_path) -> [{"bbox_xyxy": [x1, y1, x2, y2], "conf": float}, ...]

restricted to *person* detections. This module provides an ultralytics YOLOv8
backend. It is the missing piece that made ``run_edit_eval`` blind: without a
detector, ``person_inside_mask_ratio`` / ``scale_ratio`` are hardcoded to 0.0 /
None, so the eval could never tell whether the generated person actually landed
inside the mask at the right scale.

YOLO COCO class 0 == person. ``yolov8n.pt`` is tiny (~6MB) and downloads on
first use; on Kaggle (no internet during run) point ``weights`` at a dataset
mount instead, e.g. ``/kaggle/input/yolov8n/yolov8n.pt``.
"""

from pathlib import Path
from typing import Callable, List, Dict, Optional

PERSON_CLASS = 0  # COCO


def load_person_detector(
    weights: str = "yolov8n.pt",
    device: Optional[str] = None,
    imgsz: int = 640,
) -> Callable[[str], List[Dict]]:
    """Return a detector callable matching the eval contract.

    Raises a clear error (not a silent empty detector) if ultralytics is
    unavailable, so a missing dependency surfaces instead of masquerading as
    "no person detected".
    """
    try:
        from ultralytics import YOLO
    except ImportError as e:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "ultralytics is required for person metrics. Install it "
            "(`pip install ultralytics`) or pass detector=None to skip "
            "person metrics."
        ) from e

    model = YOLO(weights)
    if device is not None:
        model.to(device)

    def detect(image_path: str) -> List[Dict]:
        # verbose=False keeps the eval log clean; one image per call.
        res = model.predict(source=str(image_path), imgsz=imgsz, verbose=False)
        out: List[Dict] = []
        for r in res:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            for b in boxes:
                cls = int(b.cls[0])
                if cls != PERSON_CLASS:
                    continue
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                out.append({"bbox_xyxy": [x1, y1, x2, y2], "conf": float(b.conf[0])})
        return out

    return detect


def maybe_load_person_detector(
    weights: str = "yolov8n.pt", device: Optional[str] = None
) -> Optional[Callable[[str], List[Dict]]]:
    """Best-effort loader: return None (with a printed reason) on failure.

    Use this in eval entrypoints where a missing detector should degrade to
    background-only metrics rather than abort the whole eval run.
    """
    try:
        return load_person_detector(weights=weights, device=device)
    except Exception as e:  # noqa: BLE001 - eval should not crash on detector
        print(f"[detector] disabled ({e}); person metrics will be 0/None")
        return None
