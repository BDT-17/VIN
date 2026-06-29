"""Person detector + segmenter (ultralytics YOLOv8).

The concept segment-and-paste flow uses ``load_person_segmenter`` to cut the
generated person out before pasting it onto the original
(see ``segment_paste``). ``load_person_detector`` (bbox only) is also provided
for any bbox-level person check::

    detector(image_path) -> [{"bbox_xyxy": [x1, y1, x2, y2], "conf": float}, ...]

YOLO COCO class 0 == person. ``yolov8n.pt`` / ``yolov8n-seg.pt`` are tiny and
download on first use; on Kaggle (no internet during run) point ``weights`` at a
dataset mount instead, e.g. ``/kaggle/input/yolov8n/yolov8n.pt``.
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


def load_person_segmenter(
    weights: str = "yolov8n-seg.pt",
    device: Optional[str] = None,
    imgsz: int = 640,
    conf_thr: float = 0.25,
) -> Callable:
    """Return a person *segmenter* callable for the segment-and-paste pipeline.

    Unlike the detector (bbox only), this returns a per-person binary mask at the
    image's native resolution::

        segment(image) -> [{"mask": np.bool_ HxW, "bbox_xyxy": [..], "conf": float}, ...]

    sorted by descending confidence. ``image`` may be a path or a PIL.Image (the
    composite step works in-memory, so it passes the generated PIL frame directly
    without a disk round-trip). Person == COCO class 0.
    """
    try:
        from ultralytics import YOLO
    except ImportError as e:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "ultralytics is required for person segmentation. Install it "
            "(`pip install ultralytics`)."
        ) from e

    import numpy as np

    model = YOLO(weights)
    if device is not None:
        model.to(device)

    def segment(image) -> List[Dict]:
        # retina_masks=True returns masks at the image's NATIVE resolution (not the
        # low-res ~160px proto masks upsampled), giving a much crisper/cleaner
        # person silhouette — the "cắt rõ" half of a clean cut.
        res = model.predict(source=image, imgsz=imgsz, verbose=False, conf=conf_thr,
                            retina_masks=True)
        out: List[Dict] = []
        for r in res:
            boxes = getattr(r, "boxes", None)
            masks = getattr(r, "masks", None)
            if boxes is None or masks is None:
                continue
            # masks.data is (N, mh, mw) at the model's mask resolution; r.orig_shape
            # is the source (H, W). Resize each mask up to native so paste aligns.
            H, W = r.orig_shape
            md = masks.data.cpu().numpy()  # (N, mh, mw) float in [0,1]
            for i, b in enumerate(boxes):
                if int(b.cls[0]) != PERSON_CLASS:
                    continue
                m = md[i]
                if m.shape != (H, W):
                    from PIL import Image as _Image
                    m = np.asarray(
                        _Image.fromarray((m * 255).astype("uint8")).resize((W, H)),
                        dtype=np.float32,
                    ) / 255.0
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                out.append({
                    "mask": m > 0.5,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "conf": float(b.conf[0]),
                })
        out.sort(key=lambda d: d["conf"], reverse=True)
        return out

    return segment
