"""Contract tests for the cross-flow shared metric core (shared_metrics.py).

Uses a fake injected detector so no torch/YOLO is needed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
import shared_metrics as S

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def _img(arr):
    return Image.fromarray(arr.astype("uint8"))


def _fake_detector(boxes):
    """boxes: list of (bbox_xyxy, conf, cls). Returns a detector ignoring input."""
    def det(_image):
        return [{"bbox_xyxy": b, "conf": c, "cls": k} for (b, c, k) in boxes]
    return det


def test_person_detection_basic():
    det = _fake_detector([((10, 10, 30, 60), 0.9, 0), ((40, 40, 50, 55), 0.4, 0)])
    m = S.person_detection_metrics(_img(np.zeros((64, 64, 3))), det, conf_thr=0.25)
    assert m["person_detected"] == 1
    assert m["person_count"] == 2
    assert m["person_confidence"] == 0.9


def test_person_class_filter():
    det = _fake_detector([((10, 10, 30, 60), 0.9, 2)])  # class 2 = not person
    m = S.person_detection_metrics(_img(np.zeros((64, 64, 3))), det, person_class=0)
    assert m["person_detected"] == 0


def test_inclusion_detects_added_object():
    src_det = _fake_detector([])                       # source: no person
    res_det = _fake_detector([((10, 10, 30, 60), 0.9, 0)])  # result: one person
    # inclusion compares counts; give the same detector via a switch on identity
    class Switch:
        def __init__(self): self.calls = 0
        def __call__(self, image):
            self.calls += 1
            return [] if self.calls == 1 else [{"bbox_xyxy": (10, 10, 30, 60), "conf": 0.9, "cls": 0}]
    det = Switch()
    m = S.inclusion_metrics(_img(np.zeros((64, 64, 3))), _img(np.zeros((64, 64, 3))), det)
    assert m["inclusion_count_delta"] == 1
    assert m["object_added"] == 1
    assert m["source_person_count"] == 0


def test_scale_none_without_expected():
    det = _fake_detector([((10, 10, 30, 60), 0.9, 0)])
    m = S.scale_metrics(_img(np.zeros((64, 64, 3))), expected_height=None, detector=det)
    assert m["scale_ratio"] is None and m["scale_error"] is None
    assert m["detected_height"] == 50.0


def test_scale_ratio_and_error():
    det = _fake_detector([((10, 10, 30, 60), 0.9, 0)])   # height 50
    m = S.scale_metrics(_img(np.zeros((64, 64, 3))), expected_height=40.0, detector=det)
    assert m["scale_ratio"] == 1.25
    assert m["scale_error"] == 0.25


def test_background_identical_is_perfect():
    ref = np.full((64, 64, 3), 120, dtype=np.uint8)
    out = ref.copy()
    bbox = (20, 20, 40, 40)
    m = S.background_metrics(_img(ref), _img(out), object_bbox=bbox, dilate_px=2)
    assert m["bg_mae"] == 0.0
    assert m["bg_ssim"] >= 0.99


def test_background_change_outside_object_is_penalized():
    ref = np.full((64, 64, 3), 120, dtype=np.uint8)
    out = ref.copy()
    out[0:10, 0:10] = 0          # corruption OUTSIDE the object bbox
    bbox = (20, 20, 40, 40)
    m = S.background_metrics(_img(ref), _img(out), object_bbox=bbox, dilate_px=2)
    assert m["bg_mae"] > 0.0


def test_background_change_inside_object_is_ignored():
    ref = np.full((64, 64, 3), 120, dtype=np.uint8)
    out = ref.copy()
    out[22:38, 22:38] = 0        # change INSIDE the object bbox -> not penalized
    bbox = (20, 20, 40, 40)
    m = S.background_metrics(_img(ref), _img(out), object_bbox=bbox, dilate_px=0)
    assert m["bg_mae"] == 0.0


def test_compute_shared_schema_complete():
    det = _fake_detector([((10, 10, 30, 60), 0.9, 0)])
    ref = np.full((64, 64, 3), 120, dtype=np.uint8)
    m = S.compute_shared_metrics(
        result_image=_img(ref), source_image=_img(ref), reference_image=_img(ref),
        detector=det, expected_height=40.0, object_bbox=(20, 20, 40, 40),
    )
    for k in ("person_detected", "person_confidence", "object_added",
              "scale_ratio", "scale_error", "bg_mae", "bg_ssim"):
        assert k in m, k


def test_paired_shared_directions():
    a = [{"case_id": "c1", "seed": 1, "person_confidence": 0.5, "bg_mae": 2.0, "scale_error": 0.3}]
    b = [{"case_id": "c1", "seed": 1, "person_confidence": 0.8, "bg_mae": 3.0, "scale_error": 0.1}]
    s = S.paired_shared(a, b, label_a="base", label_b="lora")
    assert s["matched_pairs"] == 1
    assert s["metrics"]["person_confidence"]["improved"] is True     # +0.3, higher better
    assert s["metrics"]["bg_mae"]["improved"] is False               # +1.0, lower better
    assert s["metrics"]["scale_error"]["improved"] is True           # -0.2, lower better
