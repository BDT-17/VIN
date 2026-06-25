"""Inpaint metric contract: no V5 metrics; background preservation behaves."""

import numpy as np
import pytest

from LoRA.inference import inpaint_metrics as M
from LoRA.inference.report import build_paired_comparison

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def _write(tmp, name, arr):
    p = tmp / name
    Image.fromarray(arr.astype("uint8")).save(p)
    return p


def test_no_v5_metrics_present():
    banned = {"placement_score", "affordance_score", "scale_correction", "harmonization_score"}
    assert banned.isdisjoint(set(M.METRIC_DIRECTIONS))


def test_identical_outside_mask_is_perfect(tmp_path):
    ref = np.full((64, 64, 3), 120, dtype=np.uint8)
    out = ref.copy()
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:40, 20:40] = 255  # inpaint region
    rp = _write(tmp_path, "ref.png", ref)
    op = _write(tmp_path, "out.png", out)
    mp = _write(tmp_path, "mask.png", np.stack([mask] * 3, axis=2))
    m = M.background_metrics(rp, op, mp, dilate_px=2)
    assert m["outside_mask_mae"] == 0.0
    assert m["outside_mask_ssim"] >= 0.99


def test_background_change_outside_mask_is_penalized(tmp_path):
    ref = np.full((64, 64, 3), 120, dtype=np.uint8)
    out = ref.copy()
    out[0:10, 0:10] = 0  # corruption OUTSIDE the mask
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:40, 20:40] = 255
    rp = _write(tmp_path, "ref.png", ref)
    op = _write(tmp_path, "out.png", out)
    mp = _write(tmp_path, "mask.png", np.stack([mask] * 3, axis=2))
    m = M.background_metrics(rp, op, mp, dilate_px=2)
    assert m["outside_mask_mae"] > 0.0


def test_paired_comparison_delta_direction(tmp_path):
    baseline = [{"case_id": "c1", "seed": 42, "person_confidence": 0.5, "outside_mask_mae": 2.0}]
    lora = [{"case_id": "c1", "seed": 42, "person_confidence": 0.8, "outside_mask_mae": 3.0}]
    summary = build_paired_comparison(baseline, lora, tmp_path)
    assert summary["matched_pairs"] == 1
    assert summary["metrics"]["person_confidence"]["improved"] is True   # higher better, +0.3
    assert summary["metrics"]["outside_mask_mae"]["improved"] is False   # lower better, +1.0
