"""PIPE eval builder: person filter + diff-derived mask (no HF download)."""

import numpy as np
import pytest

from LoRA.data.build_eval_cases_pipe import _is_person, _derive_mask, _bbox_from_mask

pytest.importorskip("PIL")


def test_person_filter_matches_person_words():
    assert _is_person("person", "Add a person near the car")
    assert _is_person("woman", "")
    assert _is_person("", "a pedestrian crossing the road")


def test_person_filter_rejects_non_person():
    assert not _is_person("dog", "Add a dog on the grass")
    assert not _is_person("bicycle", "place a bicycle at the left")


def test_derive_mask_marks_changed_region_only():
    src = np.full((64, 64, 3), 100, dtype=np.uint8)
    tgt = src.copy()
    tgt[20:40, 25:45] = 200  # "added object" region
    mask = _derive_mask(src, tgt, thresh=25, dilate_px=0)
    # changed region is masked
    assert mask[30, 35] == 255
    # untouched background is not
    assert mask[5, 5] == 0


def test_derive_mask_identical_images_is_empty():
    src = np.full((32, 32, 3), 120, dtype=np.uint8)
    mask = _derive_mask(src, src.copy(), thresh=25, dilate_px=0)
    assert int((mask > 127).sum()) == 0


def test_bbox_from_mask():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:30, 15:35] = 255
    x1, y1, x2, y2 = _bbox_from_mask(mask)
    assert (x1, y1) == (15.0, 10.0)
    assert (x2, y2) == (34.0, 29.0)


def test_bbox_from_empty_mask_is_empty():
    assert _bbox_from_mask(np.zeros((10, 10), dtype=np.uint8)) == []
