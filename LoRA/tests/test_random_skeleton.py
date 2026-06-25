"""Random skeleton/mask generator — constraints + determinism (no GPU)."""

import numpy as np
import pytest

pytest.importorskip("PIL")

from LoRA.inference.random_skeleton import (
    PlacementConfig, sample_person_bbox, random_placement, bbox_mask, _rng)


def test_bbox_within_image_and_constrained():
    W, H = 512, 512
    cfg = PlacementConfig()
    rng = _rng(42, 0)
    for _ in range(50):
        x1, y1, x2, y2 = sample_person_bbox(W, H, rng, cfg)
        assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H        # inside, valid
        h = y2 - y1
        assert cfg.min_height_frac * H * 0.9 <= h <= cfg.max_height_frac * H * 1.1
        assert y2 <= cfg.foot_y_max_frac * H + 1               # feet in lower band


def test_aspect_ratio_roughly_person():
    W, H = 512, 512
    x1, y1, x2, y2 = sample_person_bbox(W, H, _rng(1, 1), PlacementConfig())
    ar = (x2 - x1) / (y2 - y1)
    assert 0.2 <= ar <= 0.6   # standing-person-ish


def test_deterministic_same_seed_index():
    a = random_placement((512, 512), seed=7, index=3)[1]
    b = random_placement((512, 512), seed=7, index=3)[1]
    assert a == b
    c = random_placement((512, 512), seed=7, index=4)[1]
    assert a != c   # different index -> different placement


def test_mask_is_white_inside_black_outside():
    mask = bbox_mask((100, 100), [(20, 30, 60, 90)], pad_frac=0.0)
    arr = np.asarray(mask)
    assert arr[60, 40] == 255      # inside bbox
    assert arr[5, 5] == 0          # outside
    assert arr.max() == 255 and arr.min() == 0


def test_count_within_config():
    cfg = PlacementConfig(min_people=1, max_people=3)
    _, bboxes, sk = random_placement((512, 512), seed=99, index=0, cfg=cfg)
    assert 1 <= len(bboxes) <= 3
    assert len(sk) == len(bboxes)


def test_silhouette_is_person_shaped_not_full_bbox():
    from LoRA.inference.random_skeleton import silhouette_mask
    W, H = 256, 256
    bboxes = [(80, 40, 140, 200)]
    from LoRA.inference.random_skeleton import skeleton_points
    sk = [skeleton_points(b) for b in bboxes]
    m = np.asarray(silhouette_mask((W, H), bboxes, sk, dilate_px=2))
    white = (m > 127)
    # head region masked
    assert white[55:70, 100:120].any()
    # nothing outside the image / outside the person column
    assert not white[:, :60].any() and not white[:, 170:].any()
    # silhouette covers LESS than the full bbox rectangle (person-shaped)
    bbox_area = (140 - 80) * (200 - 40)
    assert white.sum() < 0.95 * bbox_area


def test_random_placement_mask_shapes_differ():
    bb, _, _ = random_placement((256, 256), seed=5, index=0, mask_shape='bbox')
    si, _, _ = random_placement((256, 256), seed=5, index=0, mask_shape='silhouette')
    a = np.asarray(bb) > 127
    b = np.asarray(si) > 127
    # same placement (same seed/index) but silhouette fills fewer pixels than bbox
    assert b.sum() < a.sum()
