"""Standalone background preservation smoke test.

Run:
    python inpaint/test_background_preservation.py
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inpaint.config import DEFAULT_CONFIG
from inpaint.sd35_mask_refinement import bbox_to_mask, hard_restore_outside_mask, outside_mask_diff, refine_mask


def test_hard_restore_outside_mask():
    original_arr = np.zeros((64, 64, 3), dtype=np.uint8)
    original_arr[..., 0] = 20
    original_arr[..., 1] = 40
    original_arr[..., 2] = 80
    generated_arr = np.full((64, 64, 3), 220, dtype=np.uint8)
    original = Image.fromarray(original_arr, mode="RGB")
    generated = Image.fromarray(generated_arr, mode="RGB")
    bbox = (20, 18, 44, 48)
    bundle = refine_mask(bbox_to_mask((64, 64), bbox), bbox, DEFAULT_CONFIG)
    restored = hard_restore_outside_mask(original, generated, bundle)
    diff = outside_mask_diff(original, restored, bundle)
    assert diff == 0.0, f"outside_mask_diff should be 0 after hard restore, got {diff}"


if __name__ == "__main__":
    test_hard_restore_outside_mask()
    print("background preservation ok")
