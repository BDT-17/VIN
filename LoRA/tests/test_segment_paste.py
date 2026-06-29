"""Segment-and-paste compositor contract.

The mask-free -> paste pipeline's defining guarantee: OUTSIDE the pasted person
the original background is preserved byte-exact (composite, not hard-restore),
while INSIDE the person region the generated pixels win. No model needed —
synthetic person mask + frames exercise the numpy/PIL compositor directly.
"""

import numpy as np
import pytest

from LoRA.inference.segment_paste import composite_persons, _feather

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def _frames():
    H = W = 64
    orig = Image.fromarray(np.full((H, W, 3), 120, np.uint8))
    gen = np.full((H, W, 3), 120, np.uint8)
    gen[20:50, 25:40] = [200, 30, 30]      # a red "person"
    mask = np.zeros((H, W), bool)
    mask[20:50, 25:40] = True
    return orig, Image.fromarray(gen), mask


def test_background_outside_person_is_byte_exact():
    orig, gen, mask = _frames()
    out, info = composite_persons(orig, gen, [{"mask": mask, "bbox_xyxy": [25, 20, 40, 50], "conf": 0.9}],
                                  feather_px=0, color_match=0.0)
    a = np.asarray(out)
    assert info["pasted"] == 1
    # a pixel well outside the person must equal the original exactly
    assert tuple(a[5, 5]) == (120, 120, 120)
    # a pixel inside the person must be the generated red
    assert a[35, 32][0] > a[35, 32][1] and a[35, 32][0] > a[35, 32][2]


def test_low_conf_and_tiny_specks_dropped():
    orig, gen, mask = _frames()
    _, info_lc = composite_persons(orig, gen, [{"mask": mask, "bbox_xyxy": [25, 20, 40, 50], "conf": 0.1}])
    assert info_lc["pasted"] == 0
    tiny = np.zeros((64, 64), bool); tiny[0, 0] = True
    _, info_t = composite_persons(orig, gen, [{"mask": tiny, "bbox_xyxy": [0, 0, 1, 1], "conf": 0.9}])
    assert info_t["pasted"] == 0


def test_no_person_returns_original():
    orig, gen, _ = _frames()
    out, info = composite_persons(orig, gen, [])
    assert info["pasted"] == 0
    assert np.array_equal(np.asarray(out), np.asarray(orig.convert("RGB")))


def test_feather_produces_soft_edge():
    mask = np.zeros((64, 64), bool); mask[20:50, 25:40] = True
    alpha = _feather(mask, 3)
    assert alpha.min() >= 0.0 and alpha.max() <= 1.0
    # a feathered edge has intermediate (partial) alpha values
    assert ((alpha > 0.0) & (alpha < 1.0)).any()


def test_color_match_pulls_toward_background():
    orig, gen, mask = _frames()
    out, _ = composite_persons(orig, gen, [{"mask": mask, "bbox_xyxy": [25, 20, 40, 50], "conf": 0.9}],
                               feather_px=0, color_match=1.0)
    a = np.asarray(out).astype(np.float32)
    # with full colour-match toward gray-120 bg, the pasted red is less saturated
    # than the raw generated red (200,30,30) — its red channel drops toward 120.
    assert a[35, 32][0] < 200


def test_erode_tightens_cut_and_drops_boundary():
    """erode_px shrinks the mask inward, so the silhouette's outer ring reverts to
    the original background (no YOLO halo carried over) while the interior stays."""
    orig, gen, mask = _frames()   # person rect rows 20..49, cols 25..39
    out, info = composite_persons(
        orig, gen, [{"mask": mask, "bbox_xyxy": [25, 20, 40, 50], "conf": 0.9}],
        feather_px=0, erode_px=3, color_match=0.0)
    a = np.asarray(out)
    assert info["pasted"] == 1
    assert tuple(a[20, 25]) == (120, 120, 120)   # original corner pixel eroded away -> bg
    assert a[35, 32][0] > a[35, 32][1]           # interior still the generated person


def test_generate_and_paste_concept_uses_generation_and_preserves_bg():
    """The concept variant: runner.generate (no source) -> segment -> paste.

    A fake runner returns a fixed generated frame and a fake segmenter returns a
    person mask, so the numpy compositor path is exercised end-to-end without torch.
    """
    from LoRA.inference.segment_paste import generate_and_paste_concept

    orig, gen, mask = _frames()

    class FakeRunner:
        def __init__(self):
            self.called_with = None

        def generate(self, prompt, **kw):
            self.called_with = (prompt, kw)
            return gen  # the concept model's full generated frame

    def fake_segmenter(image):
        return [{"mask": mask, "bbox_xyxy": [25, 20, 40, 50], "conf": 0.9}]

    runner = FakeRunner()
    composite, generated, info = generate_and_paste_concept(
        runner, orig, "a photo of a person", fake_segmenter,
        feather_px=0, color_match=0.0)

    # generation got the prompt (no source image passed — concept is blind to orig)
    assert runner.called_with[0] == "a photo of a person"
    assert info["pasted"] == 1
    a = np.asarray(composite)
    assert tuple(a[5, 5]) == (120, 120, 120)          # bg byte-exact outside person
    assert a[35, 32][0] > a[35, 32][1]                # generated person inside
