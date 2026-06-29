"""Segment-and-paste compositor for the concept -> paste pipeline.

Pipeline:
  1. the concept (text->image) model generates a FULL image containing a person
     from the prompt alone (sd35_concept_runner.generate) — it does NOT see the
     original.
  2. YOLOv8-seg segments the person out of that generated image
     (person_detector.load_person_segmenter).
  3. this module pastes that person onto the ORIGINAL background, feathering the
     seam + matching colour to the original so the paste does not read as a sticker.

Why this composite is needed: the generated background is the model's own scene,
not the original. Pasting only the segmented person preserves the original
background byte-exact OUTSIDE the person ("100% background preserve") by composite
instead of hard-restore. The cost is harmonisation: the person was lit by the
*generated* scene, so we colour-match it to the original.

Pure numpy + PIL (no torch, no OpenCV hard dependency); Poisson blending is
offered as an optional OpenCV path. CPU-cheap — runs anywhere.
"""

from pathlib import Path
from typing import List, Dict, Optional

import numpy as np


def _to_rgb_array(img, size=None):
    from PIL import Image
    if isinstance(img, (str, Path)):
        img = Image.open(img)
    img = img.convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size)
    return np.asarray(img, dtype=np.float32), img.size  # (H,W,3), (W,H)


def _feather(mask_bool: np.ndarray, px: int) -> np.ndarray:
    """Soft alpha in [0,1] from a binary mask: erode-ish blur of the edge.

    A Gaussian blur of the hard mask gives a gradient band straddling the edge;
    that band is the feather. px==0 returns the hard mask as float.
    """
    alpha = mask_bool.astype(np.float32)
    if px <= 0:
        return alpha
    from PIL import Image, ImageFilter
    a = Image.fromarray((alpha * 255).astype("uint8"))
    a = a.filter(ImageFilter.GaussianBlur(radius=px))
    return np.asarray(a, dtype=np.float32) / 255.0


def _erode(mask_bool: np.ndarray, px: int) -> np.ndarray:
    """Shrink the mask inward by ``px`` (morphological erosion via PIL MinFilter).

    YOLO person masks usually include a thin ring of background pixels around the
    silhouette; eroding a pixel or two pulls the cut strictly INSIDE the person so
    no background halo is carried over — the "chặt" (tight) half of a clean cut.
    """
    if px <= 0:
        return mask_bool
    from PIL import Image, ImageFilter
    m = Image.fromarray((mask_bool.astype(np.uint8) * 255))
    m = m.filter(ImageFilter.MinFilter(px * 2 + 1))
    return np.asarray(m) > 127


def _match_color(person_rgb, bg_rgb, alpha, strength=0.5):
    """Shift the person's per-channel mean/std toward the background region it
    will cover, so its colour/exposure sits in the original scene.

    strength in [0,1]: 0 = no correction (raw paste), 1 = full reinhard transfer.
    Operates only where alpha>0 to estimate the person's own statistics.
    """
    if strength <= 0:
        return person_rgb
    sel = alpha > 0.1
    if sel.sum() < 16:
        return person_rgb
    out = person_rgb.copy()
    for c in range(3):
        pc = person_rgb[..., c][sel]
        bc = bg_rgb[..., c][sel]  # the original pixels under the person
        ps, bs = pc.std() + 1e-6, bc.std() + 1e-6
        pm, bm = pc.mean(), bc.mean()
        # reinhard: normalise person, rescale to bg stats; blend by strength
        shifted = (person_rgb[..., c] - pm) / ps * bs + bm
        out[..., c] = (1 - strength) * person_rgb[..., c] + strength * shifted
    return np.clip(out, 0, 255)


def composite_persons(
    original,
    generated,
    persons: List[Dict],
    feather_px: int = 2,
    erode_px: int = 1,
    color_match: float = 0.5,
    min_conf: float = 0.25,
    min_area_frac: float = 0.001,
    poisson: bool = False,
):
    """Paste segmented persons from ``generated`` onto ``original``.

    original / generated : path or PIL.Image. Resized to a common size (the
        original's size) so masks align — the concept runner outputs square
        ``resolution`` frames, so pass the original at that size or accept resize.
    persons : output of ``load_person_segmenter()(generated)`` — each has
        ``mask`` (HxW bool at generated's native size), ``bbox_xyxy``, ``conf``.
    erode_px : shrink the mask inward this many px before feathering, so the cut
        sits strictly inside the silhouette (drops the YOLO background halo —
        a tighter, cleaner cut). 0 = no erosion.
    feather_px : Gaussian feather radius on the paste alpha (seam softness). Keep
        small for a crisp cut; 0 = hard edge (byte-exact seam).
    color_match : 0..1 reinhard colour transfer toward the covered background.
    min_conf / min_area_frac : drop low-confidence or tiny specks.
    poisson : if True and OpenCV is available, use seamlessClone for the largest
        person instead of alpha blend (smoother lighting, may shift hue).

    Returns (composited PIL.Image, info dict).
    """
    from PIL import Image

    orig_arr, size = _to_rgb_array(original)               # (H,W,3)
    H, W = orig_arr.shape[:2]
    gen_arr, _ = _to_rgb_array(generated, size=size)       # align to original

    kept = [p for p in persons
            if p["conf"] >= min_conf
            and float(np.asarray(p["mask"]).sum()) / (H * W) >= min_area_frac]
    if not kept:
        return Image.fromarray(orig_arr.astype("uint8")), {"pasted": 0, "reason": "no person"}

    out = orig_arr.copy()
    used_poisson = False
    for idx, p in enumerate(kept):
        mask = np.asarray(p["mask"], dtype=bool)
        if mask.shape != (H, W):                            # mask from generated's native size
            m_img = Image.fromarray((mask.astype("uint8") * 255)).resize((W, H))
            mask = np.asarray(m_img) > 127
        mask = _erode(mask, erode_px)                       # tighten: drop bg halo
        alpha = _feather(mask, feather_px)[..., None]       # (H,W,1)

        person = gen_arr.copy()
        person = _match_color(person, out, alpha[..., 0], strength=color_match)

        if poisson and idx == 0:
            blended = _poisson_blend(out, person, mask)
            if blended is not None:
                out = blended
                used_poisson = True
                continue
        out = alpha * person + (1 - alpha) * out

    return Image.fromarray(np.clip(out, 0, 255).astype("uint8")), {
        "pasted": len(kept),
        "confs": [round(p["conf"], 3) for p in kept],
        "poisson": used_poisson,
    }


def _poisson_blend(bg_arr, person_arr, mask_bool):
    """OpenCV seamlessClone of the masked person onto bg. None if cv2 absent."""
    try:
        import cv2
    except ImportError:
        return None
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    cx, cy = int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2)
    src = person_arr.astype("uint8")[..., ::-1]    # RGB->BGR
    dst = bg_arr.astype("uint8")[..., ::-1]
    m = (mask_bool.astype("uint8") * 255)
    blended = cv2.seamlessClone(src, dst, m, (cx, cy), cv2.NORMAL_CLONE)
    return blended[..., ::-1].astype(np.float32)   # BGR->RGB


def generate_and_paste_concept(
    runner,
    original_img,
    prompt,
    segmenter,
    seed: int = 42,
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
    resolution: int = 512,
    feather_px: int = 2,
    erode_px: int = 1,
    color_match: float = 0.5,
    poisson: bool = False,
):
    """End-to-end one-image step for the CONCEPT (text->image) LoRA:
    generate (from prompt only, NO source) -> segment person -> paste onto original.

    The concept runner does not see the original — it generates a person in its own
    scene from the text prompt. We then segment that person and paste it
    onto ``original_img``, so the original background is preserved (byte-exact
    outside the person; the ``feather_px`` band blends the silhouette edge — set
    ``feather_px=0`` for a strictly byte-exact background).

    runner    : an SD35ConceptRunner (already loaded).
    segmenter : load_person_segmenter() callable.

    Returns (composited PIL.Image, generated PIL.Image, info). The placement/scale
    of the person come from the generation (the concept model is blind to the
    original), so use the prompt to steer a person that fits the target scene.
    """
    from PIL import Image
    orig = original_img
    if isinstance(orig, (str, Path)):
        orig = Image.open(orig)
    orig = orig.convert("RGB")

    generated = runner.generate(
        prompt, seed=seed, num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale, resolution=resolution,
    )
    persons = segmenter(generated)
    composite, info = composite_persons(
        orig, generated, persons,
        feather_px=feather_px, erode_px=erode_px, color_match=color_match, poisson=poisson,
    )
    return composite, generated, info
