"""Random (constrained) skeleton + mask generator for placing people.

For the inpaint-EDIT flow, the mask decides WHERE a person goes. Instead of
hand-drawing masks, this samples plausible person placements: a simple stick
skeleton (head-torso-legs-arms) at a random ground position and height, then
returns the BBOX-rectangle mask around it (a solid white region the editor fills).

Constraints keep placements realistic (no floating / oversized people):
  - feet land in the lower band of the image
  - person height is a fraction of image height
  - width follows a person aspect ratio

Deterministic given (seed, index) — no Math.random reliance.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class PlacementConfig:
    min_height_frac: float = 0.25     # person height as fraction of image H
    max_height_frac: float = 0.55
    aspect: float = 0.36              # width / height of a standing person
    foot_y_min_frac: float = 0.55     # feet land between these (fraction of H)
    foot_y_max_frac: float = 0.92
    bbox_pad_frac: float = 0.08       # padding around the skeleton bbox
    min_people: int = 1
    max_people: int = 3


def _rng(seed: int, index: int) -> np.random.RandomState:
    return np.random.RandomState((int(seed) * 100003 + int(index) * 9176 + 1) % (2**31 - 1))


def sample_person_bbox(W: int, H: int, rng: np.random.RandomState,
                       cfg: PlacementConfig) -> Tuple[int, int, int, int]:
    """One constrained person bbox -> (x1, y1, x2, y2)."""
    h = rng.uniform(cfg.min_height_frac, cfg.max_height_frac) * H
    w = h * cfg.aspect
    foot_y = rng.uniform(cfg.foot_y_min_frac, cfg.foot_y_max_frac) * H
    cx = rng.uniform(w / 2, W - w / 2)
    x1, x2 = cx - w / 2, cx + w / 2
    y2, y1 = foot_y, foot_y - h
    return int(max(0, x1)), int(max(0, y1)), int(min(W, x2)), int(min(H, y2))


def skeleton_points(bbox: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
    """A minimal stick skeleton inside the bbox (for optional visualization)."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    h = y2 - y1
    head = (cx, y1 + 0.10 * h)
    neck = (cx, y1 + 0.22 * h)
    hip = (cx, y1 + 0.55 * h)
    sh_l, sh_r = (x1 + 0.2 * (x2 - x1), neck[1]), (x2 - 0.2 * (x2 - x1), neck[1])
    hand_l, hand_r = (x1, y1 + 0.5 * h), (x2, y1 + 0.5 * h)
    foot_l, foot_r = (cx - 0.18 * (x2 - x1), y2), (cx + 0.18 * (x2 - x1), y2)
    return [tuple(map(int, p)) for p in
            (head, neck, hip, sh_l, sh_r, hand_l, hand_r, foot_l, foot_r)]


def bbox_mask(size: Tuple[int, int], bboxes: List[Tuple[int, int, int, int]],
              pad_frac: float = 0.08) -> "Image.Image":
    """White rectangle(s) over each padded bbox; black elsewhere."""
    from PIL import Image, ImageDraw
    W, H = size
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)
    for (x1, y1, x2, y2) in bboxes:
        pw, ph = (x2 - x1) * pad_frac, (y2 - y1) * pad_frac
        d.rectangle([max(0, x1 - pw), max(0, y1 - ph),
                     min(W, x2 + pw), min(H, y2 + ph)], fill=255)
    return mask


def _capsule(draw, p, q, r):
    """Thick rounded line (limb) from p to q with radius r."""
    draw.line([p, q], fill=255, width=int(max(1, 2 * r)))
    for c in (p, q):
        draw.ellipse([c[0]-r, c[1]-r, c[0]+r, c[1]+r], fill=255)


def silhouette_mask(size: Tuple[int, int], bboxes, skeletons,
                    dilate_px: int = 6) -> "Image.Image":
    """Person-shaped mask: ellipse head/torso + capsule limbs around each
    skeleton, dilated. Tighter than a bbox so the editor fills a human shape
    instead of a rectangle.

    skeleton order (from skeleton_points):
        0 head, 1 neck, 2 hip, 3 sh_l, 4 sh_r, 5 hand_l, 6 hand_r, 7 foot_l, 8 foot_r
    """
    from PIL import Image, ImageDraw, ImageFilter
    W, H = size
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)
    for (x1, y1, x2, y2), sk in zip(bboxes, skeletons):
        bw, bh = (x2 - x1), (y2 - y1)
        head, neck, hip, sh_l, sh_r, hand_l, hand_r, foot_l, foot_r = sk
        head_r = max(2, int(bw * 0.18))
        limb_r = max(2, int(bw * 0.12))
        # head
        d.ellipse([head[0]-head_r, head[1]-head_r, head[0]+head_r, head[1]+head_r], fill=255)
        # torso (neck -> hip) as a wide ellipse
        torso_w = bw * 0.5
        d.ellipse([neck[0]-torso_w/2, neck[1], hip[0]+torso_w/2, hip[1]], fill=255)
        # arms + legs as capsules
        _capsule(d, neck, sh_l, limb_r); _capsule(d, sh_l, hand_l, limb_r)
        _capsule(d, neck, sh_r, limb_r); _capsule(d, sh_r, hand_r, limb_r)
        _capsule(d, hip, foot_l, limb_r); _capsule(d, hip, foot_r, limb_r)
    if dilate_px > 0:
        mask = mask.filter(ImageFilter.MaxFilter(dilate_px * 2 + 1))
    return mask


def random_placement(image_size: Tuple[int, int], seed: int, index: int = 0,
                     cfg: PlacementConfig = None, mask_shape: str = "silhouette"):
    """Sample 1..N constrained person placements + a mask.

    mask_shape: 'silhouette' (person-shaped, recommended) or 'bbox' (rectangle).
    Returns: (mask: PIL 'L', bboxes: list, skeletons: list[list[pt]])
    """
    cfg = cfg or PlacementConfig()
    W, H = image_size
    rng = _rng(seed, index)
    n = rng.randint(cfg.min_people, cfg.max_people + 1)
    bboxes = [sample_person_bbox(W, H, rng, cfg) for _ in range(n)]
    skeletons = [skeleton_points(b) for b in bboxes]
    if mask_shape == "bbox":
        mask = bbox_mask((W, H), bboxes, pad_frac=cfg.bbox_pad_frac)
    else:
        mask = silhouette_mask((W, H), bboxes, skeletons)
    return mask, bboxes, skeletons


def draw_skeleton_overlay(image, bboxes, skeletons):
    """Optional debug overlay of bboxes + skeleton sticks on a copy of image."""
    from PIL import ImageDraw
    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    bones = [(0, 1), (1, 2), (1, 3), (1, 4), (3, 5), (4, 6), (2, 7), (2, 8)]
    for b, sk in zip(bboxes, skeletons):
        d.rectangle(list(b), outline=(0, 255, 0), width=2)
        for i, j in bones:
            d.line([sk[i], sk[j]], fill=(255, 0, 0), width=2)
        for p in sk:
            d.ellipse([p[0]-2, p[1]-2, p[0]+2, p[1]+2], fill=(255, 255, 0))
    return img
