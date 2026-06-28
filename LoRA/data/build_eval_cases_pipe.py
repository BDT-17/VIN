"""Build a REAL frozen inpaint eval set from the PIPE dataset.

PIPE (paint-by-inpaint/PIPE) gives genuine before/after pairs:
    source_img  = object ERASED (the empty-ish background)   -> inpaint INPUT
    target_img  = the real photo WITH the object              -> GROUND TRUTH
    Instruction = what to add                                  -> prompt
PIPE ships NO mask column, so the object mask is derived as the thresholded,
dilated difference |target - source| (the changed region == the added object).

This is a real golden set (target_img is a real photo, not another model's
output), so outside_mask_* metrics computed against `reference` are meaningful.

Output (mirrors build_eval_cases):
    <work>/eval/pipe_eval_v1/{cases.jsonl, images/, masks/, reference/}
        images/<case>.png     = source_img      (input)
        reference/<case>.png  = target_img      (ground truth)
        masks/<case>.png      = derived object mask
"""

import json
from pathlib import Path

import numpy as np

import re

# words that mark a person/pedestrian instruction or class
_PERSON_WORDS = ("person", "people", "pedestrian", "man", "woman", "men",
                 "women", "boy", "girl", "child", "children", "kid", "human")
# whole-word match: substring matching let "mane" (horse), "german", "ornament"
# etc. slip through because they CONTAIN "man"/"men" — that is why horses/animals
# leaked into a "person-only" filter. \b anchors to word boundaries.
_PERSON_RE = re.compile(r"\b(" + "|".join(_PERSON_WORDS) + r")\b")


def _is_person(instruction_class: str, instruction_text: str) -> bool:
    blob = f"{instruction_class or ''} {instruction_text or ''}".lower()
    return _PERSON_RE.search(blob) is not None


def _is_person_class(instruction_class: str) -> bool:
    """Strict filter: the OBJECT CLASS itself must be a person.

    PIPE is an object-ADDITION dataset, so the action is always "add"; the
    ``Instruction_Class`` field holds the object category (person, dog, car...).
    Matching only this field — never the free-text VLM caption — avoids leaks
    like class="hat" whose caption mentions "a man's hat". Use this for the
    mask-free add-a-person flow; ``_is_person`` (class OR text) stays for the
    looser flows that want maximum recall.
    """
    return _PERSON_RE.search((instruction_class or "").lower()) is not None


def _derive_mask(source_rgb: np.ndarray, target_rgb: np.ndarray,
                 thresh: int = 25, dilate_px: int = 6) -> np.ndarray:
    """mask = dilate(|target - source| > thresh). Returns uint8 {0,255}."""
    diff = np.abs(target_rgb.astype(np.int16) - source_rgb.astype(np.int16)).max(axis=2)
    binary = (diff > thresh).astype(np.uint8) * 255
    if dilate_px > 0:
        from PIL import Image, ImageFilter
        m = Image.fromarray(binary).filter(ImageFilter.MaxFilter(dilate_px * 2 + 1))
        binary = np.asarray(m, dtype=np.uint8)
    return binary


def _bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return []
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def run_build_pipe_eval(work_dir, eval_set="pipe_eval_v1", split="test",
                        person_only=True, limit=None, thresh=25, dilate_px=6,
                        min_mask_pixels=64) -> Path:
    """Stream PIPE, filter person, derive masks, write a frozen eval set.

    Args:
        limit: cap number of cases (None = all matching). PIPE test == 752 rows.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("pip install datasets") from exc
    from PIL import Image

    eval_root = Path(work_dir) / "eval" / eval_set
    for sub in ("images", "masks", "reference"):
        (eval_root / sub).mkdir(parents=True, exist_ok=True)

    ds = load_dataset("paint-by-inpaint/PIPE", split=split, streaming=True)

    cases = []
    kept = 0
    for row in ds:
        instr = row.get("Instruction_VLM-LLM", "")
        cls = row.get("Instruction_Class", "")
        if person_only and not _is_person(cls, instr):
            continue

        source = row["source_img"].convert("RGB")
        target = row["target_img"].convert("RGB")
        if source.size != target.size:
            target = target.resize(source.size)
        s_arr = np.asarray(source)
        t_arr = np.asarray(target)
        mask = _derive_mask(s_arr, t_arr, thresh=thresh, dilate_px=dilate_px)
        if int((mask > 127).sum()) < min_mask_pixels:
            continue  # no meaningful change region -> skip

        case_id = f"pipe_{row.get('img_id','')}_{row.get('ann_id','')}_{kept}"
        source.save(eval_root / "images" / f"{case_id}.png")
        target.save(eval_root / "reference" / f"{case_id}.png")
        Image.fromarray(mask).save(eval_root / "masks" / f"{case_id}.png")

        cases.append({
            "case_id": case_id,
            "image_path": f"images/{case_id}.png",
            "mask_path": f"masks/{case_id}.png",
            "reference_path": f"reference/{case_id}.png",
            "expected_bbox_xyxy": _bbox_from_mask(mask),
            "prompt_fields": {"instruction": instr,
                              "object_location": row.get("object_location", "")},
            "source_split": split,
            "eval_set": eval_set,
            "instruction_class": cls,
            "frozen": True,
            "has_ground_truth": True,
        })
        kept += 1
        if limit is not None and kept >= limit:
            break

    (eval_root / "cases.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + ("\n" if cases else ""),
        encoding="utf-8")
    print(f"  build_pipe_eval: {kept} {'person ' if person_only else ''}cases "
          f"from PIPE/{split} -> {eval_root}")
    return eval_root
