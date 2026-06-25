"""PIPE-train data loader for SD3.5 inpaint-edit LoRA training.

Streams paint-by-inpaint/PIPE, filters to person instructions, derives the edit
mask from |target - source| (PIPE ships no mask), and yields training items:
    {source(PIL), target(PIL), mask(PIL L), prompt(str)}

Reuses the same person-filter + diff-mask logic as the eval builder so train and
eval masks are derived identically.
"""

from typing import Iterator, Dict

import numpy as np

from ..data.build_eval_cases_pipe import _is_person, _derive_mask


def iter_pipe_pairs(split="train", person_only=True, num_samples=4000,
                    thresh=25, dilate_px=6, min_mask_pixels=64,
                    trigger_token="<vin_ped>", class_token="pedestrian") -> Iterator[Dict]:
    """Yield up to num_samples training items from PIPE."""
    from datasets import load_dataset

    ds = load_dataset("paint-by-inpaint/PIPE", split=split, streaming=True)
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
        mask = _derive_mask(np.asarray(source), np.asarray(target),
                            thresh=thresh, dilate_px=dilate_px)
        if int((mask > 127).sum()) < min_mask_pixels:
            continue

        from PIL import Image
        prompt = f"a photo of {trigger_token} {class_token}, " + (instr or "a person")
        yield {"source": source, "target": target,
               "mask": Image.fromarray(mask), "prompt": prompt}
        kept += 1
        if num_samples is not None and kept >= num_samples:
            break


def load_pairs(num_samples=4000, **kw) -> list:
    """Materialize a list of items (small num_samples / pre-cached use)."""
    return list(iter_pipe_pairs(num_samples=num_samples, **kw))
