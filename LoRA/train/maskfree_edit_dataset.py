"""PIPE data loader for the mask-FREE (IP2P/PIPE-style) edit trainer.

Unlike inpaint_edit_dataset, this yields NO mask: the mask-free model is
conditioned only on (source image, instruction) and must decide where/scale/pose
the object goes from context — exactly as in the PIPE paper (Wasserman et al.,
2404.18212). We still stream PIPE, filter to person instructions, and use the
|target-source| diff ONLY to drop pairs with no meaningful change (a degenerate
pair teaches nothing); the diff is never fed to the model.

Yields training items:
    {source(PIL RGB), target(PIL RGB), prompt(str)}

prompt = the raw PIPE instruction (no <vin_ped> trigger — the paper uses plain
natural-language instructions, and the pivot to mask-free dropped the trigger).
"""

from typing import Iterator, Dict

import numpy as np

from ..data.build_eval_cases_pipe import _is_person, _derive_mask


def iter_pipe_pairs(split="train", person_only=True, num_samples=4000,
                    diff_thresh=25, min_change_pixels=64) -> Iterator[Dict]:
    """Yield up to num_samples mask-free training items from PIPE.

    diff_thresh / min_change_pixels: a pair is kept only if |target-source|
    exceeds diff_thresh on at least min_change_pixels pixels (drops no-op pairs).
    The diff is a filter only — it is NOT returned or fed to the model.
    """
    from datasets import load_dataset
    from PIL import Image

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

        # change filter only (no dilation needed — we just need a yes/no signal)
        diff = _derive_mask(np.asarray(source), np.asarray(target),
                            thresh=diff_thresh, dilate_px=0)
        if int((diff > 127).sum()) < min_change_pixels:
            continue

        prompt = (instr or "add a person").strip()
        yield {"source": source, "target": target, "prompt": prompt}
        kept += 1
        if num_samples is not None and kept >= num_samples:
            break


def load_pairs(num_samples=4000, **kw) -> list:
    """Materialize a list of mask-free items."""
    return list(iter_pipe_pairs(num_samples=num_samples, **kw))
