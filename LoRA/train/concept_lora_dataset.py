"""PIPE data loader for the CONCEPT (text->image) LoRA trainer.

Goal: a plain SD3.5 LoRA that learns to GENERATE images like the PIPE
"add a person" subset — NOT an edit model. So we ignore the before/after pairing
entirely and keep only the FINISHED photo (``target_img``, the real image that
already contains the person). No mask, no source latent, no input_proj.

Filtering mirrors the mask-free flow: strictly on PIPE's ``Instruction_Class``
(``_is_person_class``, whole-word, class field only) so only person additions are
kept and a non-person caption that merely mentions a person does not leak in.

Each PIPE instruction is an EDIT imperative ("add a man in a red coat"); for a
text->image model we turn it into an image DESCRIPTION by stripping the leading
imperative verb ("add/place/put...") so the caption describes the photo, not the
edit. A trigger token / prefix can be prepended for prompt-time control.

Yields training items:
    {image (PIL RGB), caption (str)}
"""

import re
from typing import Iterator, Dict

import numpy as np

from ..data.build_eval_cases_pipe import _is_person_class, _derive_mask

# leading imperative verbs in PIPE instructions ("add a ...", "place the ...")
_IMPERATIVE_RE = re.compile(
    r"^\s*(please\s+)?(add|place|put|insert|include|introduce|spawn|draw|paint)"
    r"(\s+in|\s+on)?(\s+|$)", re.IGNORECASE)


def instruction_to_subject(instruction: str) -> str:
    """Turn a PIPE edit imperative into an image subject phrase.

    "add a man wearing a hat" -> "a man wearing a hat"
    Empty / unparseable instructions fall back to "a person".
    """
    s = (instruction or "").strip()
    s = _IMPERATIVE_RE.sub("", s).strip()
    return s or "a person"


def build_caption(instruction: str, caption_prefix: str = "a photo of ",
                  trigger_token: str = "") -> str:
    """caption = caption_prefix + [trigger ]+ subject. Both extras optional."""
    subject = instruction_to_subject(instruction)
    trig = f"{trigger_token.strip()} " if trigger_token.strip() else ""
    return f"{caption_prefix}{trig}{subject}".strip()


def iter_pipe_images(split="train", person_only=True, num_samples=4000,
                     caption_prefix="a photo of ", trigger_token="",
                     diff_thresh=25, min_change_pixels=64) -> Iterator[Dict]:
    """Yield up to num_samples {image, caption} items from PIPE.

    Only ``target_img`` (the finished photo) is used. The |target-source| diff is
    applied ONLY as a sanity filter to drop degenerate pairs whose "person" is too
    small to have actually been added — it is never returned or fed to the model.
    """
    from datasets import load_dataset

    ds = load_dataset("paint-by-inpaint/PIPE", split=split, streaming=True)
    kept = 0
    for row in ds:
        cls = row.get("Instruction_Class", "")
        if person_only and not _is_person_class(cls):
            continue
        target = row["target_img"].convert("RGB")
        source = row["source_img"].convert("RGB")
        if source.size != target.size:
            source = source.resize(target.size)

        diff = _derive_mask(np.asarray(source), np.asarray(target),
                            thresh=diff_thresh, dilate_px=0)
        if int((diff > 127).sum()) < min_change_pixels:
            continue

        caption = build_caption(row.get("Instruction_VLM-LLM", ""),
                                caption_prefix=caption_prefix,
                                trigger_token=trigger_token)
        yield {"image": target, "caption": caption}
        kept += 1
        if num_samples is not None and kept >= num_samples:
            break


def load_images(num_samples=4000, **kw) -> list:
    """Materialize a list of {image, caption} concept-LoRA items."""
    return list(iter_pipe_images(num_samples=num_samples, **kw))
