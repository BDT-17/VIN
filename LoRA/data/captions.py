"""Stage 05 — per-image captions.

Caption = caption_template with the trigger token, plus only the descriptive
fields we can derive from annotations (framing/occlusion). We do NOT invent
gender, age, profession, or emotion. The trigger token MUST appear in every
caption.
"""

from pathlib import Path
from typing import Dict

import pandas as pd

from .config import CaptionConfig, PromptConfig


def _fields_from_sample(row) -> Dict[str, str]:
    bbox_h = float(row.get("bbox_height_px", 0) or 0)
    vis = float(row.get("visible_ratio", 1.0) or 1.0)
    framing = "full body" if vis >= 0.85 else "partially visible body"
    occlusion = "partially occluded" if vis < 0.6 else ""
    view = ""  # unknown from annotations -> dropped
    pose = ""
    scene = {
        "citypersons": "urban street scene",
        "mot17_02": "surveillance street scene",
        "human_detection": "real-world scene",
    }.get(row.get("source_id", ""), "")
    lighting = "natural lighting"
    return {"framing": framing, "view": view, "pose": pose,
            "occlusion": occlusion, "scene": scene, "lighting": lighting}


def render_caption(template: str, fields: Dict[str, str]) -> str:
    text = template
    for key, val in fields.items():
        text = text.replace("{" + key + "}", val or "")
    # collapse empty comma fragments and whitespace
    parts = [p.strip() for p in text.split(",")]
    parts = [p for p in parts if p]
    return ", ".join(parts)


def run_captions(candidates_df: pd.DataFrame, prompts: PromptConfig,
                 caption_cfg: CaptionConfig) -> pd.DataFrame:
    df = candidates_df.copy()
    captions, triggers = [], []
    for _, row in df.iterrows():
        fields = _fields_from_sample(row)
        cap = render_caption(prompts.caption_template, fields)
        if prompts.trigger_token not in cap:
            cap = f"a photo of {prompts.trigger_token} {prompts.class_token}, " + cap
        captions.append(cap)
        triggers.append(prompts.trigger_token)
    df["caption"] = captions
    df["trigger_token"] = triggers
    n_tokens = df["caption"].str.split().str.len()
    print(f"  captions: {len(df)} written, median {int(n_tokens.median())} tokens, "
          f"trigger '{prompts.trigger_token}' in 100%")
    return df
