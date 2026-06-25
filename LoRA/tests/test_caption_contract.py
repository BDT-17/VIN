"""Caption contract: trigger token in every caption, no invented attributes."""

import pandas as pd

from LoRA.data.captions import render_caption, run_captions
from LoRA.data.config import CaptionConfig, load_prompt_config


def _prompts():
    return load_prompt_config()


def test_render_caption_drops_empty_fields():
    tmpl = "a photo of <vin_ped> pedestrian, {framing}, {view}, {scene}"
    out = render_caption(tmpl, {"framing": "full body", "view": "", "scene": "urban street scene"})
    assert out == "a photo of <vin_ped> pedestrian, full body, urban street scene"
    assert ", ," not in out


def test_every_caption_contains_trigger():
    prompts = _prompts()
    df = pd.DataFrame([
        {"source_id": "citypersons", "bbox_height_px": 200, "visible_ratio": 0.9},
        {"source_id": "mot17_02", "bbox_height_px": 120, "visible_ratio": 0.4},
        {"source_id": "human_detection", "bbox_height_px": 300, "visible_ratio": 1.0},
    ])
    out = run_captions(df, prompts, CaptionConfig())
    assert (out["caption"].str.contains(prompts.trigger_token, regex=False)).all()
    assert (out["trigger_token"] == prompts.trigger_token).all()


def test_caption_has_no_invented_attributes():
    prompts = _prompts()
    df = pd.DataFrame([{"source_id": "citypersons", "bbox_height_px": 200, "visible_ratio": 0.9}])
    cap = run_captions(df, prompts, CaptionConfig())["caption"].iloc[0].lower()
    for banned in ("male", "female", "woman", "man", "year-old", "happy", "sad", "doctor"):
        assert banned not in cap
