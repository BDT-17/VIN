"""Tests for caption generation and validation."""

import pytest
import pandas as pd

from LoRA.data.captions import CaptionGenerator, validate_captions


@pytest.fixture
def generator():
    return CaptionGenerator(trigger_token="<vin_ped>")


def test_generate_caption_contains_trigger(generator):
    caption = generator.generate_caption(
        bbox_height_ratio=0.3,
        visible_ratio=0.8,
        occlusion_level=0.1,
    )
    assert "<vin_ped>" in caption


def test_generate_caption_occluded(generator):
    caption = generator.generate_caption(
        bbox_height_ratio=0.3,
        visible_ratio=0.5,
        occlusion_level=0.6,
    )
    assert "occluded" in caption


def test_generate_caption_cctv_domain(generator):
    caption = generator.generate_caption(
        bbox_height_ratio=0.3,
        visible_ratio=0.9,
        occlusion_level=0.0,
        camera_domain="cctv_outdoor",
    )
    assert "CCTV" in caption


def test_validate_captions_passes(generator):
    df = pd.DataFrame([{
        "sample_id": "s1",
        "caption": "photo of <vin_ped> a pedestrian, full body, clear, in urban street",
    }])
    errors = validate_captions(df, trigger_token="<vin_ped>")
    assert errors == []


def test_validate_captions_missing_trigger():
    df = pd.DataFrame([{
        "sample_id": "s1",
        "caption": "photo of a pedestrian, full body",
    }])
    errors = validate_captions(df, trigger_token="<vin_ped>")
    assert any("<vin_ped>" in e for e in errors)


def test_validate_captions_too_short():
    df = pd.DataFrame([{
        "sample_id": "s1",
        "caption": "<vin_ped> ped",
    }])
    errors = validate_captions(df, trigger_token="<vin_ped>", min_tokens=5)
    assert any("short" in e for e in errors)


def test_validate_captions_prohibited_term():
    df = pd.DataFrame([{
        "sample_id": "s1",
        "caption": "photo of <vin_ped> a pedestrian showing emotion, clear",
    }])
    errors = validate_captions(df, trigger_token="<vin_ped>")
    assert any("emotion" in e for e in errors)
