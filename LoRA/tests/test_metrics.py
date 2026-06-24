"""Tests for release metrics and benchmark comparison."""

import pytest
import pandas as pd

from LoRA.data.metrics import summarize_release_metrics, compare_downstream_benchmarks


@pytest.fixture
def clean_dataframes():
    """Minimal clean dataframes with zero leakage."""
    samples = pd.DataFrame([
        {"sample_id": f"s{i}", "image_id": f"img_{i}", "role": "lora_positive",
         "split": "train" if i < 6 else ("val" if i < 8 else "test"),
         "split_group_id": f"g{i}", "source_id": "src1",
         "caption": f"photo of <vin_ped> a pedestrian {i}", "trigger_token": "<vin_ped>",
         "quality_score": 0.8, "bbox_height_ratio": 0.2, "visible_ratio": 0.9,
         "occlusion_level": 0.1}
        for i in range(10)
    ])

    images = pd.DataFrame([
        {"image_id": f"img_{i}", "source_id": "src1"} for i in range(10)
    ])

    groups = pd.DataFrame([
        {"split_group_id": f"g{i}", "dedupe_status": "unique"} for i in range(10)
    ])

    return samples, images, groups


def test_summarize_sample_counts(clean_dataframes):
    samples, images, groups = clean_dataframes
    result = summarize_release_metrics(samples, images, groups)

    counts = result["sample_counts"]
    assert counts["total"] == 10
    assert counts["lora_positive"] == 10
    assert counts["benchmark_frozen"] == 0


def test_summarize_zero_leakage(clean_dataframes):
    samples, images, groups = clean_dataframes
    result = summarize_release_metrics(samples, images, groups)

    safety = result["split_safety"]
    assert safety["zero_leakage"] is True
    assert safety["cross_train_val"] == 0
    assert safety["cross_train_test"] == 0


def test_summarize_detects_leakage():
    """Cross-split group should set zero_leakage=False."""
    samples = pd.DataFrame([
        {"sample_id": "s0", "image_id": "img_0", "role": "lora_positive",
         "split": "train", "split_group_id": "shared_group", "source_id": "src1",
         "caption": "photo of <vin_ped> a pedestrian", "trigger_token": "<vin_ped>",
         "quality_score": 0.8, "bbox_height_ratio": 0.2, "visible_ratio": 0.9, "occlusion_level": 0.1},
        {"sample_id": "s1", "image_id": "img_1", "role": "lora_positive",
         "split": "val", "split_group_id": "shared_group", "source_id": "src1",
         "caption": "photo of <vin_ped> a pedestrian", "trigger_token": "<vin_ped>",
         "quality_score": 0.8, "bbox_height_ratio": 0.2, "visible_ratio": 0.9, "occlusion_level": 0.1},
    ])
    images = pd.DataFrame([{"image_id": f"img_{i}", "source_id": "src1"} for i in range(2)])
    groups = pd.DataFrame([{"split_group_id": "shared_group", "dedupe_status": "unique"}])

    result = summarize_release_metrics(samples, images, groups)
    assert result["split_safety"]["zero_leakage"] is False
    assert result["split_safety"]["cross_train_val"] == 1


def test_summarize_trigger_missing_count():
    samples = pd.DataFrame([
        {"sample_id": "s0", "image_id": "img_0", "role": "lora_positive", "split": "train",
         "split_group_id": "g0", "source_id": "src1",
         "caption": "photo of <vin_ped> a pedestrian", "trigger_token": "<vin_ped>",
         "quality_score": 0.8, "bbox_height_ratio": 0.2, "visible_ratio": 0.9, "occlusion_level": 0.1},
        {"sample_id": "s1", "image_id": "img_1", "role": "lora_positive", "split": "train",
         "split_group_id": "g1", "source_id": "src1",
         "caption": "photo of a pedestrian without trigger", "trigger_token": "<vin_ped>",
         "quality_score": 0.8, "bbox_height_ratio": 0.2, "visible_ratio": 0.9, "occlusion_level": 0.1},
    ])
    images = pd.DataFrame([{"image_id": f"img_{i}", "source_id": "src1"} for i in range(2)])
    groups = pd.DataFrame([{"split_group_id": f"g{i}", "dedupe_status": "unique"} for i in range(2)])

    result = summarize_release_metrics(samples, images, groups)
    assert result["captions"]["trigger_missing_count"] == 1


def test_compare_downstream_benchmarks_improved():
    baseline = {"ap50": 0.60, "ap75": 0.40, "map50_95": 0.35, "miss_rate": 0.25}
    lora = {"ap50": 0.65, "ap75": 0.43, "map50_95": 0.38, "miss_rate": 0.20}

    result = compare_downstream_benchmarks(baseline, lora)

    assert result["ap50"]["improved"] is True
    assert result["miss_rate"]["improved"] is True  # lower is better, delta < 0
    assert round(result["ap50"]["delta"], 6) == round(0.65 - 0.60, 6)
    assert result["ap50"]["direction"] == "higher_is_better"
    assert result["miss_rate"]["direction"] == "lower_is_better"


def test_compare_downstream_benchmarks_regressed():
    baseline = {"ap50": 0.70, "ap75": 0.50, "map50_95": 0.45, "miss_rate": 0.15}
    lora = {"ap50": 0.65, "ap75": 0.48, "map50_95": 0.42, "miss_rate": 0.20}

    result = compare_downstream_benchmarks(baseline, lora)

    assert result["ap50"]["improved"] is False
    assert result["miss_rate"]["improved"] is False  # higher miss_rate is worse


def test_compare_downstream_benchmarks_missing_value():
    baseline = {"ap50": 0.60, "ap75": None, "map50_95": 0.35, "miss_rate": 0.25}
    lora = {"ap50": 0.65, "ap75": 0.43, "map50_95": 0.38, "miss_rate": 0.20}

    result = compare_downstream_benchmarks(baseline, lora)

    assert result["ap75"]["delta"] is None
    assert result["ap75"]["improved"] is None
