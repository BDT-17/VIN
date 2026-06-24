"""Tests for schema validation."""

import pytest
from pathlib import Path

from LoRA.data.schema import (
    ImageRecord,
    InstanceRecord,
    SampleRecord,
    DataRole,
    Split,
)


def test_image_record_validation():
    """Test ImageRecord validation."""
    # Valid record
    record = ImageRecord(
        image_id="test_001",
        source_id="test_source",
        source_image_id="001",
        raw_path="/path/to/image.jpg",
        width=1920,
        height=1080,
        sha256="a" * 64,
        phash="0123456789abcdef",
        group_id="group_1",
    )
    errors = record.validate()
    assert len(errors) == 0

    # Invalid width
    record_bad = ImageRecord(
        image_id="test_001",
        source_id="test_source",
        source_image_id="001",
        raw_path="/path/to/image.jpg",
        width=-100,
        height=1080,
        sha256="a" * 64,
        phash="0123456789abcdef",
        group_id="group_1",
    )
    errors = record_bad.validate()
    assert any("width" in e for e in errors)

    # Invalid sha256
    record_bad_hash = ImageRecord(
        image_id="test_001",
        source_id="test_source",
        source_image_id="001",
        raw_path="/path/to/image.jpg",
        width=1920,
        height=1080,
        sha256="tooshort",
        phash="0123456789abcdef",
        group_id="group_1",
    )
    errors = record_bad_hash.validate()
    assert any("sha256" in e for e in errors)


def test_instance_record_validation():
    """Test InstanceRecord validation."""
    # Valid instance
    record = InstanceRecord(
        instance_id="inst_001",
        image_id="img_001",
        class_name="pedestrian",
        bbox_x=100,
        bbox_y=200,
        bbox_w=50,
        bbox_h=150,
    )
    errors = record.validate(image_width=1920, image_height=1080)
    assert len(errors) == 0

    # Bbox outside image
    record_bad = InstanceRecord(
        instance_id="inst_001",
        image_id="img_001",
        class_name="pedestrian",
        bbox_x=2000,  # Outside image
        bbox_y=200,
        bbox_w=50,
        bbox_h=150,
    )
    errors = record_bad.validate(image_width=1920, image_height=1080)
    assert len(errors) > 0

    # Invalid class
    record_bad_class = InstanceRecord(
        instance_id="inst_001",
        image_id="img_001",
        class_name="invalid_class",
        bbox_x=100,
        bbox_y=200,
        bbox_w=50,
        bbox_h=150,
    )
    errors = record_bad_class.validate(image_width=1920, image_height=1080)
    assert any("class_name" in e for e in errors)


def test_sample_record_validation(tmp_path):
    """Test SampleRecord validation."""
    # Create a dummy crop file
    crop_path = tmp_path / "crop.jpg"
    crop_path.write_text("dummy")

    # Valid sample
    record = SampleRecord(
        sample_id="sample_001",
        image_id="img_001",
        instance_id="inst_001",
        role=DataRole.LORA_POSITIVE.value,
        split=Split.TRAIN.value,
        crop_path=str(crop_path),
        crop_width=256,
        crop_height=256,
        bbox_height_ratio=0.15,
        visible_ratio=0.8,
        occlusion_level=0.2,
        source_id="test_source",
        quality_score=0.85,
        caption="photo of <vin_ped> a pedestrian, full body, walking",
        trigger_token="<vin_ped>",
        duplicate_cluster_id="cluster_1",
        split_group_id="group_1",
    )
    errors = record.validate()
    assert len(errors) == 0

    # Missing trigger token in caption
    record_bad = SampleRecord(
        sample_id="sample_001",
        image_id="img_001",
        instance_id="inst_001",
        role=DataRole.LORA_POSITIVE.value,
        split=Split.TRAIN.value,
        crop_path=str(crop_path),
        crop_width=256,
        crop_height=256,
        bbox_height_ratio=0.15,
        visible_ratio=0.8,
        occlusion_level=0.2,
        source_id="test_source",
        quality_score=0.85,
        caption="photo of a pedestrian, full body, walking",
        trigger_token="<vin_ped>",
        duplicate_cluster_id="cluster_1",
        split_group_id="group_1",
    )
    errors = record_bad.validate()
    assert any("trigger token" in e for e in errors)

    # Invalid quality score
    record_bad_quality = SampleRecord(
        sample_id="sample_001",
        image_id="img_001",
        instance_id="inst_001",
        role=DataRole.LORA_POSITIVE.value,
        split=Split.TRAIN.value,
        crop_path=str(crop_path),
        crop_width=256,
        crop_height=256,
        bbox_height_ratio=0.15,
        visible_ratio=0.8,
        occlusion_level=0.2,
        source_id="test_source",
        quality_score=1.5,  # Out of range
        caption="photo of <vin_ped> a pedestrian",
        trigger_token="<vin_ped>",
        duplicate_cluster_id="cluster_1",
        split_group_id="group_1",
    )
    errors = record_bad_quality.validate()
    assert any("quality_score" in e for e in errors)
