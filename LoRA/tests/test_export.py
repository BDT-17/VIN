"""Tests for dataset export."""

import pytest
import json
from pathlib import Path
import pandas as pd
from PIL import Image

from LoRA.data.export import DatasetExporter


@pytest.fixture
def tmp_working_dir(tmp_path):
    """Create temporary working directory with crops."""
    working_dir = tmp_path / "dataset"
    working_dir.mkdir()

    curated_dir = working_dir / "curated" / "crops"
    curated_dir.mkdir(parents=True)

    # Create dummy crops
    for i in range(5):
        crop_path = curated_dir / f"sample_{i}.jpg"
        img = Image.new('RGB', (256, 256), color=(i * 50, 100, 150))
        img.save(crop_path)

    return working_dir


@pytest.fixture
def sample_data(tmp_working_dir):
    """Create sample manifests."""
    curated_dir = tmp_working_dir / "curated" / "crops"

    samples = []
    for i in range(5):
        samples.append({
            "sample_id": f"sample_{i}",
            "image_id": f"img_{i}",
            "instance_id": f"inst_{i}",
            "role": "lora_positive",
            "split": "train" if i < 3 else "val",
            "crop_path": str(curated_dir / f"sample_{i}.jpg"),
            "caption": f"photo of <vin_ped> a pedestrian {i}",
            "trigger_token": "<vin_ped>",
            "source_id": "test_source",
            "quality_score": 0.8,
        })

    samples_df = pd.DataFrame(samples)

    images_df = pd.DataFrame([
        {"image_id": f"img_{i}", "source_id": "test_source", "camera_domain": "cctv"}
        for i in range(5)
    ])

    groups_df = pd.DataFrame([
        {"split_group_id": f"group_{i}", "dedupe_status": "unique"}
        for i in range(5)
    ])

    return samples_df, images_df, groups_df


def test_export_release_creates_structure(tmp_working_dir, sample_data):
    """Test that export creates expected directory structure."""
    samples_df, images_df, groups_df = sample_data

    exporter = DatasetExporter(
        working_dir=tmp_working_dir,
        release_name="test_release_v1",
    )

    release_dir = exporter.export_release(
        samples_df=samples_df,
        images_df=images_df,
        groups_df=groups_df,
        config_hash="abc123",
        git_commit="def456",
    )

    assert release_dir.exists()
    assert (release_dir / "lora_train").exists()
    assert (release_dir / "lora_val").exists()
    assert (release_dir / "lora_train" / "images").exists()
    assert (release_dir / "lora_val" / "images").exists()


def test_export_creates_metadata_jsonl(tmp_working_dir, sample_data):
    """Test that metadata.jsonl is created correctly."""
    samples_df, images_df, groups_df = sample_data

    exporter = DatasetExporter(
        working_dir=tmp_working_dir,
        release_name="test_release_v1",
    )

    release_dir = exporter.export_release(
        samples_df=samples_df,
        images_df=images_df,
        groups_df=groups_df,
        config_hash="abc123",
    )

    # Check train metadata
    train_metadata = release_dir / "lora_train" / "metadata.jsonl"
    assert train_metadata.exists()

    with open(train_metadata, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 3  # 3 train samples

        first = json.loads(lines[0])
        assert "file_name" in first
        assert "text" in first
        assert "<vin_ped>" in first["text"]


def test_export_creates_release_json(tmp_working_dir, sample_data):
    """Test that release.json is created with correct status."""
    samples_df, images_df, groups_df = sample_data

    exporter = DatasetExporter(
        working_dir=tmp_working_dir,
        release_name="test_release_v1",
    )

    release_dir = exporter.export_release(
        samples_df=samples_df,
        images_df=images_df,
        groups_df=groups_df,
        config_hash="abc123",
        git_commit="def456",
    )

    release_json = release_dir / "release.json"
    assert release_json.exists()

    with open(release_json, 'r') as f:
        metadata = json.load(f)
        assert metadata["release_name"] == "test_release_v1"
        assert metadata["dataset_status"] == "exported"
        assert metadata["git_commit"] == "def456"
        assert metadata["config_hash"] == "abc123"
        assert metadata["stats"]["lora_train"] == 3
        assert metadata["stats"]["lora_val"] == 2


def test_export_creates_manifests(tmp_working_dir, sample_data):
    """Test that manifest parquet files are created."""
    samples_df, images_df, groups_df = sample_data

    exporter = DatasetExporter(
        working_dir=tmp_working_dir,
        release_name="test_release_v1",
    )

    release_dir = exporter.export_release(
        samples_df=samples_df,
        images_df=images_df,
        groups_df=groups_df,
        config_hash="abc123",
    )

    assert (release_dir / "samples.parquet").exists()
    assert (release_dir / "images.parquet").exists()
    assert (release_dir / "groups.parquet").exists()

    # Verify round-trip
    loaded_samples = pd.read_parquet(release_dir / "samples.parquet")
    assert len(loaded_samples) == 5


def test_export_copies_crops(tmp_working_dir, sample_data):
    """Test that crop images are copied correctly."""
    samples_df, images_df, groups_df = sample_data

    exporter = DatasetExporter(
        working_dir=tmp_working_dir,
        release_name="test_release_v1",
    )

    release_dir = exporter.export_release(
        samples_df=samples_df,
        images_df=images_df,
        groups_df=groups_df,
        config_hash="abc123",
    )

    # Check train crops
    train_images = list((release_dir / "lora_train" / "images").glob("*.jpg"))
    assert len(train_images) == 3

    # Check val crops
    val_images = list((release_dir / "lora_val" / "images").glob("*.jpg"))
    assert len(val_images) == 2
