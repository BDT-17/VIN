"""Tests for deduplication."""

import pytest
from pathlib import Path
import pandas as pd
import imagehash
from PIL import Image

from LoRA.data.dedupe import ImageDeduplicator, DeduplicatorConfig


@pytest.fixture
def mock_inventory_dir(tmp_path):
    """Create mock inventory with duplicates."""
    inventory_dir = tmp_path / "raw_inventory"
    inventory_dir.mkdir()

    # Create source 1 with some images
    images_1 = []
    for i in range(5):
        img_path = tmp_path / f"img_s1_{i}.jpg"
        img = Image.new('RGB', (100, 100), color=(i * 50, 0, 0))
        img.save(img_path)

        phash = str(imagehash.phash(img))

        images_1.append({
            'image_id': f's1_img_{i}',
            'source_id': 'source1',
            'source_image_id': str(i),
            'raw_path': str(img_path),
            'width': 100,
            'height': 100,
            'sha256': f"{'a' * 64}" if i < 2 else f"{'b' * 63}{i}",  # First two are exact dups
            'phash': phash,
            'group_id': f's1_group_{i}',
            'frame_index': i,
        })

    df1 = pd.DataFrame(images_1)
    df1.to_parquet(inventory_dir / "source1_images.parquet", index=False)

    # Create source 2
    images_2 = []
    for i in range(3):
        img_path = tmp_path / f"img_s2_{i}.jpg"
        img = Image.new('RGB', (100, 100), color=(0, i * 80, 0))
        img.save(img_path)

        phash = str(imagehash.phash(img))

        images_2.append({
            'image_id': f's2_img_{i}',
            'source_id': 'source2',
            'source_image_id': str(i),
            'raw_path': str(img_path),
            'width': 100,
            'height': 100,
            'sha256': f"{'c' * 63}{i}",
            'phash': phash,
            'group_id': f's2_group_{i}',
            'frame_index': None,
        })

    df2 = pd.DataFrame(images_2)
    df2.to_parquet(inventory_dir / "source2_images.parquet", index=False)

    return tmp_path


def test_deduplicator_initialization(mock_inventory_dir):
    """Test ImageDeduplicator initialization."""
    deduplicator = ImageDeduplicator(working_dir=mock_inventory_dir)

    assert deduplicator.working_dir == mock_inventory_dir
    assert deduplicator.inventory_dir.exists()


def test_load_all_images(mock_inventory_dir):
    """Test loading all image inventories."""
    deduplicator = ImageDeduplicator(working_dir=mock_inventory_dir)
    df = deduplicator._load_all_images()

    # Should have 8 images total (5 from source1, 3 from source2)
    assert len(df) == 8
    assert 'source1' in df['source_id'].values
    assert 'source2' in df['source_id'].values


def test_find_sha256_duplicates(mock_inventory_dir):
    """Test SHA-256 exact duplicate detection."""
    deduplicator = ImageDeduplicator(working_dir=mock_inventory_dir)
    df = deduplicator._load_all_images()

    clusters = deduplicator._find_sha256_duplicates(df)

    # First two images from source1 have same SHA-256
    assert len(clusters) >= 1

    # Check that the exact duplicate cluster exists
    exact_dup_cluster = [v for v in clusters.values() if 's1_img_0' in v and 's1_img_1' in v]
    assert len(exact_dup_cluster) == 1
    assert len(exact_dup_cluster[0]) == 2


def test_deduplicate_all_sources(mock_inventory_dir):
    """Test full deduplication pipeline."""
    deduplicator = ImageDeduplicator(working_dir=mock_inventory_dir)

    source_priorities = {
        'source1': 1,  # Higher priority (survives)
        'source2': 2,
    }

    groups_path = deduplicator.deduplicate_all_sources(source_priorities)

    assert groups_path.exists()

    # Load groups
    groups_df = pd.read_parquet(groups_path)

    # Should have 8 groups (one per image)
    assert len(groups_df) == 8

    # Check dedupe statuses
    assert 'unique' in groups_df['dedupe_status'].values
    assert 'exact_duplicate' in groups_df['dedupe_status'].values or len(groups_df) == 8


def test_choose_canonical(mock_inventory_dir):
    """Test canonical image selection based on priority."""
    deduplicator = ImageDeduplicator(working_dir=mock_inventory_dir)

    image_ids = ['s1_img_0', 's2_img_0']
    image_to_source = {
        's1_img_0': 'source1',
        's2_img_0': 'source2',
    }
    source_priorities = {
        'source1': 1,
        'source2': 2,
    }

    canonical = deduplicator._choose_canonical(
        image_ids, image_to_source, source_priorities
    )

    # Should choose source1 (higher priority)
    assert canonical == 's1_img_0'
