"""Tests for split leakage prevention."""

import pytest
import pandas as pd
import numpy as np

from LoRA.data.splits import DatasetSplitter, SplitValidator, create_split_locks


@pytest.fixture
def mock_images_and_groups():
    """Create mock images and groups data."""
    # Create 100 images across 20 groups
    images = []
    groups = []

    for group_idx in range(20):
        for img_idx in range(5):
            image_id = f"img_{group_idx}_{img_idx}"
            source_id = f"source{group_idx % 3}"

            images.append({
                'image_id': image_id,
                'source_id': source_id,
                'width': 1024,
                'height': 512,
            })

            groups.append({
                'image_id': image_id,
                'duplicate_cluster_id': f"cluster_{group_idx}",
                'split_group_id': f"group_{group_idx}",
                'dedupe_status': 'unique',
                'canonical_image_id': image_id,
            })

    images_df = pd.DataFrame(images)
    groups_df = pd.DataFrame(groups)

    return images_df, groups_df


def test_dataset_splitter_initialization(tmp_path):
    """Test DatasetSplitter initialization."""
    splitter = DatasetSplitter(
        working_dir=tmp_path,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        split_seed=42,
    )

    assert splitter.train_ratio == 0.7
    assert splitter.val_ratio == 0.15
    assert splitter.test_ratio == 0.15
    assert splitter.split_seed == 42


def test_assign_splits_zero_leakage(tmp_path, mock_images_and_groups):
    """Test that split assignment has zero cross-split leakage."""
    images_df, groups_df = mock_images_and_groups

    splitter = DatasetSplitter(
        working_dir=tmp_path,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    splits_df = splitter.assign_splits(images_df, groups_df)

    # Verify all images have splits
    assert len(splits_df) == len(images_df)
    assert splits_df['assigned_split'].notna().all()

    # Verify splits exist
    assert 'train' in splits_df['assigned_split'].values
    assert 'val' in splits_df['assigned_split'].values
    assert 'test' in splits_df['assigned_split'].values

    # Verify no group spans multiple splits
    merged = splits_df.merge(groups_df, on='image_id')
    group_splits = merged.groupby('split_group_id')['assigned_split'].nunique()

    # All groups should have exactly 1 split
    assert (group_splits == 1).all()


def test_split_validator_no_errors(tmp_path, mock_images_and_groups):
    """Test split validation with valid data."""
    images_df, groups_df = mock_images_and_groups

    splitter = DatasetSplitter(working_dir=tmp_path)
    splits_df = splitter.assign_splits(images_df, groups_df)

    # Create mock samples
    samples = []
    for idx, row in splits_df.iterrows():
        samples.append({
            'sample_id': f"sample_{idx}",
            'image_id': row['image_id'],
            'role': 'lora_positive',
            'split': row['assigned_split'],
            'split_group_id': f"group_{idx // 5}",
        })

    samples_df = pd.DataFrame(samples)

    # Validate
    result = SplitValidator.validate_split_safety(
        images_df=images_df,
        groups_df=groups_df,
        samples_df=samples_df,
        benchmark_locked_sources=[],
    )

    assert result['valid'] is True
    assert len(result['errors']) == 0


def test_split_validator_detects_cross_split_leakage(tmp_path, mock_images_and_groups):
    """Test that validator detects cross-split leakage."""
    images_df, groups_df = mock_images_and_groups

    # Create samples with deliberate leakage
    samples = []
    for idx in range(10):
        samples.append({
            'sample_id': f"sample_{idx}",
            'image_id': f"img_0_{idx % 5}",
            'role': 'lora_positive',
            'split': 'train' if idx < 5 else 'val',  # Same group in different splits!
            'split_group_id': 'group_0',
        })

    samples_df = pd.DataFrame(samples)

    # Validate
    result = SplitValidator.validate_split_safety(
        images_df=images_df,
        groups_df=groups_df,
        samples_df=samples_df,
        benchmark_locked_sources=[],
    )

    assert result['valid'] is False
    assert len(result['errors']) > 0
    assert 'spanning multiple splits' in result['errors'][0]


def test_split_validator_detects_benchmark_leak(tmp_path, mock_images_and_groups):
    """Test that validator detects benchmark-locked images in training."""
    images_df, groups_df = mock_images_and_groups

    # Mark source0 as benchmark-locked
    benchmark_sources = ['source0']

    # Create samples including benchmark images in training
    samples = []
    for idx in range(20):
        img_id = f"img_{idx}_0"
        samples.append({
            'sample_id': f"sample_{idx}",
            'image_id': img_id,
            'role': 'lora_positive',
            'split': 'train',  # Benchmark images in training!
            'split_group_id': f"group_{idx}",
        })

    samples_df = pd.DataFrame(samples)

    # Validate
    result = SplitValidator.validate_split_safety(
        images_df=images_df,
        groups_df=groups_df,
        samples_df=samples_df,
        benchmark_locked_sources=benchmark_sources,
    )

    assert result['valid'] is False
    assert len(result['errors']) > 0
    assert 'benchmark-locked' in result['errors'][0]


def test_create_split_locks(mock_images_and_groups):
    """Test creating split locks for benchmark samples."""
    images_df, groups_df = mock_images_and_groups

    # Mark source0 as benchmark
    benchmark_sources = ['source0']

    # Create samples
    samples = []
    for idx in range(20):
        img_id = f"img_{idx}_0"
        source_id = f"source{idx % 3}"

        samples.append({
            'sample_id': f"sample_{idx}",
            'image_id': img_id,
            'role': 'lora_positive',
            'split': 'val' if idx < 10 else 'test',
            'split_group_id': f"group_{idx}",
        })

    samples_df = pd.DataFrame(samples)

    # Apply locks
    locked_df = create_split_locks(
        samples_df=samples_df,
        benchmark_locked_sources=benchmark_sources,
        images_df=images_df,
    )

    # Check that benchmark images have correct roles
    benchmark_images = images_df[images_df['source_id'].isin(benchmark_sources)]['image_id']
    benchmark_samples = locked_df[locked_df['image_id'].isin(benchmark_images)]

    # Should have detector roles, not lora roles
    assert 'detector_val_real_frozen' in benchmark_samples['role'].values
    assert 'detector_test_real_frozen' in benchmark_samples['role'].values
    assert 'lora_positive' not in benchmark_samples['role'].values
