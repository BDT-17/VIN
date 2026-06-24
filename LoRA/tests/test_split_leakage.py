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
            # Groups 0-13 come from the source 'train' folder, 14-19 from 'valid'.
            original_split = 'train' if group_idx < 14 else 'valid'

            images.append({
                'image_id': image_id,
                'source_id': source_id,
                'width': 1024,
                'height': 512,
                'original_split': original_split,
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


def test_split_validator_detects_group_leak(tmp_path, mock_images_and_groups):
    """Validator flags a split_group_id shared by LoRA and frozen benchmark roles."""
    images_df, groups_df = mock_images_and_groups

    # group_0 appears in BOTH a LoRA role and a frozen benchmark role -> leakage
    samples = [
        {'sample_id': 's_lora', 'image_id': 'img_0_0', 'role': 'lora_positive',
         'split': 'train', 'split_group_id': 'group_0'},
        {'sample_id': 's_bench', 'image_id': 'img_0_1', 'role': 'detector_test_real_frozen',
         'split': 'test', 'split_group_id': 'group_0'},
    ]
    samples_df = pd.DataFrame(samples)

    result = SplitValidator.validate_split_safety(
        images_df=images_df,
        groups_df=groups_df,
        samples_df=samples_df,
        benchmark_locked_sources=['source0'],
    )

    assert result['valid'] is False
    assert any('shared between LoRA and frozen benchmark' in e for e in result['errors'])


def test_split_validator_allows_benchmark_train_for_lora(tmp_path, mock_images_and_groups):
    """A benchmark source MAY supply LoRA training data when scenes are disjoint."""
    images_df, groups_df = mock_images_and_groups

    samples = [
        # benchmark source0, distinct groups -> no shared scene -> no leak
        {'sample_id': 's_lora', 'image_id': 'img_0_0', 'role': 'lora_positive',
         'split': 'train', 'split_group_id': 'group_0'},
        {'sample_id': 's_bench', 'image_id': 'img_3_0', 'role': 'detector_test_real_frozen',
         'split': 'test', 'split_group_id': 'group_3'},
    ]
    samples_df = pd.DataFrame(samples)

    result = SplitValidator.validate_split_safety(
        images_df=images_df,
        groups_df=groups_df,
        samples_df=samples_df,
        benchmark_locked_sources=['source0'],
    )

    assert result['valid'] is True


def test_create_split_locks_carves_and_keeps_lora(mock_images_and_groups):
    """New policy: valid->frozen val, train 'test' slice->frozen test,
    train 'train'/'val' slice stays lora_positive (used for LoRA)."""
    images_df, groups_df = mock_images_and_groups
    benchmark_sources = ['source0']

    # source0 groups: train-origin 0,3,6,9,12 ; valid-origin 15,18
    rows = [
        ('img_0_0',  'test',  'group_0'),   # train-origin, splitter put in test -> frozen test
        ('img_3_0',  'train', 'group_3'),   # train-origin -> stays lora_positive
        ('img_6_0',  'val',   'group_6'),   # train-origin -> stays lora_positive
        ('img_15_0', 'train', 'group_15'),  # valid-origin -> frozen val (split forced to val)
        ('img_18_0', 'test',  'group_18'),  # valid-origin -> frozen val (split forced to val)
    ]
    samples_df = pd.DataFrame([
        {'sample_id': f"s_{i}", 'image_id': iid, 'role': 'lora_positive',
         'split': sp, 'split_group_id': g}
        for i, (iid, sp, g) in enumerate(rows)
    ])

    locked = create_split_locks(samples_df, benchmark_sources, images_df)
    role = lambda iid: locked.loc[locked['image_id'] == iid, 'role'].iloc[0]
    split = lambda iid: locked.loc[locked['image_id'] == iid, 'split'].iloc[0]

    # valid-origin -> frozen val, split forced to 'val'
    assert role('img_15_0') == 'detector_val_real_frozen' and split('img_15_0') == 'val'
    assert role('img_18_0') == 'detector_val_real_frozen' and split('img_18_0') == 'val'
    # train-origin in 'test' -> frozen test
    assert role('img_0_0') == 'detector_test_real_frozen' and split('img_0_0') == 'test'
    # train-origin in train/val -> kept for LoRA
    assert role('img_3_0') == 'lora_positive'
    assert role('img_6_0') == 'lora_positive'

    # scene-disjoint: no group shared between lora and benchmark
    lora_g = set(locked[locked['role'] == 'lora_positive']['split_group_id'])
    bench_g = set(locked[locked['role'].str.startswith('detector_')]['split_group_id'])
    assert lora_g.isdisjoint(bench_g)
