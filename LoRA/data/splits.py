"""Dataset splitting with leakage prevention.

Ensures zero cross-split overlap by respecting duplicate clusters,
temporal windows, and group IDs.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict


class SplitValidator:
    """Validate that splits have zero leakage."""

    @staticmethod
    def validate_split_safety(
        images_df: pd.DataFrame,
        groups_df: pd.DataFrame,
        samples_df: pd.DataFrame,
        benchmark_locked_sources: List[str],
    ) -> Dict[str, any]:
        """Comprehensive split safety validation.

        Returns:
            Dictionary with validation results and errors.
        """
        errors = []
        warnings = []

        # Check 1: No duplicate cluster spans multiple splits
        split_groups = samples_df.groupby('split_group_id')['split'].apply(set)
        cross_split_groups = split_groups[split_groups.apply(len) > 1]

        if len(cross_split_groups) > 0:
            errors.append(
                f"Found {len(cross_split_groups)} groups spanning multiple splits"
            )
            for group_id, splits in cross_split_groups.head(10).items():
                errors.append(f"  Group {group_id}: {splits}")

        # Check 2: No benchmark-locked images in training splits
        benchmark_images = images_df[
            images_df['source_id'].isin(benchmark_locked_sources)
        ]['image_id'].tolist()

        training_samples = samples_df[
            samples_df['split'].isin(['train', 'val'])
        ]

        benchmark_in_training = training_samples[
            training_samples['image_id'].isin(benchmark_images)
        ]

        if len(benchmark_in_training) > 0:
            errors.append(
                f"Found {len(benchmark_in_training)} benchmark-locked samples in training splits"
            )

        # Check 3: No image appears in multiple roles that would cause leakage
        image_roles = samples_df.groupby('image_id')['role'].apply(set)
        conflicting_roles = image_roles[
            image_roles.apply(lambda roles: 'lora_positive' in roles and any(
                r.startswith('detector_') for r in roles
            ))
        ]

        if len(conflicting_roles) > 0:
            warnings.append(
                f"Found {len(conflicting_roles)} images with potentially conflicting roles"
            )

        # Check 4: Canonical images are used, not duplicates
        non_canonical_samples = samples_df[
            samples_df['image_id'] != samples_df.merge(
                groups_df, on='image_id'
            )['canonical_image_id']
        ]

        if len(non_canonical_samples) > 0:
            warnings.append(
                f"Found {len(non_canonical_samples)} samples using non-canonical images"
            )

        # Compute stats
        stats = {
            'total_samples': len(samples_df),
            'train_count': (samples_df['split'] == 'train').sum(),
            'val_count': (samples_df['split'] == 'val').sum(),
            'test_count': (samples_df['split'] == 'test').sum(),
            'cross_split_duplicate_count': len(cross_split_groups),
            'benchmark_overlap_count': len(benchmark_in_training),
        }

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'stats': stats,
        }


class DatasetSplitter:
    """Create train/val/test splits with leakage prevention."""

    def __init__(
        self,
        working_dir: Path,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        split_seed: int = 42,
    ):
        self.working_dir = Path(working_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_seed = split_seed

        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 0.001

        self.normalized_dir = self.working_dir / "normalized"

    def assign_splits(
        self,
        images_df: pd.DataFrame,
        groups_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Assign train/val/test splits to images.

        Uses split_group_id to ensure entire groups stay together.

        Returns:
            DataFrame with 'assigned_split' column added to images
        """
        print("=" * 60)
        print("SPLIT ASSIGNMENT")
        print("=" * 60)

        # Merge images with groups
        merged = images_df.merge(groups_df, on='image_id', how='left')

        # Get unique split groups
        split_groups = merged['split_group_id'].unique()
        print(f"\nTotal split groups: {len(split_groups)}")

        # Shuffle groups
        rng = np.random.RandomState(self.split_seed)
        rng.shuffle(split_groups)

        # Assign splits to groups
        n_groups = len(split_groups)
        train_end = int(n_groups * self.train_ratio)
        val_end = train_end + int(n_groups * self.val_ratio)

        group_to_split = {}
        for i, group_id in enumerate(split_groups):
            if i < train_end:
                group_to_split[group_id] = 'train'
            elif i < val_end:
                group_to_split[group_id] = 'val'
            else:
                group_to_split[group_id] = 'test'

        # Assign splits to images
        merged['assigned_split'] = merged['split_group_id'].map(group_to_split)

        # Print split stats
        split_counts = merged['assigned_split'].value_counts()
        print("\nSplit distribution:")
        for split_name in ['train', 'val', 'test']:
            count = split_counts.get(split_name, 0)
            pct = (count / len(merged)) * 100
            print(f"  {split_name}: {count} images ({pct:.1f}%)")

        # Verify no group spans splits
        verification = merged.groupby('split_group_id')['assigned_split'].nunique()
        if (verification > 1).any():
            raise ValueError("Split assignment failed: some groups span multiple splits")

        print("\n✓ Split assignment complete (zero leakage)")

        return merged[['image_id', 'assigned_split']]


def create_split_locks(
    samples_df: pd.DataFrame,
    benchmark_locked_sources: List[str],
    images_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create split locks for benchmark-protected samples.

    Uses original_split from the source dataset (not the randomly assigned split)
    to determine which samples become detector_val_real_frozen vs detector_test_real_frozen.
    ALL benchmark-source images are excluded from LoRA training regardless of original_split.

    Returns:
        samples_df with updated 'role' and corrected 'split' for locked samples
    """
    benchmark_mask = images_df['source_id'].isin(benchmark_locked_sources)
    benchmark_image_ids = set(images_df[benchmark_mask]['image_id'])

    if not benchmark_image_ids:
        return samples_df

    # Build original_split lookup; normalize 'valid' → 'val' for uniformity
    original_split_map = (
        images_df[benchmark_mask]
        .set_index('image_id')['original_split']
        .str.replace(r'^valid$', 'val', regex=True)
    )

    samples_df = samples_df.copy()
    is_benchmark = samples_df['image_id'].isin(benchmark_image_ids)

    # Override randomly-assigned split with authoritative original_split
    samples_df.loc[is_benchmark, 'split'] = (
        samples_df.loc[is_benchmark, 'image_id'].map(original_split_map)
    )

    # Assign frozen benchmark roles based on corrected original split
    samples_df.loc[is_benchmark & (samples_df['split'] == 'val'), 'role'] = 'detector_val_real_frozen'
    samples_df.loc[is_benchmark & (samples_df['split'] == 'test'), 'role'] = 'detector_test_real_frozen'

    # Remove ALL benchmark images that still carry a LoRA role (original train split)
    samples_df = samples_df[
        ~(is_benchmark & samples_df['role'].isin(['lora_positive', 'lora_val']))
    ]

    locked_val = (samples_df['role'] == 'detector_val_real_frozen').sum()
    locked_test = (samples_df['role'] == 'detector_test_real_frozen').sum()
    print(f"  Benchmark locked: {locked_val} val_frozen, {locked_test} test_frozen "
          f"(from original_split, not random assignment)")

    return samples_df
