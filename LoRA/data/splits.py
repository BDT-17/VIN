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

        # Check 2: No scene (split_group_id) shared between LoRA roles and the
        # frozen benchmark roles. This is the real leakage invariant: a
        # benchmark source MAY contribute LoRA training data, as long as the
        # frozen val/test scenes are group-disjoint from the LoRA scenes.
        lora_roles = {'lora_positive', 'lora_val'}
        benchmark_roles = {'detector_val_real_frozen', 'detector_test_real_frozen'}
        lora_groups = set(
            samples_df[samples_df['role'].isin(lora_roles)]['split_group_id'].dropna()
        )
        bench_groups = set(
            samples_df[samples_df['role'].isin(benchmark_roles)]['split_group_id'].dropna()
        )
        group_leak = lora_groups & bench_groups

        if len(group_leak) > 0:
            errors.append(
                f"Found {len(group_leak)} split_group_ids shared between LoRA and frozen benchmark"
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
            'benchmark_overlap_count': len(group_leak),
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

    Policy (a benchmark source contributes BOTH frozen benchmark and LoRA data,
    kept scene-disjoint via the group-aware DatasetSplitter):

      - source 'valid'/'val' folder           -> detector_val_real_frozen (frozen val)
      - source 'train' folder, group assigned
        'test' by the splitter (~15%)          -> detector_test_real_frozen (frozen test)
      - source 'train' folder, group assigned
        'train'/'val' (~85%)                    -> kept as lora_positive (LoRA training)

    Scene-disjointness is guaranteed because the DatasetSplitter never lets a
    split_group_id span splits, and group IDs embed original_split (so train- and
    valid-folder groups never collide).

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
    orig_split = samples_df['image_id'].map(original_split_map)  # NaN for non-benchmark

    # --- Frozen VAL: everything from the source 'valid' folder ---
    is_orig_val = is_benchmark & (orig_split == 'val')
    samples_df.loc[is_orig_val, 'split'] = 'val'
    samples_df.loc[is_orig_val, 'role'] = 'detector_val_real_frozen'

    # --- Frozen TEST: the group-disjoint 'test' slice of the source 'train'
    #     folder (assigned by the group-aware splitter, ~test_ratio of train) ---
    is_orig_train = is_benchmark & (orig_split == 'train')
    is_train_to_test = is_orig_train & (samples_df['split'] == 'test')
    samples_df.loc[is_train_to_test, 'role'] = 'detector_test_real_frozen'

    # --- LoRA: source 'train' folder groups assigned train/val keep
    #     role 'lora_positive' (the default) and their split. Nothing to do. ---

    locked_val = int((samples_df['role'] == 'detector_val_real_frozen').sum())
    locked_test = int((samples_df['role'] == 'detector_test_real_frozen').sum())
    lora_from_bench = int((is_orig_train & samples_df['split'].isin(['train', 'val'])).sum())
    print(f"  Benchmark: {locked_val} val_frozen (source valid), "
          f"{locked_test} test_frozen (group-disjoint slice of source train); "
          f"{lora_from_bench} source-train samples kept for LoRA")

    return samples_df
