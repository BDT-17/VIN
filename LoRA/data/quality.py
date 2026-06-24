"""Quality filtering for LoRA training samples.

Applies quality thresholds to determine which samples are suitable
for LoRA training.
"""

from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np


class QualityFilter:
    """Filter samples by quality thresholds."""

    def __init__(
        self,
        min_bbox_height_px: int = 96,
        min_visible_ratio: float = 0.45,
        min_quality_score: float = 0.70,
        max_source_share: float = 0.50,
    ):
        """Initialize quality filter.

        Args:
            min_bbox_height_px: Minimum bbox height in pixels
            min_visible_ratio: Minimum visible portion
            min_quality_score: Minimum overall quality score
            max_source_share: Maximum fraction of samples from one source
        """
        self.min_bbox_height_px = min_bbox_height_px
        self.min_visible_ratio = min_visible_ratio
        self.min_quality_score = min_quality_score
        self.max_source_share = max_source_share

    def compute_quality_score(
        self,
        bbox_h: float,
        visible_ratio: float,
        occlusion_level: float,
    ) -> float:
        """Compute overall quality score.

        Args:
            bbox_h: Bbox height in pixels
            visible_ratio: Visible portion [0, 1]
            occlusion_level: Occlusion level [0, 1]

        Returns:
            Quality score [0, 1]
        """
        # Height score (0 at min threshold, 1 at 200px)
        height_score = np.clip((bbox_h - self.min_bbox_height_px) / 104, 0, 1)

        # Visibility score
        visibility_score = visible_ratio

        # Occlusion penalty
        occlusion_score = 1.0 - (occlusion_level or 0)

        # Weighted average
        quality = (
            0.4 * height_score +
            0.4 * visibility_score +
            0.2 * occlusion_score
        )

        return float(np.clip(quality, 0, 1))

    def filter_samples(
        self,
        instances_df: pd.DataFrame,
        images_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Filter instances by quality thresholds.

        Args:
            instances_df: Instance manifest
            images_df: Image manifest

        Returns:
            Filtered instances with quality_score column
        """
        print("=" * 60)
        print("QUALITY FILTERING")
        print("=" * 60)

        # Merge to get image dimensions
        merged = instances_df.merge(
            images_df[['image_id', 'width', 'height']],
            on='image_id'
        )

        print(f"\nInput instances: {len(merged)}")

        # Filter by bbox height
        passed_height = merged['bbox_h'] >= self.min_bbox_height_px
        print(f"  After height filter (>= {self.min_bbox_height_px}px): {passed_height.sum()}")

        # Compute visible ratio
        if 'visible_bbox_w' in merged.columns and 'visible_bbox_h' in merged.columns:
            visible_area = merged['visible_bbox_w'] * merged['visible_bbox_h']
            total_area = merged['bbox_w'] * merged['bbox_h']
            visible_ratio = (visible_area / total_area.replace(0, 1)).fillna(1.0)
        else:
            visible_ratio = pd.Series(1.0, index=merged.index)

        merged['visible_ratio'] = visible_ratio

        # Filter by visible ratio
        passed_visible = visible_ratio >= self.min_visible_ratio
        print(f"  After visible ratio filter (>= {self.min_visible_ratio}): {passed_visible.sum()}")

        # Compute quality score
        occlusion_level = merged['occlusion_level'].fillna(0)
        quality_scores = merged.apply(
            lambda row: self.compute_quality_score(
                row['bbox_h'],
                row['visible_ratio'],
                row['occlusion_level'] or 0,
            ),
            axis=1
        )

        merged['quality_score'] = quality_scores

        # Filter by quality score
        passed_quality = quality_scores >= self.min_quality_score
        print(f"  After quality score filter (>= {self.min_quality_score}): {passed_quality.sum()}")

        # Remove ignore flags
        passed_not_ignore = ~merged['ignore_flag']
        print(f"  After removing ignore flags: {passed_not_ignore.sum()}")

        # Apply all filters
        final_mask = passed_height & passed_visible & passed_quality & passed_not_ignore
        filtered = merged[final_mask].copy()

        print(f"\n✓ Final filtered instances: {len(filtered)}")

        # Add quality metrics
        filtered['bbox_height_ratio'] = filtered['bbox_h'] / filtered['height']

        return filtered

    def balance_sources(
        self,
        filtered_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Balance source representation.

        Args:
            filtered_df: Filtered instances

        Returns:
            Balanced instances
        """
        print("\n" + "=" * 60)
        print("SOURCE BALANCING")
        print("=" * 60)

        source_counts = filtered_df['source_id'].value_counts()
        print("\nSource distribution before balancing:")
        for source, count in source_counts.items():
            pct = (count / len(filtered_df)) * 100
            print(f"  {source}: {count} ({pct:.1f}%)")

        # Compute max samples per source
        max_samples = int(len(filtered_df) * self.max_source_share)

        # Sample from over-represented sources
        balanced_dfs = []
        for source_id, group in filtered_df.groupby('source_id'):
            if len(group) > max_samples:
                # Sample by quality score (keep highest quality)
                sampled = group.nlargest(max_samples, 'quality_score')
                balanced_dfs.append(sampled)
                print(f"  {source_id}: sampled {len(sampled)} from {len(group)}")
            else:
                balanced_dfs.append(group)

        balanced = pd.concat(balanced_dfs, ignore_index=True)

        print(f"\n✓ Balanced instances: {len(balanced)}")

        return balanced
