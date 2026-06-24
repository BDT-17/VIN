"""Dataset normalization.

Combines raw inventories with deduplication results into
normalized manifests.
"""

from pathlib import Path
import pandas as pd
from typing import Dict


class DatasetNormalizer:
    """Normalize raw inventories into canonical manifests."""

    def __init__(self, working_dir: Path):
        self.working_dir = Path(working_dir)
        self.inventory_dir = self.working_dir / "raw_inventory"
        self.normalized_dir = self.working_dir / "normalized"
        self.normalized_dir.mkdir(parents=True, exist_ok=True)

    def normalize_images(self, groups_df: pd.DataFrame) -> Path:
        """Create normalized images.parquet.

        Combines all source inventories and adds canonical status.
        """
        print("=" * 60)
        print("NORMALIZATION: IMAGES")
        print("=" * 60)

        # Load all image inventories
        dfs = []
        for parquet_file in self.inventory_dir.glob("*_images.parquet"):
            df = pd.read_parquet(parquet_file)
            dfs.append(df)

        all_images = pd.concat(dfs, ignore_index=True)

        print(f"\nTotal images: {len(all_images)}")

        # Merge with groups to add canonical status
        normalized = all_images.merge(
            groups_df[['image_id', 'canonical_image_id', 'dedupe_status']],
            on='image_id',
            how='left'
        )

        # Save normalized images
        output_path = self.normalized_dir / "images.parquet"
        normalized.to_parquet(output_path, index=False)

        print(f"✓ Normalized images saved: {output_path}")

        return output_path

    def normalize_instances(self) -> Path:
        """Create normalized instances.parquet.

        Combines all source instance inventories.
        """
        print("\n" + "=" * 60)
        print("NORMALIZATION: INSTANCES")
        print("=" * 60)

        # Load all instance inventories
        dfs = []
        for parquet_file in self.inventory_dir.glob("*_instances.parquet"):
            df = pd.read_parquet(parquet_file)
            if len(df) > 0:
                dfs.append(df)

        if not dfs:
            print("No instances found")
            return None

        all_instances = pd.concat(dfs, ignore_index=True)

        print(f"\nTotal instances: {len(all_instances)}")
        print(f"  Pedestrian: {(all_instances['class_name'] == 'pedestrian').sum()}")
        print(f"  Ignore: {(all_instances['ignore_flag'] == True).sum()}")

        # Save normalized instances
        output_path = self.normalized_dir / "instances.parquet"
        all_instances.to_parquet(output_path, index=False)

        print(f"✓ Normalized instances saved: {output_path}")

        return output_path
