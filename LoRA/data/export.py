"""Dataset export to ImageFolder JSONL format.

Exports curated samples to Hugging Face ImageFolder-compatible format
for LoRA training.
"""

from pathlib import Path
import json
import shutil
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime


class DatasetExporter:
    """Export dataset to release format."""

    def __init__(
        self,
        working_dir: Path,
        release_name: str,
    ):
        self.working_dir = Path(working_dir)
        self.release_name = release_name

        self.curated_dir = self.working_dir / "curated"
        self.releases_dir = self.working_dir / "releases"
        self.release_dir = self.releases_dir / release_name

    def export_release(
        self,
        samples_df: pd.DataFrame,
        images_df: pd.DataFrame,
        groups_df: pd.DataFrame,
        config_hash: str,
        git_commit: Optional[str] = None,
    ) -> Path:
        """Export full dataset release.

        Args:
            samples_df: Curated samples
            images_df: Image manifest
            groups_df: Groups manifest
            config_hash: Hash of configuration
            git_commit: Git commit hash

        Returns:
            Path to release directory
        """
        print("=" * 60)
        print("EXPORTING DATASET RELEASE")
        print("=" * 60)
        print(f"Release: {self.release_name}")
        print()

        self.release_dir.mkdir(parents=True, exist_ok=True)

        # Export LoRA training split
        self._export_lora_split(samples_df, split='train')

        # Export LoRA validation split
        self._export_lora_split(samples_df, split='val')

        # Export manifests
        self._export_manifests(samples_df, images_df, groups_df)

        # Create release metadata
        self._create_release_metadata(
            samples_df, config_hash, git_commit
        )

        # Create dataset card
        self._create_dataset_card(samples_df)

        print(f"\n✓ Release exported: {self.release_dir}")

        return self.release_dir

    def _export_lora_split(
        self,
        samples_df: pd.DataFrame,
        split: str,
    ):
        """Export LoRA training/validation split.

        Creates ImageFolder structure with metadata.jsonl.
        """
        split_samples = samples_df[
            (samples_df['split'] == split) &
            (samples_df['role'] == 'lora_positive')
        ]

        if len(split_samples) == 0:
            print(f"  Warning: No samples for LoRA {split}")
            return

        split_dir = self.release_dir / f"lora_{split}"
        images_dir = split_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nExporting LoRA {split} split...")
        print(f"  Samples: {len(split_samples)}")

        # Copy crops and build metadata
        metadata_records = []

        for idx, row in split_samples.iterrows():
            # Copy crop
            src_path = Path(row['crop_path'])
            if not src_path.exists():
                print(f"  Warning: Crop not found: {src_path}")
                continue

            dst_filename = f"{row['sample_id']}.jpg"
            dst_path = images_dir / dst_filename

            shutil.copy2(src_path, dst_path)

            # Add metadata record
            metadata_records.append({
                "file_name": f"images/{dst_filename}",
                "text": row['caption'],
            })

        # Write metadata.jsonl
        metadata_path = split_dir / "metadata.jsonl"
        with open(metadata_path, 'w') as f:
            for record in metadata_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"  ✓ Exported {len(metadata_records)} samples to {split_dir}")

    def _export_manifests(
        self,
        samples_df: pd.DataFrame,
        images_df: pd.DataFrame,
        groups_df: pd.DataFrame,
    ):
        """Export manifests for traceability."""
        print("\nExporting manifests...")

        samples_df.to_parquet(self.release_dir / "samples.parquet", index=False)
        images_df.to_parquet(self.release_dir / "images.parquet", index=False)
        groups_df.to_parquet(self.release_dir / "groups.parquet", index=False)

        print("  ✓ Manifests exported")

    def _create_release_metadata(
        self,
        samples_df: pd.DataFrame,
        config_hash: str,
        git_commit: Optional[str],
    ):
        """Create release.json metadata."""
        print("\nCreating release metadata...")

        # Compute manifest hash
        import hashlib
        manifest_hash = hashlib.sha256(
            samples_df.to_json().encode()
        ).hexdigest()[:16]

        split_counts = samples_df[samples_df['role'] == 'lora_positive']['split'].value_counts()

        metadata = {
            "release_name": self.release_name,
            "release_version": "1.0.0",
            "dataset_status": "exported",
            "created_at": datetime.now().isoformat(),
            "git_commit": git_commit,
            "config_hash": config_hash,
            "manifest_hash": manifest_hash,
            "stats": {
                "total_samples": len(samples_df),
                "lora_train": int(split_counts.get('train', 0)),
                "lora_val": int(split_counts.get('val', 0)),
                "total_images": samples_df['image_id'].nunique(),
            },
        }

        with open(self.release_dir / "release.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print("  ✓ release.json created")

    def _create_dataset_card(self, samples_df: pd.DataFrame):
        """Create dataset_card.json for documentation."""
        split_counts = samples_df['split'].value_counts()
        source_counts = samples_df['source_id'].value_counts()

        card = {
            "dataset_name": self.release_name,
            "description": "Domain-adapted pedestrian LoRA training dataset for SD3.5",
            "task": "text-to-image lora training",
            "trigger_token": samples_df['trigger_token'].iloc[0] if len(samples_df) > 0 else "<vin_ped>",
            "splits": {
                split: int(count) for split, count in split_counts.items()
            },
            "sources": {
                source: int(count) for source, count in source_counts.items()
            },
            "quality_thresholds": {
                "min_bbox_height_px": 96,
                "min_visible_ratio": 0.45,
                "min_quality_score": 0.70,
            },
        }

        with open(self.release_dir / "dataset_card.json", 'w') as f:
            json.dump(card, f, indent=2)

        print("  ✓ dataset_card.json created")


def export_imagefolder(
    samples_df: pd.DataFrame,
    output_dir: Path,
    split: str = "train",
) -> Path:
    """Export ImageFolder format for specific split.

    Args:
        samples_df: Sample manifest
        output_dir: Output directory
        split: Split to export

    Returns:
        Path to exported directory
    """
    split_dir = output_dir / split
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    split_samples = samples_df[samples_df['split'] == split]

    metadata = []
    for idx, row in split_samples.iterrows():
        src = Path(row['crop_path'])
        if not src.exists():
            continue

        dst = images_dir / f"{row['sample_id']}.jpg"
        shutil.copy2(src, dst)

        metadata.append({
            "file_name": f"images/{dst.name}",
            "text": row['caption'],
        })

    # Write metadata.jsonl
    with open(split_dir / "metadata.jsonl", 'w') as f:
        for record in metadata:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return split_dir
