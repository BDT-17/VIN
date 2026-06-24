"""Person crop extraction with context margins.

Extracts person crops from full images with contextual padding
for LoRA training.
"""

from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import pandas as pd


class CropBuilder:
    """Build person crops with context for LoRA training."""

    def __init__(
        self,
        context_ratio: float = 0.25,
        min_size: int = 128,
        max_size: int = 768,
    ):
        """Initialize crop builder.

        Args:
            context_ratio: Ratio of bbox dimension to add as context margin
            min_size: Minimum crop dimension
            max_size: Maximum crop dimension
        """
        self.context_ratio = context_ratio
        self.min_size = min_size
        self.max_size = max_size

    def build_crop(
        self,
        image_path: Path,
        bbox: Tuple[float, float, float, float],
        output_path: Path,
    ) -> Optional[Tuple[int, int]]:
        """Extract crop from image with context.

        Args:
            image_path: Path to source image
            bbox: Bounding box (x, y, w, h) in pixels
            output_path: Path to save crop

        Returns:
            (crop_width, crop_height) or None if failed
        """
        try:
            img = Image.open(image_path)
            img_w, img_h = img.size

            x, y, w, h = bbox

            # Add context margin
            margin_w = w * self.context_ratio
            margin_h = h * self.context_ratio

            crop_x1 = max(0, x - margin_w)
            crop_y1 = max(0, y - margin_h)
            crop_x2 = min(img_w, x + w + margin_w)
            crop_y2 = min(img_h, y + h + margin_h)

            # Ensure minimum size
            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1

            if crop_w < self.min_size or crop_h < self.min_size:
                return None

            # Extract crop
            crop = img.crop((int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)))

            # Resize if too large
            if crop_w > self.max_size or crop_h > self.max_size:
                crop.thumbnail((self.max_size, self.max_size), Image.Resampling.LANCZOS)

            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(output_path, quality=95)

            return crop.size

        except Exception as e:
            print(f"Warning: Failed to build crop from {image_path}: {e}")
            return None


def build_all_crops(
    images_df: pd.DataFrame,
    instances_df: pd.DataFrame,
    samples_df: pd.DataFrame,
    output_dir: Path,
    context_ratio: float = 0.25,
) -> pd.DataFrame:
    """Build crops for all samples.

    Args:
        images_df: Image manifest
        instances_df: Instance manifest
        samples_df: Sample manifest (will be updated with crop info)
        output_dir: Directory to save crops
        context_ratio: Context margin ratio

    Returns:
        Updated samples_df with crop_path and crop dimensions
    """
    print("=" * 60)
    print("BUILDING CROPS")
    print("=" * 60)

    builder = CropBuilder(context_ratio=context_ratio)

    # Merge to get full info
    merged = samples_df.merge(
        instances_df[['instance_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']],
        on='instance_id'
    ).merge(
        images_df[['image_id', 'raw_path']],
        on='image_id'
    )

    crops_built = 0
    crops_failed = 0

    for idx, row in merged.iterrows():
        # Build crop filename
        crop_filename = f"{row['sample_id']}.jpg"
        crop_path = output_dir / "crops" / crop_filename

        # Build crop
        bbox = (row['bbox_x'], row['bbox_y'], row['bbox_w'], row['bbox_h'])
        result = builder.build_crop(
            image_path=Path(row['raw_path']),
            bbox=bbox,
            output_path=crop_path,
        )

        if result:
            crop_w, crop_h = result
            samples_df.loc[samples_df['sample_id'] == row['sample_id'], 'crop_path'] = str(crop_path)
            samples_df.loc[samples_df['sample_id'] == row['sample_id'], 'crop_width'] = crop_w
            samples_df.loc[samples_df['sample_id'] == row['sample_id'], 'crop_height'] = crop_h
            crops_built += 1
        else:
            crops_failed += 1

        if (idx + 1) % 100 == 0:
            print(f"  Progress: {idx + 1}/{len(merged)} crops")

    print(f"\n✓ Crops built: {crops_built}")
    if crops_failed > 0:
        print(f"  Failed: {crops_failed}")

    return samples_df
