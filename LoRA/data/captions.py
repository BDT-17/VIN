"""Caption generation for LoRA training.

Generates descriptive captions with trigger token for each sample.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np


class CaptionGenerator:
    """Generate descriptive captions for pedestrian samples."""

    def __init__(
        self,
        trigger_token: str = "<vin_ped>",
        template_version: str = "v1",
    ):
        self.trigger_token = trigger_token
        self.template_version = template_version

    def generate_caption(
        self,
        bbox_height_ratio: float,
        visible_ratio: float,
        occlusion_level: Optional[float],
        camera_domain: Optional[str] = None,
    ) -> str:
        """Generate caption for a single sample.

        Args:
            bbox_height_ratio: Person height as fraction of image height
            visible_ratio: Visible portion of person
            occlusion_level: Occlusion level (0 = none, 1 = fully occluded)
            camera_domain: Camera domain hint

        Returns:
            Caption string with trigger token
        """
        parts = [f"photo of {self.trigger_token} a pedestrian"]

        # Body visibility
        if bbox_height_ratio > 0.35:
            parts.append("full body")
        elif bbox_height_ratio > 0.20:
            parts.append("full body")
        elif bbox_height_ratio > 0.10:
            parts.append("distant pedestrian")
        else:
            parts.append("distant pedestrian")

        # Occlusion
        if occlusion_level is not None and occlusion_level > 0.4:
            parts.append("partially occluded")
        elif visible_ratio < 0.6:
            parts.append("partially occluded")
        else:
            parts.append("clear")

        # Scene context
        if camera_domain and "cctv" in camera_domain.lower():
            parts.append("in CCTV scene")
        elif camera_domain and "surveillance" in camera_domain.lower():
            parts.append("in urban street")
        else:
            parts.append("in urban street")

        return ", ".join(parts)

    def generate_all_captions(self, samples_df: pd.DataFrame, images_df: pd.DataFrame) -> pd.DataFrame:
        """Generate captions for all samples.

        Args:
            samples_df: Sample manifest
            images_df: Image manifest (for camera_domain)

        Returns:
            samples_df with caption column filled
        """
        print("=" * 60)
        print("GENERATING CAPTIONS")
        print("=" * 60)

        # Merge to get camera domain
        merged = samples_df.merge(
            images_df[['image_id', 'camera_domain']],
            on='image_id',
            how='left'
        )

        captions = []

        for _, row in merged.iterrows():
            caption = self.generate_caption(
                bbox_height_ratio=row.get('bbox_height_ratio', 0.15),
                visible_ratio=row.get('visible_ratio', 1.0),
                occlusion_level=row.get('occlusion_level'),
                camera_domain=row.get('camera_domain'),
            )
            captions.append(caption)

        samples_df['caption'] = captions
        samples_df['trigger_token'] = self.trigger_token

        print(f"\n✓ Generated {len(captions)} captions")
        print(f"\nSample captions:")
        for caption in captions[:5]:
            print(f"  - {caption}")

        return samples_df


def validate_captions(
    samples_df: pd.DataFrame,
    trigger_token: str,
    min_tokens: int = 5,
    max_tokens: int = 77,
) -> List[str]:
    """Validate caption quality.

    Args:
        samples_df: Sample manifest with captions
        trigger_token: Required trigger token
        min_tokens: Minimum token count
        max_tokens: Maximum token count

    Returns:
        List of validation errors
    """
    errors = []

    for idx, row in samples_df.iterrows():
        caption = row.get('caption', '')

        # Check trigger token
        if trigger_token not in caption:
            errors.append(f"Sample {row['sample_id']}: missing trigger token '{trigger_token}'")

        # Check length
        token_count = len(caption.split())
        if token_count < min_tokens:
            errors.append(f"Sample {row['sample_id']}: caption too short ({token_count} tokens)")
        elif token_count > max_tokens:
            errors.append(f"Sample {row['sample_id']}: caption too long ({token_count} tokens)")

        # Check for prohibited terms
        prohibited = ['gender', 'age', 'race', 'ethnicity', 'emotion', 'identity', 'name']
        caption_lower = caption.lower()
        for term in prohibited:
            if term in caption_lower:
                errors.append(f"Sample {row['sample_id']}: contains prohibited term '{term}'")

    return errors
