"""Classification folder parser.

Parses binary classification datasets organized into positive/negative folders.
Requires person detection for positive samples to become LoRA training candidates.
"""

import hashlib
import imagehash
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image

from ..schema import ImageRecord, InstanceRecord


class ClassificationParser:
    """Parser for binary classification folder structure.

    Expected structure:
        dataset_root/
            0/  # negative (background only)
                img001.jpg
                ...
            1/  # positive (contains person)
                img002.jpg
                ...

    Note: Positive images are image-level labels only.
    Person detection and pseudo-labeling is required before
    these can become LoRA training samples.
    """

    def __init__(
        self,
        source_id: str,
        mount_path: Path,
        positive_dir: str = "1",
        negative_dir: str = "0",
    ):
        self.source_id = source_id
        self.mount_path = Path(mount_path)
        self.positive_dir = positive_dir
        self.negative_dir = negative_dir

    def parse(self) -> Tuple[List[ImageRecord], List[InstanceRecord]]:
        """Parse classification dataset into canonical records.

        Returns:
            - ImageRecord for all images
            - InstanceRecord will be EMPTY (pseudo-labeling required later)
        """
        images = []
        instances = []  # Will remain empty; pseudo-labeling needed

        # Parse positive folder
        pos_dir = self.mount_path / self.positive_dir
        if pos_dir.exists():
            pos_images = self._parse_folder(pos_dir, "positive")
            images.extend(pos_images)

        # Parse negative folder
        neg_dir = self.mount_path / self.negative_dir
        if neg_dir.exists():
            neg_images = self._parse_folder(neg_dir, "negative")
            images.extend(neg_images)

        return images, instances

    def _parse_folder(
        self,
        folder: Path,
        label: str,
    ) -> List[ImageRecord]:
        """Parse all images in a folder."""
        records = []

        image_files = (
            list(folder.glob("*.jpg"))
            + list(folder.glob("*.jpeg"))
            + list(folder.glob("*.png"))
        )

        for img_path in sorted(image_files):
            image_id = f"{self.source_id}_{label}_{img_path.stem}"

            try:
                with Image.open(img_path) as pil_img:
                    width, height = pil_img.size

                    # Compute hashes
                    sha256 = self._compute_sha256(img_path)
                    phash = str(imagehash.phash(pil_img))
            except Exception as e:
                print(f"Warning: Failed to process {img_path}: {e}")
                continue

            # Group ID: use perceptual cluster (will be computed later via CLIP/DINO)
            # For now, assign placeholder
            group_id = f"{self.source_id}_{label}_cluster_unknown"

            record = ImageRecord(
                image_id=image_id,
                source_id=self.source_id,
                source_image_id=img_path.stem,
                raw_path=str(img_path),
                width=width,
                height=height,
                sha256=sha256,
                phash=phash,
                group_id=group_id,
                original_split=label,
                frame_index=None,
                camera_domain="human_detection_cctv",
                file_size_bytes=img_path.stat().st_size,
            )
            records.append(record)

        return records

    def _compute_sha256(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
