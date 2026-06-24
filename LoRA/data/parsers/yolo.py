"""YOLO format parser.

Parses YOLO-style detection datasets with normalized bbox annotations.
"""

import hashlib
import imagehash
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from PIL import Image

from ..schema import ImageRecord, InstanceRecord


class YOLOParser:
    """Parser for YOLO-format datasets.

    Expected structure:
        dataset_root/
            train/
                images/
                    img001.jpg
                    ...
                labels/
                    img001.txt
                    ...
            valid/ or val/
                images/
                labels/
            test/
                images/
                labels/

    Label format (one bbox per line):
        class_id center_x center_y width height

    All coordinates are normalized to [0, 1].
    """

    def __init__(
        self,
        source_id: str,
        mount_path: Path,
        splits: Dict[str, str],
        label_dirs: Dict[str, str],
        class_mapping: Optional[Dict[int, str]] = None,
    ):
        self.source_id = source_id
        self.mount_path = Path(mount_path)
        self.splits = splits
        self.label_dirs = label_dirs

        # Default: class 0 = pedestrian
        self.class_mapping = class_mapping or {0: "pedestrian"}

    def parse(self) -> Tuple[List[ImageRecord], List[InstanceRecord]]:
        """Parse YOLO dataset into canonical records."""
        images = []
        instances = []

        for split_name, img_subdir in self.splits.items():
            img_dir = self.mount_path / img_subdir
            label_dir = self.mount_path / self.label_dirs.get(split_name, "")

            if not img_dir.exists():
                print(f"Warning: Image directory not found: {img_dir}")
                continue

            # Discover images
            image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))

            for img_path in sorted(image_files):
                # Create image record
                image_id = f"{self.source_id}_{split_name}_{img_path.stem}"

                with Image.open(img_path) as pil_img:
                    width, height = pil_img.size

                    # Compute hashes
                    sha256 = self._compute_sha256(img_path)
                    phash = str(imagehash.phash(pil_img))

                # Group ID: use original scene identifier (filename prefix or full name)
                group_id = self._compute_group_id(img_path.stem, split_name)

                image_rec = ImageRecord(
                    image_id=image_id,
                    source_id=self.source_id,
                    source_image_id=img_path.stem,
                    raw_path=str(img_path),
                    width=width,
                    height=height,
                    sha256=sha256,
                    phash=phash,
                    group_id=group_id,
                    original_split=split_name,
                    frame_index=None,
                    camera_domain="citypersons_surveillance",
                    file_size_bytes=img_path.stat().st_size,
                )
                images.append(image_rec)

                # Parse instances from label file
                label_path = label_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    frame_instances = self._parse_label_file(
                        label_path, image_id, width, height
                    )
                    instances.extend(frame_instances)

        return images, instances

    def _parse_label_file(
        self,
        label_path: Path,
        image_id: str,
        width: int,
        height: int,
    ) -> List[InstanceRecord]:
        """Parse YOLO label file for a single image."""
        instances = []

        with open(label_path, 'r') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                class_name = self.class_mapping.get(class_id, "ignore")

                # YOLO normalized coordinates: center_x, center_y, width, height
                cx_norm = float(parts[1])
                cy_norm = float(parts[2])
                w_norm = float(parts[3])
                h_norm = float(parts[4])

                # Convert to absolute pixel coordinates (top-left corner)
                bbox_w = w_norm * width
                bbox_h = h_norm * height
                bbox_x = (cx_norm * width) - (bbox_w / 2)
                bbox_y = (cy_norm * height) - (bbox_h / 2)

                # Clamp to image bounds
                bbox_x = max(0, bbox_x)
                bbox_y = max(0, bbox_y)
                bbox_w = min(bbox_w, width - bbox_x)
                bbox_h = min(bbox_h, height - bbox_y)

                instance_id = f"{image_id}_inst_{line_idx:04d}"

                instance = InstanceRecord(
                    instance_id=instance_id,
                    image_id=image_id,
                    class_name=class_name,
                    bbox_x=bbox_x,
                    bbox_y=bbox_y,
                    bbox_w=bbox_w,
                    bbox_h=bbox_h,
                    visible_bbox_x=bbox_x,
                    visible_bbox_y=bbox_y,
                    visible_bbox_w=bbox_w,
                    visible_bbox_h=bbox_h,
                    track_id=None,
                    occlusion_level=0.0,
                    ignore_flag=(class_name == "ignore"),
                    confidence=1.0,
                    annotation_origin="yolo_label",
                )
                instances.append(instance)

        return instances

    def _compute_group_id(self, stem: str, split_name: str) -> str:
        """Compute group ID from filename.

        CityPersons images often have structure like:
            aachen_000000_000019_leftImg8bit

        We group by the base scene (e.g., aachen_000000_000019).
        """
        # Split on underscore and take first few parts as scene identifier
        parts = stem.split("_")
        if len(parts) >= 3:
            scene_id = "_".join(parts[:3])
        else:
            scene_id = stem

        return f"{self.source_id}_{split_name}_{scene_id}"

    def _compute_sha256(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
