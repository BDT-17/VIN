"""MOT (Multiple Object Tracking) format parser.

Parses MOT17-style tracking datasets with gt.txt annotations.
"""

import hashlib
import imagehash
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from PIL import Image
import pandas as pd

from ..schema import ImageRecord, InstanceRecord


class MOTParser:
    """Parser for MOT17-style video tracking datasets.

    Expected structure:
        MOT17-02-FRCNN/
            img1/
                000001.jpg
                000002.jpg
                ...
            gt/
                gt.txt

    gt.txt format (CSV, no header):
        frame_id, track_id, x, y, w, h, conf, class, visibility
    """

    def __init__(
        self,
        source_id: str,
        mount_path: Path,
        sequence_dir: str,
        gt_dir: str,
        temporal_sampling_fps: Optional[float] = None,
        temporal_window_seconds: Optional[float] = None,
    ):
        self.source_id = source_id
        self.mount_path = Path(mount_path)
        self.sequence_dir = sequence_dir
        self.gt_dir = gt_dir
        self.temporal_sampling_fps = temporal_sampling_fps or 2.0
        self.temporal_window_seconds = temporal_window_seconds or 2.0

        self.img_dir = self.mount_path / sequence_dir
        self.gt_path = self.mount_path / gt_dir / "gt.txt"

        if not self.img_dir.exists():
            raise FileNotFoundError(f"MOT image directory not found: {self.img_dir}")

    def parse(self) -> Tuple[List[ImageRecord], List[InstanceRecord]]:
        """Parse MOT dataset into canonical records."""
        images = []
        instances = []

        # Load ground truth if available
        gt_df = None
        if self.gt_path.exists():
            gt_df = self._load_gt()

        # Discover frames
        image_files = sorted(self.img_dir.glob("*.jpg")) + sorted(self.img_dir.glob("*.png"))

        # Apply temporal sampling
        sampled_files = self._temporal_sample(image_files)

        for frame_idx, img_path in enumerate(sampled_files, start=1):
            # Create image record
            image_id = f"{self.source_id}_{img_path.stem}"

            with Image.open(img_path) as pil_img:
                width, height = pil_img.size

                # Compute hashes
                sha256 = self._compute_sha256(img_path)
                phash = str(imagehash.phash(pil_img))

            # Group ID based on temporal window
            group_id = self._compute_group_id(frame_idx)

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
                original_split="train",
                frame_index=int(img_path.stem),
                camera_domain="mot_surveillance",
                file_size_bytes=img_path.stat().st_size,
            )
            images.append(image_rec)

            # Parse instances for this frame
            if gt_df is not None:
                frame_num = int(img_path.stem)
                frame_instances = self._parse_frame_instances(
                    gt_df, frame_num, image_id, width, height
                )
                instances.extend(frame_instances)

        return images, instances

    def _load_gt(self) -> pd.DataFrame:
        """Load MOT ground truth file."""
        # MOT format: frame, track_id, x, y, w, h, conf, class, visibility
        df = pd.read_csv(
            self.gt_path,
            header=None,
            names=["frame", "track_id", "x", "y", "w", "h", "conf", "class", "visibility"],
        )
        return df

    def _temporal_sample(self, image_files: List[Path]) -> List[Path]:
        """Sample frames temporally to reduce redundancy."""
        if not image_files:
            return []

        # Assume 30 FPS for MOT17
        source_fps = 30.0
        sample_interval = max(1, int(source_fps / self.temporal_sampling_fps))

        return image_files[::sample_interval]

    def _compute_group_id(self, frame_idx: int) -> str:
        """Compute group ID based on temporal window."""
        # Group frames within temporal window together
        window_frames = int(self.temporal_window_seconds * self.temporal_sampling_fps)
        group_num = (frame_idx - 1) // window_frames
        return f"{self.source_id}_window_{group_num:04d}"

    def _parse_frame_instances(
        self,
        gt_df: pd.DataFrame,
        frame_num: int,
        image_id: str,
        width: int,
        height: int,
    ) -> List[InstanceRecord]:
        """Parse instances for a single frame."""
        instances = []
        frame_df = gt_df[gt_df["frame"] == frame_num]

        for idx, row in frame_df.iterrows():
            # MOT class labels: 1=pedestrian, 2=person_on_vehicle, etc.
            # Only keep pedestrians (class 1)
            if row["class"] != 1:
                continue

            instance_id = f"{image_id}_inst_{row['track_id']:04d}"

            # MOT coordinates are 1-indexed, convert to 0-indexed
            x = max(0, row["x"] - 1)
            y = max(0, row["y"] - 1)
            w = row["w"]
            h = row["h"]

            # Visibility is fraction visible [0, 1]
            visibility = row["visibility"]

            instance = InstanceRecord(
                instance_id=instance_id,
                image_id=image_id,
                class_name="pedestrian",
                bbox_x=x,
                bbox_y=y,
                bbox_w=w,
                bbox_h=h,
                visible_bbox_x=x,
                visible_bbox_y=y,
                visible_bbox_w=w,
                visible_bbox_h=h,
                track_id=int(row["track_id"]),
                occlusion_level=1.0 - visibility,
                ignore_flag=False,
                confidence=row["conf"],
                annotation_origin="mot_gt",
            )
            instances.append(instance)

        return instances

    def _compute_sha256(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
