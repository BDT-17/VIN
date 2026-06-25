"""Classification-folders parser: <mount>/<positive_dir> holds person images.

Only positive ("1") images become LoRA candidates, and they must be
detected/pseudo-labeled downstream (curate) before a usable crop exists.
Each positive image gets one whole-image instance as a detection placeholder.
"""

from pathlib import Path
from typing import List, Tuple

from ..schema import ImageRecord, InstanceRecord, PEDESTRIAN

_IMG_EXT = (".jpg", ".jpeg", ".png")


class ClassificationParser:
    def __init__(self, source_id: str, mount_path, positive_dir="1", negative_dir="0",
                 lora_splits=None, eval_splits=None):
        self.source_id = source_id
        self.mount_path = Path(mount_path)
        self.positive_dir = positive_dir
        self.split_name = (lora_splits or ["train"])[0] if (lora_splits or eval_splits) else "train"

    def parse(self) -> Tuple[List[ImageRecord], List[InstanceRecord]]:
        pos_dir = self.mount_path / self.positive_dir
        if not pos_dir.exists():
            return [], []

        images: List[ImageRecord] = []
        instances: List[InstanceRecord] = []
        for img_path in sorted(pos_dir.iterdir()):
            if img_path.suffix.lower() not in _IMG_EXT:
                continue
            w, h = _image_size(img_path)
            image_id = f"{self.source_id}_pos_{img_path.stem}"
            images.append(ImageRecord(
                image_id=image_id,
                source_id=self.source_id,
                raw_path=str(img_path),
                source_image_id=img_path.stem,
                original_split=self.split_name,
                width=w, height=h,
                group_id=f"{self.source_id}_pos_{img_path.stem}",
            ))
            # whole-image placeholder; curate will tighten via detection
            instances.append(InstanceRecord(
                instance_id=f"{image_id}_0",
                image_id=image_id,
                class_name=PEDESTRIAN,
                bbox_x=0.0, bbox_y=0.0, bbox_w=float(w), bbox_h=float(h),
                visible_bbox_x=0.0, visible_bbox_y=0.0,
                visible_bbox_w=float(w), visible_bbox_h=float(h),
                ignore_flag=False,
                confidence=0.0,  # 0 == not yet detected
            ))
        return images, instances


def _image_size(path: Path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        return 0, 0
