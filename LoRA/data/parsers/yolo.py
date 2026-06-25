"""YOLO-format parser: <mount>/<split>/images + <mount>/<split>/labels.

Label lines: `class cx cy w h` (normalized). class 0 == pedestrian.
"""

from pathlib import Path
from typing import Dict, List, Tuple

from ..schema import ImageRecord, InstanceRecord, PEDESTRIAN

_IMG_EXT = (".jpg", ".jpeg", ".png")


class YOLOParser:
    def __init__(self, source_id: str, mount_path, splits: Dict[str, str],
                 label_dirs: Dict[str, str], lora_splits=None, eval_splits=None,
                 class_mapping=None):
        self.source_id = source_id
        self.mount_path = Path(mount_path)
        self.splits = splits or {}
        self.label_dirs = label_dirs or {}
        # only parse splits we actually use (lora or eval)
        self.active = set(lora_splits or []) | set(eval_splits or [])
        self.class_mapping = class_mapping or {0: PEDESTRIAN}

    def parse(self) -> Tuple[List[ImageRecord], List[InstanceRecord]]:
        images: List[ImageRecord] = []
        instances: List[InstanceRecord] = []

        for split_name, img_subdir in self.splits.items():
            if self.active and split_name not in self.active:
                continue
            img_dir = self.mount_path / img_subdir
            label_dir = self.mount_path / self.label_dirs.get(split_name, "")
            if not img_dir.exists():
                continue

            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() not in _IMG_EXT:
                    continue
                w, h = _image_size(img_path)
                image_id = f"{self.source_id}_{split_name}_{img_path.stem}"
                images.append(ImageRecord(
                    image_id=image_id,
                    source_id=self.source_id,
                    raw_path=str(img_path),
                    source_image_id=img_path.stem,
                    original_split=split_name,
                    width=w, height=h,
                    group_id=f"{self.source_id}_{split_name}_{img_path.stem}",
                ))
                label_path = label_dir / f"{img_path.stem}.txt"
                instances.extend(self._parse_labels(label_path, image_id, w, h))

        return images, instances

    def _parse_labels(self, label_path: Path, image_id: str, w: int, h: int):
        out: List[InstanceRecord] = []
        if not label_path.exists():
            return out
        for i, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            class_name = self.class_mapping.get(cls, "ignore")
            cx, cy, bw, bh = (float(parts[1]) * w, float(parts[2]) * h,
                              float(parts[3]) * w, float(parts[4]) * h)
            x, y = cx - bw / 2.0, cy - bh / 2.0
            out.append(InstanceRecord(
                instance_id=f"{image_id}_{i}",
                image_id=image_id,
                class_name=class_name,
                bbox_x=x, bbox_y=y, bbox_w=bw, bbox_h=bh,
                visible_bbox_x=x, visible_bbox_y=y, visible_bbox_w=bw, visible_bbox_h=bh,
                ignore_flag=(class_name != PEDESTRIAN),
            ))
        return out


def _image_size(path: Path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        return 0, 0
