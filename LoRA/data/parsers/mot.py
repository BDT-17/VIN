"""MOT-format parser: <mount>/<sequence_dir> frames + <mount>/<gt_dir>/gt.txt.

gt.txt columns: frame,id,x,y,w,h,conf,class,visibility
Temporal sampling avoids consecutive-frame redundancy; frames are grouped by a
time window so a whole window stays on one side of any split.
"""

from pathlib import Path
from typing import List, Tuple

from ..schema import ImageRecord, InstanceRecord, PEDESTRIAN

_IMG_EXT = (".jpg", ".jpeg", ".png")
_MOT_PEDESTRIAN_CLASS = 1


class MOTParser:
    def __init__(self, source_id: str, mount_path, sequence_dir: str, gt_dir: str,
                 temporal_sampling_fps=2.0, temporal_window_seconds=2.0,
                 source_fps=30.0, lora_splits=None, eval_splits=None):
        self.source_id = source_id
        self.mount_path = Path(mount_path)
        self.img_dir = self.mount_path / sequence_dir
        self.gt_path = self.mount_path / gt_dir / "gt.txt" if gt_dir else None
        self.fps = float(temporal_sampling_fps or 2.0)
        self.window_s = float(temporal_window_seconds or 2.0)
        self.source_fps = float(source_fps)
        # MOT sequence frames are treated as 'train' candidates by default
        self.split_name = (lora_splits or ["train"])[0] if (lora_splits or eval_splits) else "train"

    def parse(self) -> Tuple[List[ImageRecord], List[InstanceRecord]]:
        if not self.img_dir.exists():
            return [], []

        gt = self._load_gt()
        frames = sorted(p for p in self.img_dir.iterdir() if p.suffix.lower() in _IMG_EXT)
        sampled = self._temporal_sample(frames)
        window_frames = max(1, int(self.window_s * self.fps))

        images: List[ImageRecord] = []
        instances: List[InstanceRecord] = []
        for order, img_path in enumerate(sampled):
            frame_idx = _frame_index(img_path)
            w, h = _image_size(img_path)
            image_id = f"{self.source_id}_{img_path.stem}"
            window_id = order // window_frames
            images.append(ImageRecord(
                image_id=image_id,
                source_id=self.source_id,
                raw_path=str(img_path),
                source_image_id=img_path.stem,
                original_split=self.split_name,
                width=w, height=h,
                group_id=f"{self.source_id}_window_{window_id}",
            ))
            for i, (x, y, bw, bh, vis) in enumerate(gt.get(frame_idx, [])):
                instances.append(InstanceRecord(
                    instance_id=f"{image_id}_{i}",
                    image_id=image_id,
                    class_name=PEDESTRIAN,
                    bbox_x=x, bbox_y=y, bbox_w=bw, bbox_h=bh,
                    visible_bbox_x=x, visible_bbox_y=y, visible_bbox_w=bw, visible_bbox_h=bh,
                    occlusion_level=(1.0 - vis) if vis is not None else None,
                    ignore_flag=False,
                ))
        return images, instances

    def _temporal_sample(self, frames):
        interval = max(1, int(self.source_fps / self.fps))
        return frames[::interval]

    def _load_gt(self):
        out = {}
        if not self.gt_path or not self.gt_path.exists():
            return out
        for line in self.gt_path.read_text(encoding="utf-8").splitlines():
            c = line.split(",")
            if len(c) < 6:
                continue
            frame = int(float(c[0]))
            if len(c) >= 8 and int(float(c[7])) != _MOT_PEDESTRIAN_CLASS:
                continue
            x, y, bw, bh = float(c[2]), float(c[3]), float(c[4]), float(c[5])
            vis = float(c[8]) if len(c) >= 9 else None
            out.setdefault(frame, []).append((x, y, bw, bh, vis))
        return out


def _frame_index(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else 0


def _image_size(path: Path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        return 0, 0
