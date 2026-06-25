"""Stage 00 — ingest: parse sources into a raw image inventory.

Output: <work>/etl/inventory/raw_inventory.parquet
Each row: source_id, raw_path, source_image_id, original_split, width, height,
          sha256, phash, group_id  (+ parsed instances kept in memory/parquet).
"""

import hashlib
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .config import SourcesConfig
from .parsers import YOLOParser, MOTParser, ClassificationParser
from .schema import ImageRecord, InstanceRecord


def _build_parser(src):
    common = dict(lora_splits=src.lora_splits, eval_splits=src.eval_splits)
    if src.parser == "yolo":
        return YOLOParser(src.source_id, src.kaggle_mount, src.splits, src.label_dirs, **common)
    if src.parser == "mot":
        return MOTParser(src.source_id, src.kaggle_mount, src.sequence_dir, src.gt_dir,
                         temporal_sampling_fps=src.temporal_sampling_fps,
                         temporal_window_seconds=src.temporal_window_seconds, **common)
    if src.parser == "classification_folders":
        return ClassificationParser(src.source_id, src.kaggle_mount,
                                    src.positive_dir, src.negative_dir, **common)
    raise ValueError(f"Unknown parser: {src.parser}")


def run_ingest(config: SourcesConfig, work_dir: Path) -> Path:
    work_dir = Path(work_dir)
    out_dir = work_dir / "etl" / "inventory"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_images: List[ImageRecord] = []
    all_instances: List[InstanceRecord] = []
    for src in config.sources:
        parser = _build_parser(src)
        images, instances = parser.parse()
        print(f"  {src.source_id}: {len(images)} images, {len(instances)} instances")
        all_images.extend(images)
        all_instances.extend(instances)

    if not all_images:
        raise FileNotFoundError("No images parsed. Check dataset mounts in sources.yaml.")

    # content hashes (sha256 exact, phash near-dup)
    for img in all_images:
        img.sha256, img.phash = _hashes(Path(img.raw_path))

    images_df = pd.DataFrame([i.to_dict() for i in all_images])
    instances_df = pd.DataFrame([i.to_dict() for i in all_instances])

    images_df.to_parquet(out_dir / "raw_inventory.parquet", index=False)
    instances_df.to_parquet(out_dir / "raw_instances.parquet", index=False)
    print(f"  -> {out_dir / 'raw_inventory.parquet'} ({len(images_df)} rows)")
    return out_dir


def _hashes(path: Path) -> Tuple[str, str]:
    sha = ""
    ph = ""
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        sha = h.hexdigest()
    except Exception:
        pass
    try:
        import imagehash
        from PIL import Image
        with Image.open(path) as im:
            ph = str(imagehash.phash(im))
    except Exception:
        pass
    return sha, ph
