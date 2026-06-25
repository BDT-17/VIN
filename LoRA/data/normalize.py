"""Stage 01 — normalize: canonical images.parquet + instances.parquet.

Keeps only class_name == pedestrian and ignore_flag == False.
Output: <work>/etl/normalized/{images,instances}.parquet
"""

from pathlib import Path

import pandas as pd

from .schema import PEDESTRIAN


def run_normalize(inventory_dir: Path, work_dir: Path) -> Path:
    inventory_dir = Path(inventory_dir)
    out_dir = Path(work_dir) / "etl" / "normalized"
    out_dir.mkdir(parents=True, exist_ok=True)

    images_df = pd.read_parquet(inventory_dir / "raw_inventory.parquet")
    instances_df = pd.read_parquet(inventory_dir / "raw_instances.parquet")

    kept = instances_df[
        (instances_df["class_name"] == PEDESTRIAN) & (~instances_df["ignore_flag"].astype(bool))
    ].copy()

    # drop images with no surviving pedestrian instance
    keep_ids = set(kept["image_id"])
    images_df = images_df[images_df["image_id"].isin(keep_ids)].copy()

    images_df.to_parquet(out_dir / "images.parquet", index=False)
    kept.to_parquet(out_dir / "instances.parquet", index=False)
    print(f"  normalized: {len(images_df)} images, {len(kept)} pedestrian instances")
    return out_dir
