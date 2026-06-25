"""Stage 06 — group-aware train/val split of LoRA candidates.

Splits by group_id (never per-image), so an entire scene / sequence-window /
near-duplicate cluster stays on one side. Eval-locked images are already absent
from candidates (curate excluded them), so no benchmark leakage is possible here.

Output: samples_df with role=lora_positive and split in {train, val}.
"""

import numpy as np
import pandas as pd

from .config import SplitConfig
from .schema import ROLE_LORA_POSITIVE


def assign_splits(candidates_df: pd.DataFrame, split_cfg: SplitConfig,
                  groups_df: pd.DataFrame = None) -> pd.DataFrame:
    df = candidates_df.copy()

    # collapse near/exact duplicates to canonical images if dedupe info present
    if groups_df is not None and "canonical_image_id" in groups_df.columns:
        canon = groups_df.set_index("image_id")["canonical_image_id"].to_dict()
        keep = df["image_id"].map(lambda i: canon.get(i, i) == i)
        df = df[keep].copy()
        cl = groups_df.set_index("image_id")["duplicate_cluster_id"].to_dict()
        df["duplicate_cluster_id"] = df["image_id"].map(lambda i: cl.get(i, i))
    else:
        df["duplicate_cluster_id"] = df["image_id"]

    groups = df["group_id"].dropna().unique().tolist()
    rng = np.random.RandomState(split_cfg.split_seed)
    rng.shuffle(groups)
    train_end = int(len(groups) * split_cfg.train_ratio)
    group_split = {g: ("train" if i < train_end else "val") for i, g in enumerate(groups)}

    df["role"] = ROLE_LORA_POSITIVE
    df["split"] = df["group_id"].map(group_split)
    df = df[df["split"].notna()].copy()

    n_train = int((df["split"] == "train").sum())
    n_val = int((df["split"] == "val").sum())
    print(f"  split: {n_train} train, {n_val} val "
          f"({df['group_id'].nunique()} groups, group-disjoint)")
    return df
