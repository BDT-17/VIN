"""Stage 02 — dedupe + group lock.

- sha256: exact duplicates collapse to one canonical image.
- phash (Hamming <= threshold) within a source: near-duplicate cluster.
- group_id is preserved from the parser (scene / sequence-window / per-image)
  so entire groups stay together at split time.

Output: <work>/etl/curated/groups.parquet with columns
    image_id, group_id, duplicate_cluster_id, dedupe_status, canonical_image_id
"""

from pathlib import Path

import pandas as pd

_PHASH_HAMMING_MAX = 6


def _hamming(a: str, b: str) -> int:
    if not a or not b or len(a) != len(b):
        return 999
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except ValueError:
        return 999


def run_dedupe(normalized_dir: Path, work_dir: Path) -> Path:
    normalized_dir = Path(normalized_dir)
    out_dir = Path(work_dir) / "etl" / "curated"
    out_dir.mkdir(parents=True, exist_ok=True)

    images_df = pd.read_parquet(normalized_dir / "images.parquet").reset_index(drop=True)

    rows = []
    # exact duplicates by sha256
    canonical_by_sha = {}
    for _, img in images_df.iterrows():
        sha = img.get("sha256") or img["image_id"]
        canonical = canonical_by_sha.setdefault(sha, img["image_id"])
        status = "unique" if canonical == img["image_id"] else "exact_duplicate"
        rows.append({
            "image_id": img["image_id"],
            "group_id": img["group_id"],
            "source_id": img["source_id"],
            "phash": img.get("phash", ""),
            "dedupe_status": status,
            "canonical_image_id": canonical,
        })

    groups_df = pd.DataFrame(rows)

    # near-duplicate clusters by phash within a source (only among uniques)
    groups_df["duplicate_cluster_id"] = groups_df["image_id"]
    for source_id, grp in groups_df[groups_df["dedupe_status"] == "unique"].groupby("source_id"):
        members = grp.to_dict("records")
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if _hamming(members[i]["phash"], members[j]["phash"]) <= _PHASH_HAMMING_MAX:
                    cid = members[i]["duplicate_cluster_id"]
                    groups_df.loc[groups_df["image_id"] == members[j]["image_id"],
                                  "duplicate_cluster_id"] = cid
                    if members[j]["dedupe_status"] == "unique":
                        groups_df.loc[groups_df["image_id"] == members[j]["image_id"],
                                      "dedupe_status"] = "near_duplicate"

    groups_df = groups_df.drop(columns=["phash"])
    groups_df.to_parquet(out_dir / "groups.parquet", index=False)
    n_dup = int((groups_df["dedupe_status"] != "unique").sum())
    print(f"  dedupe: {len(groups_df)} images, {n_dup} duplicates/near-duplicates, "
          f"{groups_df['duplicate_cluster_id'].nunique()} clusters, "
          f"{groups_df['group_id'].nunique()} split groups")
    return out_dir
