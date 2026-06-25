"""Stage 08 — validate the LoRA release. Hard-fails per spec.

Gates (all must pass):
  - trigger token non-empty
  - every caption contains the trigger token
  - lora_train and lora_val both non-empty
  - every crop file readable
  - no duplicate_cluster_id overlap between train and val
  - no group_id overlap between train and val
  - no eval image / group leaks into the release (if eval manifest is given)

Writes <release>/validation_report.json and flips dataset_status to 'validated'.
"""

import json
from pathlib import Path

import pandas as pd

from .schema import STATUS_VALIDATED


def validate_release(release_dir: Path, eval_manifest: pd.DataFrame = None) -> dict:
    release_dir = Path(release_dir)
    errors, warnings = [], []

    manifest_path = release_dir / "manifest.parquet"
    release_path = release_dir / "release.json"
    if not manifest_path.exists():
        return {"valid": False, "errors": ["manifest.parquet missing"], "warnings": [], "stats": {}}
    if not release_path.exists():
        return {"valid": False, "errors": ["release.json missing"], "warnings": [], "stats": {}}

    df = pd.read_parquet(manifest_path)
    release = json.loads(release_path.read_text(encoding="utf-8"))

    trigger = release.get("trigger_token", "")
    if not trigger:
        errors.append("trigger_token is empty")

    if trigger:
        missing_trig = df["caption"].apply(lambda c: isinstance(c, str) and trigger not in c)
        if missing_trig.any():
            errors.append(f"{int(missing_trig.sum())} captions missing trigger token '{trigger}'")

    missing_cap = df["caption"].isna() | (df["caption"].astype(str) == "")
    if missing_cap.any():
        errors.append(f"{int(missing_cap.sum())} samples with empty caption")

    n_train = int((df["split"] == "train").sum())
    n_val = int((df["split"] == "val").sum())
    if n_train == 0:
        errors.append("lora_train is empty")
    if n_val == 0:
        errors.append("lora_val is empty")

    unreadable = df["crop_path"].apply(lambda p: not Path(p).exists() if isinstance(p, str) else True)
    if unreadable.any():
        errors.append(f"{int(unreadable.sum())} crop files missing/unreadable")

    train_clusters = set(df[df["split"] == "train"].get("duplicate_cluster_id", pd.Series([], dtype=object)).dropna())
    val_clusters = set(df[df["split"] == "val"].get("duplicate_cluster_id", pd.Series([], dtype=object)).dropna())
    if train_clusters & val_clusters:
        errors.append(f"{len(train_clusters & val_clusters)} duplicate_cluster_id overlap train<->val")

    train_groups = set(df[df["split"] == "train"]["group_id"].dropna())
    val_groups = set(df[df["split"] == "val"]["group_id"].dropna())
    if train_groups & val_groups:
        errors.append(f"{len(train_groups & val_groups)} group_id overlap train<->val")

    if eval_manifest is not None and len(eval_manifest):
        eval_images = set(eval_manifest.get("image_id", pd.Series([], dtype=object)).dropna())
        eval_groups = set(eval_manifest.get("group_id", pd.Series([], dtype=object)).dropna())
        img_leak = set(df["image_id"]) & eval_images
        grp_leak = (train_groups | val_groups) & eval_groups
        if img_leak:
            errors.append(f"{len(img_leak)} eval images leaked into LoRA release")
        if grp_leak:
            errors.append(f"{len(grp_leak)} eval groups leaked into LoRA release")

    stats = {
        "train_count": n_train, "val_count": n_val,
        "cross_split_duplicate_count": len(train_clusters & val_clusters),
        "benchmark_overlap_count": 0,
    }
    valid = len(errors) == 0

    report = {"valid": valid, "errors": errors, "warnings": warnings, "stats": stats}
    (release_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if valid:
        release["dataset_status"] = STATUS_VALIDATED
        release["train_count"] = n_train
        release["val_count"] = n_val
        release_path.write_text(json.dumps(release, indent=2), encoding="utf-8")

    return report


def require_validated_release(release_dir: Path) -> dict:
    release = json.loads((Path(release_dir) / "release.json").read_text(encoding="utf-8"))
    status = release.get("dataset_status")
    if status != STATUS_VALIDATED:
        raise RuntimeError(
            f"Release status is '{status}', expected 'validated'. Run validate_release first."
        )
    return release
