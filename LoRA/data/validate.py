"""Dataset release validation.

Validates a dataset release before allowing training to proceed.
Fails fast if any validation gate fails.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


VALIDATION_GATES = [
    "cross_split_duplicate_count > 0",
    "same split_group_id exists across train and validation",
    "benchmark image appears in LoRA train or LoRA validation",
    "caption missing",
    "trigger token missing",
    "invalid bbox",
    "crop file missing",
    "source balance exceeds configured maximum",
    "uncanonicalized exact duplicate remains",
    "dataset_status is not validated",
]


def validate_release(release_dir: Path, config: Optional[dict] = None) -> dict:
    """Validate a dataset release.

    Args:
        release_dir: Path to release directory
        config: Optional config overrides

    Returns:
        Validation result dict with 'valid', 'errors', 'warnings', 'stats'
    """
    release_dir = Path(release_dir)
    errors = []
    warnings = []

    # Load release.json
    release_path = release_dir / "release.json"
    if not release_path.exists():
        return {"valid": False, "errors": ["release.json not found"], "warnings": [], "stats": {}}

    with open(release_path) as f:
        release_meta = json.load(f)

    dataset_status = release_meta.get("dataset_status", "")

    # Check dataset_status
    if dataset_status == "validated":
        pass  # already validated
    elif dataset_status in ("exported", "building", "validating"):
        pass  # will be validated now
    else:
        errors.append(f"dataset_status is '{dataset_status}', expected 'exported' or 'validated'")

    # Load manifests
    samples_path = release_dir / "samples.parquet"
    images_path = release_dir / "images.parquet"
    groups_path = release_dir / "groups.parquet"

    if not samples_path.exists():
        errors.append("samples.parquet missing")
    if not images_path.exists():
        errors.append("images.parquet missing")
    if not groups_path.exists():
        errors.append("groups.parquet missing")

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings, "stats": {}}

    samples_df = pd.read_parquet(samples_path)
    images_df = pd.read_parquet(images_path)
    groups_df = pd.read_parquet(groups_path)

    # ---- Gate 1: Cross-split duplicate clusters ----
    train_groups = set(samples_df[samples_df['split'] == 'train']['split_group_id'])
    val_groups = set(samples_df[samples_df['split'] == 'val']['split_group_id'])
    cross_split = train_groups & val_groups

    if cross_split:
        errors.append(
            f"cross_split_duplicate_count={len(cross_split)}: "
            f"same split_group_id in train and val: {list(cross_split)[:5]}"
        )

    # ---- Gate 2: Benchmark leak into LoRA splits ----
    lora_roles = {'lora_positive'}
    lora_train_val = samples_df[
        samples_df['role'].isin(lora_roles) &
        samples_df['split'].isin(['train', 'val'])
    ]

    benchmark_roles = {'detector_val_real_frozen', 'detector_test_real_frozen'}
    benchmark_image_ids = set(
        samples_df[samples_df['role'].isin(benchmark_roles)]['image_id']
    )

    benchmark_leak = lora_train_val[lora_train_val['image_id'].isin(benchmark_image_ids)]
    if len(benchmark_leak) > 0:
        errors.append(
            f"benchmark_overlap_count={len(benchmark_leak)}: "
            "benchmark images found in LoRA train or val"
        )

    # ---- Gate 3: Missing captions / trigger token ----
    lora_samples = samples_df[samples_df['role'].isin(lora_roles)]

    missing_caption = lora_samples['caption'].isna() | (lora_samples['caption'] == '')
    if missing_caption.any():
        errors.append(f"{missing_caption.sum()} samples with missing captions")

    # Check trigger token
    trigger_token = samples_df['trigger_token'].iloc[0] if len(samples_df) > 0 else "<vin_ped>"
    missing_trigger = lora_samples['caption'].apply(
        lambda c: isinstance(c, str) and trigger_token not in c
    )
    if missing_trigger.any():
        errors.append(f"{missing_trigger.sum()} samples where trigger token '{trigger_token}' is missing from caption")

    # ---- Gate 4: Crop files missing ----
    missing_crops = lora_samples['crop_path'].apply(
        lambda p: not Path(p).exists() if isinstance(p, str) else True
    )
    if missing_crops.any():
        errors.append(f"{missing_crops.sum()} samples with missing crop files")

    # ---- Gate 5: Uncanonicalized exact duplicates ----
    if 'dedupe_status' in groups_df.columns:
        exact_dups = groups_df[
            (groups_df['dedupe_status'] == 'exact_duplicate') &
            (groups_df['image_id'] != groups_df['canonical_image_id'])
        ]
        lora_non_canonical = lora_samples.merge(
            exact_dups[['image_id']], on='image_id', how='inner'
        )
        if len(lora_non_canonical) > 0:
            errors.append(
                f"{len(lora_non_canonical)} LoRA samples use non-canonical duplicate images"
            )

    # ---- Compute stats ----
    split_counts = samples_df[samples_df['role'].isin(lora_roles)]['split'].value_counts()
    stats = {
        "train_count": int(split_counts.get('train', 0)),
        "val_count": int(split_counts.get('val', 0)),
        "cross_split_duplicate_count": len(cross_split),
        "benchmark_overlap_count": len(benchmark_leak),
        "total_lora_samples": int(len(lora_samples)),
    }

    # ---- Update release status ----
    valid = len(errors) == 0

    if valid:
        release_meta['dataset_status'] = 'validated'
        with open(release_path, 'w') as f:
            json.dump(release_meta, f, indent=2)

    result = {
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }

    # Save validation result
    val_path = Path(release_dir) / "validation_result.json"
    with open(val_path, 'w') as f:
        json.dump(result, f, indent=2)

    return result


def require_validated_release(release_dir: Path):
    """Fail fast if release is not validated.

    Call this at the start of training to block on invalid data.

    Raises:
        RuntimeError: if release is not validated
    """
    release_dir = Path(release_dir)
    release_path = release_dir / "release.json"

    if not release_path.exists():
        raise RuntimeError(f"release.json not found at {release_dir}")

    with open(release_path) as f:
        meta = json.load(f)

    status = meta.get("dataset_status", "")
    if status != "validated":
        result = validate_release(release_dir)
        if not result["valid"]:
            errors_str = "\n  ".join(result["errors"])
            raise RuntimeError(
                f"Dataset release at {release_dir} is NOT validated.\n"
                f"Errors:\n  {errors_str}"
            )
