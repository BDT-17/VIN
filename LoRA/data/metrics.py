"""Release metrics and downstream benchmark comparison utilities."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


DETECTION_METRIC_DIRECTIONS = {
    "ap50": "higher_is_better",
    "ap75": "higher_is_better",
    "map50_95": "higher_is_better",
    "mr_minus_2": "lower_is_better",
}


def _safe_float(value: Any) -> Optional[float]:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _count_dict(series: pd.Series) -> Dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def _numeric_summary(series: pd.Series) -> Dict[str, Optional[float]]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {"mean": None, "median": None, "p05": None, "p95": None, "min": None, "max": None}
    return {
        "mean": round(float(numeric.mean()), 6),
        "median": round(float(numeric.median()), 6),
        "p05": round(float(numeric.quantile(0.05)), 6),
        "p95": round(float(numeric.quantile(0.95)), 6),
        "min": round(float(numeric.min()), 6),
        "max": round(float(numeric.max()), 6),
    }


def summarize_release_metrics(
    samples_df: pd.DataFrame,
    images_df: pd.DataFrame,
    groups_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Summarize the data contract metrics needed for release reporting."""
    lora_samples = samples_df[samples_df["role"] == "lora_positive"]
    benchmark_roles = {"detector_val_real_frozen", "detector_test_real_frozen"}
    benchmark_samples = samples_df[samples_df["role"].isin(benchmark_roles)]

    train_groups = set(samples_df[samples_df["split"] == "train"]["split_group_id"].dropna())
    val_groups = set(samples_df[samples_df["split"] == "val"]["split_group_id"].dropna())
    test_groups = set(samples_df[samples_df["split"] == "test"]["split_group_id"].dropna())
    lora_train_val = lora_samples[lora_samples["split"].isin(["train", "val"])]
    benchmark_image_ids = set(benchmark_samples["image_id"].dropna())
    benchmark_overlap_count = int(lora_train_val["image_id"].isin(benchmark_image_ids).sum())

    caption_lengths = lora_samples["caption"].fillna("").astype(str).str.split().str.len()
    trigger_missing_count = 0
    if "trigger_token" in lora_samples.columns and len(lora_samples) > 0:
        trigger_missing_count = int(
            lora_samples.apply(
                lambda row: str(row.get("trigger_token", "")) not in str(row.get("caption", "")),
                axis=1,
            ).sum()
        )

    return {
        "sample_counts": {
            "total": int(len(samples_df)),
            "lora_positive": int(len(lora_samples)),
            "benchmark_frozen": int(len(benchmark_samples)),
            "by_role": _count_dict(samples_df["role"]),
            "by_split": _count_dict(samples_df["split"]),
            "lora_by_split": _count_dict(lora_samples["split"]),
            "lora_by_source": _count_dict(lora_samples["source_id"]),
        },
        "image_counts": {
            "total_manifest_images": int(len(images_df)),
            "unique_sample_images": int(samples_df["image_id"].nunique()),
            "sources": int(images_df["source_id"].nunique()) if "source_id" in images_df else 0,
        },
        "quality": {
            "quality_score": _numeric_summary(lora_samples["quality_score"]),
            "bbox_height_ratio": _numeric_summary(lora_samples["bbox_height_ratio"]),
            "visible_ratio": _numeric_summary(lora_samples["visible_ratio"]),
            "occlusion_level": _numeric_summary(lora_samples["occlusion_level"]),
        },
        "captions": {
            "missing_caption_count": int((lora_samples["caption"].fillna("") == "").sum()),
            "trigger_missing_count": trigger_missing_count,
            "token_count": _numeric_summary(caption_lengths),
        },
        "deduplication": {
            "by_status": _count_dict(groups_df["dedupe_status"]),
            "unique_split_groups": int(samples_df["split_group_id"].nunique()),
        },
        "split_safety": {
            "train_groups": int(len(train_groups)),
            "val_groups": int(len(val_groups)),
            "test_groups": int(len(test_groups)),
            "cross_train_val": int(len(train_groups & val_groups)),
            "cross_train_test": int(len(train_groups & test_groups)),
            "cross_val_test": int(len(val_groups & test_groups)),
            "benchmark_overlap_count": benchmark_overlap_count,
            "zero_leakage": bool(
                len(train_groups & val_groups) == 0
                and len(train_groups & test_groups) == 0
                and len(val_groups & test_groups) == 0
                and benchmark_overlap_count == 0
            ),
        },
    }


def load_metrics_json(path: Path) -> Dict[str, Any]:
    """Load a detector metrics JSON file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compare_downstream_benchmarks(
    baseline_metrics: Dict[str, Any],
    lora_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare baseline and LoRA downstream detector metrics."""
    comparison = {}
    for metric, direction in DETECTION_METRIC_DIRECTIONS.items():
        baseline_value = _safe_float(baseline_metrics.get(metric))
        lora_value = _safe_float(lora_metrics.get(metric))
        delta = None
        relative_delta = None
        improved = None
        if baseline_value is not None and lora_value is not None:
            delta = round(lora_value - baseline_value, 6)
            if baseline_value != 0:
                relative_delta = round(delta / abs(baseline_value), 6)
            improved = delta >= 0 if direction == "higher_is_better" else delta <= 0
        comparison[metric] = {
            "baseline": baseline_value,
            "lora": lora_value,
            "delta": delta,
            "relative_delta": relative_delta,
            "direction": direction,
            "improved": improved,
        }
    return comparison


def write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_flat_csv(path: Path, payload: Dict[str, Any]) -> Path:
    """Write a one-row CSV with dotted keys for nested dictionaries."""
    def flatten(prefix: str, value: Any, output: Dict[str, Any]) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                flatten(f"{prefix}.{key}" if prefix else str(key), item, output)
        else:
            output[prefix] = value

    row: Dict[str, Any] = {}
    flatten("", payload, row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return path


def write_markdown_report(
    path: Path,
    release_name: str,
    summary: Dict[str, Any],
    benchmark: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a compact PR4 markdown report for release review."""
    sample_counts = summary["sample_counts"]
    split_safety = summary["split_safety"]
    quality = summary["quality"]["quality_score"]
    lines = [
        f"# {release_name} metrics report",
        "",
        "## Dataset",
        f"- total samples: {sample_counts['total']}",
        f"- LoRA positives: {sample_counts['lora_positive']}",
        f"- frozen benchmark samples: {sample_counts['benchmark_frozen']}",
        f"- LoRA split counts: {sample_counts['lora_by_split']}",
        "",
        "## Quality",
        f"- mean quality score: {quality['mean']}",
        f"- p05/p95 quality score: {quality['p05']} / {quality['p95']}",
        "",
        "## Split safety",
        f"- zero leakage: {split_safety['zero_leakage']}",
        f"- cross train/val groups: {split_safety['cross_train_val']}",
        f"- benchmark overlap count: {split_safety['benchmark_overlap_count']}",
    ]
    if benchmark:
        lines.extend(["", "## Downstream benchmark"])
        for metric, values in benchmark.items():
            lines.append(
                f"- {metric}: baseline={values['baseline']} lora={values['lora']} "
                f"delta={values['delta']} improved={values['improved']}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

