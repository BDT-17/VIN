"""Generation metrics: baseline vs LoRA comparison on a fixed manifest.

Both runs MUST use the same background manifest and random seeds.
If inputs differ between runs the comparison is not valid.

Output:
    reports/generation/baseline_vs_lora.csv
    reports/generation/baseline_vs_lora_summary.json
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


GENERATION_METRIC_DIRECTIONS = {
    "accept_rate": "higher_is_better",
    "reject_rate": "lower_is_better",
    "person_score": "higher_is_better",
    "scale_score": "higher_is_better",
    "background_score": "higher_is_better",
    "edge_score": "higher_is_better",
    "quality_score": "higher_is_better",
    "placement_score": "higher_is_better",
    "occlusion_score": "higher_is_better",
    "affordance_score": "higher_is_better",
    "background_preservation_score": "higher_is_better",
    "detected_height": "neutral",
    "expected_height": "neutral",
    "scale_ratio_before": "neutral",
    "scale_ratio_after": "neutral",
    "object_mask_inside_ratio": "higher_is_better",
}


def compare_generation_runs(
    baseline_csv: Path,
    lora_csv: Path,
    output_dir: Path,
    join_on: str = "image_id",
) -> Dict[str, Any]:
    """Compare per-image generation metrics from baseline and LoRA runs.

    Args:
        baseline_csv: augmentation_metrics.csv from the baseline (no LoRA) run
        lora_csv: augmentation_metrics.csv from the LoRA run
        output_dir: destination for comparison outputs
        join_on: column identifying individual generation samples for join

    Returns:
        summary dict (also written to baseline_vs_lora_summary.json)

    Raises:
        ValueError: if no rows match after joining (probably different manifests)
    """
    baseline_csv = Path(baseline_csv)
    lora_csv = Path(lora_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_df = pd.read_csv(baseline_csv)
    lora_df = pd.read_csv(lora_csv)

    print(f"  Baseline: {len(baseline_df)} rows from {baseline_csv.name}")
    print(f"  LoRA:     {len(lora_df)} rows from {lora_csv.name}")

    merged = baseline_df.merge(
        lora_df,
        on=join_on,
        how="inner",
        suffixes=("_baseline", "_lora"),
    )

    if len(merged) == 0:
        raise ValueError(
            f"No matching rows after joining on '{join_on}'. "
            "Ensure both runs used the same fixed manifest and the same join key."
        )

    print(f"  Matched:  {len(merged)} rows")

    numeric_cols = (
        set(baseline_df.select_dtypes("number").columns)
        & set(lora_df.select_dtypes("number").columns)
    )
    numeric_cols.discard(join_on)

    comparison_rows: List[Dict] = []
    summary_metrics: Dict[str, Any] = {}

    for col in sorted(numeric_cols):
        b_col = f"{col}_baseline"
        l_col = f"{col}_lora"
        if b_col not in merged.columns or l_col not in merged.columns:
            continue

        b_vals = merged[b_col].dropna()
        l_vals = merged[l_col].dropna()
        if len(b_vals) == 0:
            continue

        b_mean = float(b_vals.mean())
        l_mean = float(l_vals.mean())
        delta = round(l_mean - b_mean, 6)
        direction = GENERATION_METRIC_DIRECTIONS.get(col, "neutral")
        improved: Optional[bool] = None
        if direction == "higher_is_better":
            improved = delta >= 0
        elif direction == "lower_is_better":
            improved = delta <= 0

        entry = {
            "metric": col,
            "baseline_mean": round(b_mean, 6),
            "lora_mean": round(l_mean, 6),
            "delta": delta,
            "direction": direction,
            "improved": improved,
        }
        comparison_rows.append(entry)
        summary_metrics[col] = {k: v for k, v in entry.items() if k != "metric"}

    # Reject reason distribution
    reject_distribution: Dict[str, Any] = {}
    for col in merged.columns:
        if "reject_reason" not in col.lower():
            continue
        suffix = "_baseline" if col.endswith("_baseline") else "_lora" if col.endswith("_lora") else None
        if suffix:
            key = suffix.lstrip("_")
            reject_distribution[key] = (
                merged[col].value_counts().to_dict()
            )

    # Write comparison CSV
    if comparison_rows:
        csv_path = output_dir / "baseline_vs_lora.csv"
        _write_csv(csv_path, comparison_rows)
        print(f"  ✓ Comparison CSV → {csv_path}")

    # Build and write summary JSON
    summary: Dict[str, Any] = {
        "baseline_csv": str(baseline_csv),
        "lora_csv": str(lora_csv),
        "matched_samples": len(merged),
        "metrics": summary_metrics,
        "reject_distribution": reject_distribution,
    }
    summary_path = output_dir / "baseline_vs_lora_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  ✓ Summary JSON    → {summary_path}")

    return summary


def print_comparison_table(summary: Dict[str, Any]) -> None:
    """Print a compact comparison table to stdout."""
    metrics = summary.get("metrics", {})
    if not metrics:
        print("No metrics in summary.")
        return
    print(
        f"\n{'Metric':<38} {'Baseline':>10} {'LoRA':>10} {'Delta':>10} {'Better?':>8}"
    )
    print("-" * 80)
    for name, vals in sorted(metrics.items()):
        improved = vals.get("improved")
        mark = "✓" if improved is True else "✗" if improved is False else "~"
        print(
            f"{name:<38} {vals['baseline_mean']:>10.4f} {vals['lora_mean']:>10.4f} "
            f"{vals['delta']:>+10.4f} {mark:>8}"
        )


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
