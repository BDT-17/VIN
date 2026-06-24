"""Dataset reporting.

Generates comprehensive reports for dataset releases.
"""

import json
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .metrics import (
    compare_downstream_benchmarks,
    load_metrics_json,
    summarize_release_metrics,
    write_flat_csv,
    write_json,
    write_markdown_report,
)


def generate_release_report(
    release_dir: Path,
    output_dir: Path,
    baseline_metrics_path: Optional[Path] = None,
    lora_metrics_path: Optional[Path] = None,
):
    """Generate comprehensive dataset report.

    Args:
        release_dir: Path to dataset release
        output_dir: Path to save report artifacts
        baseline_metrics_path: Optional detector metrics JSON for real-only baseline
        lora_metrics_path: Optional detector metrics JSON for real+LoRA benchmark
    """
    release_dir = Path(release_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GENERATING DATASET REPORT")
    print("=" * 60)
    print(f"Release: {release_dir.name}")
    print(f"Output: {output_dir}")
    print()

    # Load manifests
    samples_df = pd.read_parquet(release_dir / "samples.parquet")
    images_df = pd.read_parquet(release_dir / "images.parquet")
    groups_df = pd.read_parquet(release_dir / "groups.parquet")

    # Generate reports
    _generate_summary_report(samples_df, images_df, groups_df, output_dir)
    _generate_source_balance_report(samples_df, output_dir)
    _generate_split_report(samples_df, output_dir)
    _generate_quality_distribution_plot(samples_df, output_dir)
    _generate_size_distribution_plot(images_df, samples_df, output_dir)
    _generate_pr4_metrics_report(
        release_dir=release_dir,
        samples_df=samples_df,
        images_df=images_df,
        groups_df=groups_df,
        output_dir=output_dir,
        baseline_metrics_path=baseline_metrics_path,
        lora_metrics_path=lora_metrics_path,
    )

    print(f"\nâœ“ Report generated: {output_dir}")



def _generate_pr4_metrics_report(
    release_dir: Path,
    samples_df: pd.DataFrame,
    images_df: pd.DataFrame,
    groups_df: pd.DataFrame,
    output_dir: Path,
    baseline_metrics_path: Optional[Path] = None,
    lora_metrics_path: Optional[Path] = None,
):
    """Generate PR4 metrics, downstream benchmark, and markdown report artifacts."""
    release_name = release_dir.name
    summary = summarize_release_metrics(samples_df, images_df, groups_df)
    write_json(output_dir / "release_metrics.json", summary)
    write_flat_csv(output_dir / "release_metrics.csv", summary)

    benchmark = None
    if baseline_metrics_path and lora_metrics_path:
        benchmark = compare_downstream_benchmarks(
            load_metrics_json(baseline_metrics_path),
            load_metrics_json(lora_metrics_path),
        )
        write_json(output_dir / "downstream_benchmark.json", benchmark)
        write_flat_csv(output_dir / "downstream_benchmark.csv", benchmark)
        print("downstream_benchmark.json/csv")

    write_markdown_report(
        output_dir / "pr4_metrics_report.md",
        release_name=release_name,
        summary=summary,
        benchmark=benchmark,
    )
    print("release_metrics.json/csv")
    print("pr4_metrics_report.md")

def _generate_summary_report(
    samples_df: pd.DataFrame,
    images_df: pd.DataFrame,
    groups_df: pd.DataFrame,
    output_dir: Path,
):
    """Generate summary JSON report."""
    lora_samples = samples_df[samples_df['role'] == 'lora_positive']

    split_counts = lora_samples['split'].value_counts()
    source_counts = lora_samples['source_id'].value_counts()

    summary = {
        "total_samples": len(samples_df),
        "lora_positive_samples": len(lora_samples),
        "total_images": samples_df['image_id'].nunique(),
        "total_sources": samples_df['source_id'].nunique(),
        "splits": {
            "train": int(split_counts.get('train', 0)),
            "val": int(split_counts.get('val', 0)),
            "test": int(split_counts.get('test', 0)),
        },
        "sources": {str(k): int(v) for k, v in source_counts.items()},
        "quality": {
            "mean_quality_score": float(lora_samples['quality_score'].mean()),
            "mean_bbox_height_ratio": float(lora_samples['bbox_height_ratio'].mean()),
            "mean_visible_ratio": float(lora_samples['visible_ratio'].mean()),
        },
        "deduplication": {
            "unique": int((groups_df['dedupe_status'] == 'unique').sum()),
            "exact_duplicate": int((groups_df['dedupe_status'] == 'exact_duplicate').sum()),
            "near_duplicate": int((groups_df['dedupe_status'] == 'near_duplicate').sum()),
            "temporal_neighbor": int((groups_df['dedupe_status'] == 'temporal_neighbor').sum()),
        }
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("âœ“ summary.json")


def _generate_source_balance_report(samples_df: pd.DataFrame, output_dir: Path):
    """Generate source balance CSV."""
    lora_samples = samples_df[samples_df['role'] == 'lora_positive']

    balance = lora_samples.groupby(['source_id', 'split']).size().reset_index(name='count')
    balance = balance.pivot(index='source_id', columns='split', values='count').fillna(0)

    balance['total'] = balance.sum(axis=1)
    balance['train_pct'] = (balance.get('train', 0) / balance['total'] * 100).round(1)

    balance.to_csv(output_dir / "source_balance.csv")

    print("âœ“ source_balance.csv")


def _generate_split_report(samples_df: pd.DataFrame, output_dir: Path):
    """Generate split leakage report."""
    train_groups = set(samples_df[samples_df['split'] == 'train']['split_group_id'])
    val_groups = set(samples_df[samples_df['split'] == 'val']['split_group_id'])
    test_groups = set(samples_df[samples_df['split'] == 'test']['split_group_id'])

    cross_train_val = train_groups & val_groups
    cross_train_test = train_groups & test_groups
    cross_val_test = val_groups & test_groups

    report = {
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "test_groups": len(test_groups),
        "cross_train_val": len(cross_train_val),
        "cross_train_test": len(cross_train_test),
        "cross_val_test": len(cross_val_test),
        "zero_leakage": len(cross_train_val) == 0 and len(cross_train_test) == 0,
    }

    with open(output_dir / "split_leakage_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    print("âœ“ split_leakage_report.json")


def _generate_quality_distribution_plot(samples_df: pd.DataFrame, output_dir: Path):
    """Generate quality score distribution plot."""
    lora_samples = samples_df[samples_df['role'] == 'lora_positive']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Quality score
    axes[0, 0].hist(lora_samples['quality_score'], bins=30, edgecolor='black')
    axes[0, 0].set_xlabel('Quality Score')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Quality Score Distribution')

    # Bbox height ratio
    axes[0, 1].hist(lora_samples['bbox_height_ratio'], bins=30, edgecolor='black')
    axes[0, 1].set_xlabel('Bbox Height Ratio')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Bbox Height Ratio Distribution')

    # Visible ratio
    axes[1, 0].hist(lora_samples['visible_ratio'], bins=30, edgecolor='black')
    axes[1, 0].set_xlabel('Visible Ratio')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Visible Ratio Distribution')

    # Split distribution
    split_counts = lora_samples['split'].value_counts()
    axes[1, 1].bar(split_counts.index, split_counts.values)
    axes[1, 1].set_xlabel('Split')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Split Distribution')

    plt.tight_layout()
    plt.savefig(output_dir / "quality_distribution.png", dpi=150)
    plt.close()

    print("âœ“ quality_distribution.png")


def _generate_size_distribution_plot(
    images_df: pd.DataFrame,
    samples_df: pd.DataFrame,
    output_dir: Path,
):
    """Generate image and bbox size distribution plots."""
    # Merge to get bbox sizes
    merged = samples_df.merge(images_df[['image_id', 'width', 'height']], on='image_id')
    lora_samples = merged[merged['role'] == 'lora_positive']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Image size distribution
    axes[0].scatter(images_df['width'], images_df['height'], alpha=0.3, s=10)
    axes[0].set_xlabel('Width (px)')
    axes[0].set_ylabel('Height (px)')
    axes[0].set_title('Image Size Distribution')
    axes[0].grid(True, alpha=0.3)

    # Crop size distribution
    axes[1].scatter(lora_samples['crop_width'], lora_samples['crop_height'], alpha=0.3, s=10)
    axes[1].set_xlabel('Crop Width (px)')
    axes[1].set_ylabel('Crop Height (px)')
    axes[1].set_title('Crop Size Distribution')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "size_distribution.png", dpi=150)
    plt.close()

    print("âœ“ size_distribution.png")


