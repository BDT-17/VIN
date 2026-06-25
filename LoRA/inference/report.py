"""Paired baseline-vs-LoRA report for the SD3.5 inpaint evaluation.

Reads per-case metrics for both conditions (same case_id + seed), computes
per-metric deltas (LoRA - baseline), writes:
    metrics_per_case.csv
    metrics_summary.json
    paired_comparison.csv
No single fused quality score — component metrics only.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List

from .inpaint_metrics import METRIC_DIRECTIONS

_PAIRED = ["person_confidence", "person_inside_mask_ratio", "scale_error",
           "outside_mask_mae", "outside_mask_ssim", "edge_seam_score"]


def _mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 4) if vals else None


def write_per_case_csv(rows: List[Dict], out_path: Path) -> Path:
    out_path = Path(out_path)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return out_path
    fields = sorted({k for r in rows for k in r.keys()})
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return out_path


def build_paired_comparison(baseline_rows: List[Dict], lora_rows: List[Dict],
                            out_dir: Path) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bl = {(r["case_id"], r["seed"]): r for r in baseline_rows}
    lo = {(r["case_id"], r["seed"]): r for r in lora_rows}
    keys = sorted(set(bl) & set(lo))

    paired_rows = []
    for key in keys:
        b, l = bl[key], lo[key]
        row = {"case_id": key[0], "seed": key[1]}
        for m in _PAIRED:
            bv, lv = b.get(m), l.get(m)
            if isinstance(bv, (int, float)) and isinstance(lv, (int, float)):
                row[f"delta_{m}"] = round(lv - bv, 4)
            else:
                row[f"delta_{m}"] = None
        paired_rows.append(row)

    paired_path = out_dir / "paired_comparison.csv"
    write_per_case_csv(paired_rows, paired_path)

    # summary: per-metric baseline mean, lora mean, mean delta, direction
    summary = {"matched_pairs": len(keys), "metrics": {}}
    all_metrics = sorted(set(METRIC_DIRECTIONS) &
                         ({k for r in baseline_rows for k in r} | {k for r in lora_rows for k in r}))
    for m in all_metrics:
        b_mean = _mean([bl[k].get(m) for k in keys])
        l_mean = _mean([lo[k].get(m) for k in keys])
        delta = round(l_mean - b_mean, 4) if (b_mean is not None and l_mean is not None) else None
        direction = METRIC_DIRECTIONS.get(m, "neutral")
        improved = None
        if delta is not None and direction == "higher_is_better":
            improved = delta >= 0
        elif delta is not None and direction == "lower_is_better":
            improved = delta <= 0
        summary["metrics"][m] = {"baseline_mean": b_mean, "lora_mean": l_mean,
                                 "delta": delta, "direction": direction, "improved": improved}

    (out_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def print_summary(summary: Dict) -> None:
    print(f"\nMatched pairs: {summary.get('matched_pairs', 0)}")
    print(f"{'metric':<28}{'baseline':>11}{'lora':>11}{'delta':>11}{'better?':>9}")
    print("-" * 70)
    for name, v in summary.get("metrics", {}).items():
        mark = "~" if v["improved"] is None else ("yes" if v["improved"] else "no")
        b = "n/a" if v["baseline_mean"] is None else f"{v['baseline_mean']:.4f}"
        l = "n/a" if v["lora_mean"] is None else f"{v['lora_mean']:.4f}"
        d = "n/a" if v["delta"] is None else f"{v['delta']:+.4f}"
        print(f"{name:<28}{b:>11}{l:>11}{d:>11}{mark:>9}")
