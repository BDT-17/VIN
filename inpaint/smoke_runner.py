"""Full smoke runner for the standalone AI Replace flow.

Examples:
    python inpaint/smoke_runner.py --input-dir /kaggle/input/my-dataset --num-images 20 --git-pull
    python inpaint/smoke_runner.py --input-dir ./samples --num-images 5 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inpaint.sd35_ai_replace import AIReplacePipeline

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def git_pull_ff_only(repo_root: Path = ROOT, remote: str = "origin", branch: str = "main") -> None:
    subprocess.run(["git", "fetch", remote, branch], cwd=repo_root, check=True)
    subprocess.run(["git", "pull", "--ff-only", remote, branch], cwd=repo_root, check=True)


def list_source_images(input_dir: Path, limit: int | None = None) -> list[Path]:
    paths = [
        path for path in sorted(input_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    return paths[:limit] if limit else paths


def default_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    w, h = image.size
    box_h = int(h * 0.38)
    box_w = int(box_h * 0.36)
    cx = int(w * 0.52)
    y2 = int(h * 0.88)
    return (max(0, cx - box_w // 2), max(0, y2 - box_h), min(w, cx + box_w // 2), min(h, y2))


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {"root": output_dir, "previews": output_dir / "previews", "metrics": output_dir / "metrics"}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_diff_outside_mask(original: Image.Image, composite: Image.Image, mask, path: Path) -> None:
    diff = ImageChops.difference(original.convert("RGB"), composite.convert("RGB").resize(original.size))
    mask_img = mask.to_pil().resize(original.size, Image.NEAREST)
    outside = ImageChops.invert(mask_img)
    black = Image.new("RGB", original.size, 0)
    Image.composite(diff, black, outside).save(path)


def save_object_mask(mask_array: np.ndarray, path: Path) -> None:
    Image.fromarray((mask_array > 0.5).astype(np.uint8) * 255, mode="L").save(path)


def write_manifest(rows: list[dict], output_dir: Path) -> Path:
    jsonl_path = output_dir / "manifest.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    csv_path = output_dir / "manifest.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return jsonl_path


def summarize_rows(rows: list[dict], output_dir: Path) -> dict:
    total = len(rows)
    accepted = sum(1 for row in rows if row.get("accepted") is True)
    reject_counts = Counter(row.get("reject_reason") or "" for row in rows if row.get("accepted") is not True)

    def mean(key: str) -> float:
        values = [float(row[key]) for row in rows if row.get(key) not in ("", None)]
        return round(float(np.mean(values)), 6) if values else 0.0

    summary = {
        "num_generated": total,
        "num_accepted": accepted,
        "accept_rate": round(accepted / max(1, total), 6),
        "reject_reason_counts": dict(reject_counts),
        "mean_outside_mask_diff": mean("outside_mask_diff"),
        "mean_object_inside_ratio": mean("object_mask_inside_ratio"),
        "mean_background_preservation_score": mean("background_preservation_score"),
        "mean_harmonization_score": mean("harmonization_score"),
        "mean_opacity_score": mean("opacity_score"),
        "mean_detector_conf_drop": mean("detector_conf_drop"),
        "ghost_reject_rate": round(sum(1 for row in rows if str(row.get("reject_reason", "")).startswith("ghost")) / max(1, total), 6),
    }
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (metrics_dir / "metrics_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    return summary


def maybe_load_yolo(enabled: bool):
    if not enabled:
        return None
    try:
        from ultralytics import YOLO
        return YOLO("yolov8m-seg.pt")
    except Exception as exc:
        print(f"YOLO unavailable; continuing without detector: {type(exc).__name__}: {exc}")
        return None


def run_smoke(input_dir: Path, output_dir: Path, num_images: int = 20, seed: int = 42, load_model: bool = False, use_yolo: bool = True):
    paths = ensure_dirs(output_dir)
    images = list_source_images(input_dir, limit=num_images)
    if not images:
        raise FileNotFoundError(f"No source images found under {input_dir}")
    pipe = AIReplacePipeline.from_pretrained(device="cuda") if load_model else AIReplacePipeline(pipe=None, device="cpu")
    yolo = maybe_load_yolo(use_yolo)
    rows = []
    for index, image_path in enumerate(images):
        image = Image.open(image_path).convert("RGB")
        bbox = default_bbox(image)
        result = pipe.run(image, bbox=bbox, seed=seed + index, yolo_segmenter=yolo)
        stem = f"{index:04d}_{image_path.stem}"
        resized_original = image.resize(result.harmonized_image.size)
        resized_original.save(paths["previews"] / f"{stem}_original.png")
        result.mask_bundle.to_pil().save(paths["previews"] / f"{stem}_mask.png")
        result.raw_image.save(paths["previews"] / f"{stem}_generated_raw.png")
        if result.object_result.object_mask is not None:
            save_object_mask(result.object_result.object_mask, paths["previews"] / f"{stem}_object_mask.png")
        result.harmonized_image.save(paths["previews"] / f"{stem}_harmonized.png")
        save_diff_outside_mask(resized_original, result.harmonized_image, result.mask_bundle, paths["previews"] / f"{stem}_diff_outside_mask.png")
        row = dict(result.manifest_row)
        row.update({"source_path": str(image_path), "output_path": str(paths["previews"] / f"{stem}_harmonized.png"), "preview_stem": stem})
        rows.append(row)
        print(f"[{index + 1}/{len(images)}] accepted={row['accepted']} reject={row.get('reject_reason', '')} source={image_path.name}")
    write_manifest(rows, output_dir)
    summary = summarize_rows(rows, output_dir)
    print("AI Replace smoke summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full AI Replace smoke test.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/ai_replace_smoke"))
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--git-pull", action="store_true")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--load-model", action="store_true", help="Load SD2 inpainting; omit for dry-run wiring test.")
    parser.add_argument("--no-yolo", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.git_pull:
        git_pull_ff_only(remote=args.remote, branch=args.branch)
    run_smoke(args.input_dir, args.output_dir, args.num_images, args.seed, args.load_model, not args.no_yolo)


if __name__ == "__main__":
    main()