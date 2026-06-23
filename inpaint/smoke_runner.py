"""Full smoke runner for the standalone AI Replace flow.

Examples:
    python inpaint/smoke_runner.py --input-dir /kaggle/input/my-dataset --num-images 20 --git-pull
    python inpaint/smoke_runner.py --input-dir ./samples --num-images 5 --dry-run
    python inpaint/smoke_runner.py --kaggle-presets --num-images 3 --load-model
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

KAGGLE_DATASET_PRESETS = {
    "citypersons_bg_yolo": {
        "input_dir": Path("/kaggle/input/datasets/muttahirulislam/citypersons-dataset-with-bg-image/yolo_dir/yolo_dir"),
        "num_images": 3,
    },
    "cityperson_nguyena": {
        "input_dir": Path("/kaggle/input/datasets/nguyenaabcxyzeric/cityperson"),
        "num_images": 3,
    },
    "mot17_02_frcnn": {
        "input_dir": Path("/kaggle/input/datasets/kyoru4444/mot17-02-fcrnn/MOT17-02-FRCNN"),
        "num_images": 3,
    },
    "human_detection_dataset": {
        "input_dir": Path("/kaggle/input/datasets/constantinwerner/human-detection-dataset/human detection dataset/0"),
        "num_images": 3,
    },
}


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
    # Keep the insertion box large enough for a full body, but not so large that
    # SD3.5 fills the foreground with an oversized/cropped pedestrian.
    box_h = int(h * 0.42)
    box_w = int(box_h * 0.36)
    cx = int(w * 0.52)
    y2 = int(h * 0.90)
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
    failed_check_counts = Counter()
    for row in rows:
        checks = row.get("validation_checks") or {}
        if isinstance(checks, str):
            try:
                checks = json.loads(checks)
            except json.JSONDecodeError:
                checks = {}
        failed_check_counts.update(key for key, value in checks.items() if value is False)

    def mean(key: str) -> float:
        values = [float(row[key]) for row in rows if row.get(key) not in ("", None)]
        return round(float(np.mean(values)), 6) if values else 0.0

    summary = {
        "num_generated": total,
        "num_accepted": accepted,
        "accept_rate": round(accepted / max(1, total), 6),
        "reject_reason_counts": dict(reject_counts),
        "failed_validation_check_counts": dict(failed_check_counts),
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


def summarize_combined_rows(rows: list[dict], dataset_summaries: dict[str, dict], output_dir: Path) -> dict:
    summary = summarize_rows(rows, output_dir)
    summary["num_datasets"] = len(dataset_summaries)
    summary["datasets"] = dataset_summaries
    metrics_dir = output_dir / "metrics"
    (metrics_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (metrics_dir / "metrics_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset_preset", "num_generated", "num_accepted", "accept_rate", "ghost_reject_rate"],
        )
        writer.writeheader()
        for name, dataset_summary in dataset_summaries.items():
            writer.writerow(
                {
                    "dataset_preset": name,
                    "num_generated": dataset_summary.get("num_generated", 0),
                    "num_accepted": dataset_summary.get("num_accepted", 0),
                    "accept_rate": dataset_summary.get("accept_rate", 0.0),
                    "ghost_reject_rate": dataset_summary.get("ghost_reject_rate", 0.0),
                }
            )
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


def run_smoke(
    input_dir: Path,
    output_dir: Path,
    num_images: int = 20,
    seed: int = 42,
    load_model: bool = False,
    use_yolo: bool = True,
    pipe: AIReplacePipeline | None = None,
    yolo_segmenter=None,
):
    paths = ensure_dirs(output_dir)
    images = list_source_images(input_dir, limit=num_images)
    if not images:
        raise FileNotFoundError(f"No source images found under {input_dir}")
    pipe = pipe or (AIReplacePipeline.from_pretrained(device="cuda") if load_model else AIReplacePipeline(pipe=None, device="cpu"))
    yolo = yolo_segmenter if yolo_segmenter is not None else maybe_load_yolo(use_yolo)
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
        failed_checks = [key for key, value in row.get("validation_checks", {}).items() if value is False]
        print(
            f"[{index + 1}/{len(images)}] accepted={row['accepted']} "
            f"reject={row.get('reject_reason', '')} failed_checks={failed_checks} "
            f"inside={row.get('object_mask_inside_ratio')} area={row.get('object_area_ratio')} "
            f"source={image_path.name}"
        )
    write_manifest(rows, output_dir)
    summary = summarize_rows(rows, output_dir)
    print("AI Replace smoke summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return rows, summary


def run_all_kaggle_presets(
    output_dir: Path = Path("/kaggle/working/ai_replace_smoke"),
    num_images: int | None = None,
    seed: int = 42,
    load_model: bool = False,
    use_yolo: bool = True,
    presets: dict[str, dict] | None = None,
    fail_on_missing: bool = True,
):
    selected_presets = presets or KAGGLE_DATASET_PRESETS
    output_dir.mkdir(parents=True, exist_ok=True)
    pipe = AIReplacePipeline.from_pretrained(device="cuda") if load_model else AIReplacePipeline(pipe=None, device="cpu")
    yolo = maybe_load_yolo(use_yolo)
    combined_rows = []
    dataset_summaries = {}

    for preset_index, (name, preset) in enumerate(selected_presets.items()):
        input_dir = Path(preset["input_dir"])
        preset_limit = num_images if num_images is not None else int(preset.get("num_images", 20))
        dataset_output_dir = output_dir / name
        if not input_dir.exists():
            message = f"Kaggle preset path not found for {name}: {input_dir}"
            if fail_on_missing:
                raise FileNotFoundError(message)
            print(f"Skipping {message}")
            dataset_summaries[name] = {"skipped": True, "reason": "missing_input_dir", "input_dir": str(input_dir)}
            continue

        rows, summary = run_smoke(
            input_dir=input_dir,
            output_dir=dataset_output_dir,
            num_images=preset_limit,
            seed=seed + (preset_index * 1000),
            load_model=load_model,
            use_yolo=False,
            pipe=pipe,
            yolo_segmenter=yolo,
        )
        dataset_summaries[name] = {**summary, "input_dir": str(input_dir), "output_dir": str(dataset_output_dir)}
        for row in rows:
            combined_rows.append({**row, "dataset_preset": name, "dataset_output_dir": str(dataset_output_dir)})

    write_manifest(combined_rows, output_dir)
    combined_summary = summarize_combined_rows(combined_rows, dataset_summaries, output_dir)
    print("Combined AI Replace smoke summary:")
    print(json.dumps(combined_summary, indent=2, ensure_ascii=False))
    return combined_rows, combined_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full AI Replace smoke test.")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/ai_replace_smoke"))
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--git-pull", action="store_true")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--load-model", action="store_true", help="Load SD3.5 inpainting; omit for dry-run wiring test.")
    parser.add_argument("--no-yolo", action="store_true")
    parser.add_argument("--kaggle-presets", action="store_true", help="Run every built-in Kaggle dataset preset and emit combined artifacts.")
    parser.add_argument("--skip-missing", action="store_true", help="Skip missing Kaggle preset paths instead of failing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.git_pull:
        git_pull_ff_only(remote=args.remote, branch=args.branch)
    if args.kaggle_presets or args.input_dir is None:
        run_all_kaggle_presets(
            output_dir=args.output_dir,
            num_images=args.num_images,
            seed=args.seed,
            load_model=args.load_model,
            use_yolo=not args.no_yolo,
            fail_on_missing=not args.skip_missing,
        )
        return
    run_smoke(args.input_dir, args.output_dir, args.num_images, args.seed, args.load_model, not args.no_yolo)


if __name__ == "__main__":
    main()