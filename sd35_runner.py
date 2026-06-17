"""SD3.5 CityPersons augmentation: job orchestration, manifests, autotune, and exports."""

import csv
import gc
import json
from datetime import datetime
import math
import numpy as np
import os
import random
import re
import statistics
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps, ImageDraw, ImageFilter, ImageChops

try:
    import cv2
except ImportError:
    cv2 = None

from sd35_config import *
from sd35_data import *
from sd35_utils import *
from sd35_model import *
from sd35_evaluation import *
from sd35_pipeline import *

def variant_targets(variant):
    insertion = {
        "add_single_pedestrian": "single_pedestrian",
        "add_two_pedestrians": "two_pedestrians",
        "add_small_group": "small_group",
        "add_occluded_pedestrian": "occluded_pedestrian",
        "add_distant_pedestrian": "distant_pedestrian",
        "add_near_pedestrian": "near_pedestrian",
    }.get(variant, "pedestrian_insertion")
    return insertion, ""


def write_manifest(rows, output_dir=OUTPUT_DIR):
    manifest_path = Path(output_dir) / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split", "bucket", "source_context", "source_timeofday", "source_scene",
        "target_insertion", "target_timeofday", "original_path", "augmented_path",
        "comparison_path", "variant", "strength", "guidance_scale", "num_inference_steps",
        "generation_mode", "insert_bbox", "patch_bbox", "patch_debug_path",
        "expected_new_person_count", "detected_new_person_count",
        "expected_person_height", "detected_person_height", "scale_ratio_before_correction",
        "expected_height", "detected_height", "scale_ratio_before",
        "scale_corrected", "resized_person_height", "resized_person_width",
        "scale_correction_status", "seamless_clone_used", "fallback_alpha_paste",
        "fallback_alpha_used", "foreground_occlusion_used", "foreground_occluder_count",
        "foreground_occlusion_overlap_ratio", "foreground_occlusion_removed_ratio",
        "person_score", "scale_score", "background_score", "edge_score", "quality_score",
        "edge_contrast_score", "boundary_laplacian_score", "mask_feather_radius",
        "edge_harmonization_applied", "contact_shadow_applied", "edge_harmonization_debug_path",
        "retry_attempts", "last_reject_reason", "reject_reason",
        "seed", "source_path", "label_path", "output_path",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved manifest: {manifest_path}")
    return manifest_path


def build_augmentation_jobs(records, variants, target_per_bucket, target_splits):
    rng = random.Random(SEED)
    grouped = records_by_split_and_bucket(records)
    jobs = []
    job_index = 0
    for split in target_splits:
        for bucket in SCENE_BUCKETS:
            bucket_records = grouped.get(split, {}).get(bucket, [])
            if not bucket_records:
                print(f"SKIP {split}/{bucket}: no source images found")
                continue
            variant_weights = [AUGMENTATION_VARIANT_WEIGHTS.get(variant, 1.0) for variant in variants]
            print(f"Queued {target_per_bucket} images for {split}/{bucket} with EDA-aware variant weights")
            for bucket_index in range(1, target_per_bucket + 1):
                variant = rng.choices(list(variants), weights=variant_weights, k=1)[0]
                record = choose_record_for_variant(bucket_records, variant, rng)
                output_path = generated_image_path(OUTPUT_DIR, record, variant, bucket_index)
                comparison_path = comparison_image_path(OUTPUT_DIR, record, variant, bucket_index)
                jobs.append({
                    "job_index": job_index,
                    "record": record,
                    "variant": variant,
                    "output_path": output_path,
                    "comparison_path": comparison_path,
                    "seed": SEED + job_index,
                })
                job_index += 1
    return jobs

def run_augmentation_jobs_on_device(device, jobs, total_jobs, backend):
    if not jobs:
        return [], [], {}
    if str(device).startswith("cuda"):
        torch.cuda.set_device(torch.device(device).index or 0)
    print(f"[{device}] loading pipeline for {len(jobs)} jobs")
    if BACKGROUND_PRESERVATION_MODE == "context_person_composite" and CONTEXT_PERSON_GENERATION_PIPELINE == "img2img":
        pipe = build_img2img_pipeline(backend=backend, device=device)
    elif BACKGROUND_PRESERVATION_MODE in {"human_mask_inpaint", "bbox_inpaint", "context_person_composite"}:
        pipe = build_inpaint_pipeline(backend=backend, device=device)
    else:
        pipe = build_img2img_pipeline(backend=backend, device=device)
    outputs = []
    rows = []
    local_reject_reasons = {}
    for local_index, job in enumerate(jobs, 1):
        record = job["record"]
        target_insertion, target_timeofday = variant_targets(job["variant"])
        try:
            saved, generation_config = generate_variant_with_pipe(
                pipe=pipe,
                record=record,
                variant=job["variant"],
                output_path=job["output_path"],
                seed=job["seed"],
                device=device,
                strength=None,
                debug_index=job["job_index"],
            )
        except RuntimeError as exc:
            reason = normalize_reject_reason(exc)
            local_reject_reasons[reason] = local_reject_reasons.get(reason, 0) + 1
            print(f"[{device}] rejected {record.path.name} / {job['variant']}: {exc}")
            continue
        source_preview = resize_center_crop(load_source_image(record.path), resolution=RESOLUTION)
        augmented_preview = ImageOps.exif_transpose(Image.open(saved)).convert("RGB")
        comparison_title = (
            f"{job['variant']} | mode={generation_config['generation_mode']} | strength={generation_config['strength']} | "
            f"guidance={generation_config['guidance_scale']} | "
            f"steps={generation_config['num_inference_steps']} | seed={job['seed']}"
        )
        comparison_saved = save_comparison_pair(
            source_preview,
            augmented_preview,
            job["comparison_path"],
            comparison_title,
        )
        outputs.append(saved)
        rows.append({
            "split": record.split,
            "bucket": record.bucket,
            "source_context": record.scene or "urban",
            "source_timeofday": record.timeofday or "",
            "source_scene": record.scene or "",
            "target_insertion": target_insertion,
            "target_timeofday": target_timeofday,
            "original_path": str(record.path),
            "augmented_path": str(saved),
            "comparison_path": str(comparison_saved),
            "variant": job["variant"],
            "strength": generation_config["strength"],
            "guidance_scale": generation_config["guidance_scale"],
            "num_inference_steps": generation_config["num_inference_steps"],
            "generation_mode": generation_config["generation_mode"],
            "insert_bbox": json.dumps(generation_config["insert_bbox"]),
            "patch_bbox": json.dumps(generation_config["patch_bbox"]),
            "patch_debug_path": generation_config["patch_debug_path"],
            "expected_new_person_count": generation_config["expected_new_person_count"],
            "detected_new_person_count": generation_config["detected_new_person_count"],
            "expected_person_height": generation_config.get("expected_person_height", ""),
            "detected_person_height": generation_config.get("detected_person_height", ""),
            "scale_ratio_before_correction": generation_config.get("scale_ratio_before_correction", ""),
            "expected_height": generation_config.get("expected_height", generation_config.get("expected_person_height", "")),
            "detected_height": generation_config.get("detected_height", generation_config.get("detected_person_height", "")),
            "scale_ratio_before": generation_config.get("scale_ratio_before", generation_config.get("scale_ratio_before_correction", "")),
            "scale_corrected": generation_config.get("scale_corrected", False),
            "resized_person_height": generation_config.get("resized_person_height", ""),
            "resized_person_width": generation_config.get("resized_person_width", ""),
            "scale_correction_status": generation_config.get("scale_correction_status", "none"),
            "seamless_clone_used": generation_config.get("seamless_clone_used", False),
            "fallback_alpha_paste": generation_config.get("fallback_alpha_paste", False),
            "fallback_alpha_used": generation_config.get("fallback_alpha_used", generation_config.get("fallback_alpha_paste", False)),
            "foreground_occlusion_used": generation_config.get("foreground_occlusion_used", False),
            "foreground_occluder_count": generation_config.get("foreground_occluder_count", 0),
            "foreground_occlusion_overlap_ratio": generation_config.get("foreground_occlusion_overlap_ratio", 0.0),
            "foreground_occlusion_removed_ratio": generation_config.get("foreground_occlusion_removed_ratio", 0.0),
            "person_score": generation_config.get("person_score", ""),
            "scale_score": generation_config.get("scale_score", ""),
            "background_score": generation_config.get("background_score", ""),
            "edge_score": generation_config.get("edge_score", ""),
            "quality_score": generation_config.get("quality_score", ""),
            "edge_contrast_score": generation_config.get("edge_contrast_score", ""),
            "boundary_laplacian_score": generation_config.get("boundary_laplacian_score", ""),
            "mask_feather_radius": generation_config.get("mask_feather_radius", ""),
            "edge_harmonization_applied": generation_config.get("edge_harmonization_applied", False),
            "contact_shadow_applied": generation_config.get("contact_shadow_applied", False),
            "edge_harmonization_debug_path": generation_config.get("edge_harmonization_debug_path", ""),
            "retry_attempts": generation_config.get("retry_attempts", 0),
            "last_reject_reason": generation_config.get("last_reject_reason", ""),
            "reject_reason": generation_config.get("reject_reason", generation_config.get("last_reject_reason", "")),
            "seed": job["seed"],
            "source_path": str(record.path),
            "label_path": str(record.label_path) if record.label_path else "",
            "output_path": str(saved),
        })
        completed = job["job_index"] + 1
        if completed == 1 or completed % 25 == 0 or local_index == len(jobs):
            print(f"[{device}] [{completed}/{total_jobs}] saved {saved.name}")
    del pipe
    clear_cuda()
    return outputs, rows, local_reject_reasons


def augment_dataset(records, variants=AUGMENTATION_VARIANTS, backend=MODEL_BACKEND, target_per_bucket=AUGMENTATIONS_PER_BUCKET, target_splits=TARGET_SPLITS, write_manifest_file=True, return_manifest_rows=False):
    global LAST_MANIFEST_ROWS, LAST_REJECT_HISTOGRAM, LAST_AUGMENTATION_SUMMARY
    if not records:
        raise FileNotFoundError("No images found. Mount dataset folder and rerun scan_dataset().")
    devices = resolve_augmentation_devices()
    jobs = build_augmentation_jobs(records, variants, target_per_bucket, target_splits)
    if not jobs:
        print("No augmentation jobs were queued.")
        LAST_MANIFEST_ROWS = []
        LAST_REJECT_HISTOGRAM = {}
        LAST_AUGMENTATION_SUMMARY = {
            "total_jobs": 0,
            "accepted": 0,
            "rejected": 0,
            "accept_rate": 0.0,
            "reject_histogram": {},
        }
        return ([], []) if return_manifest_rows else []
    total_jobs = len(jobs)
    print(f"Using augmentation devices: {devices}")
    shards = [jobs[index::len(devices)] for index in range(len(devices))]
    all_outputs = []
    manifest_rows = []
    reject_histogram = {}
    active_shards = [(device, shard) for device, shard in zip(devices, shards) if shard]
    shard_summary = ", ".join(f"{device}:{len(shard)}" for device, shard in active_shards)
    print(f"Device job split: {shard_summary}")

    if len(active_shards) == 1:
        device, shard = active_shards[0]
        device_results = [run_augmentation_jobs_on_device(device, shard, total_jobs, backend)]
    else:
        max_workers = len(active_shards)
        print(f"Running {max_workers} augmentation shards in parallel.")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_device = {
                executor.submit(run_augmentation_jobs_on_device, device, shard, total_jobs, backend): device
                for device, shard in active_shards
            }
            device_results = []
            for future in as_completed(future_to_device):
                device = future_to_device[future]
                try:
                    device_results.append(future.result())
                except Exception as exc:
                    print(f"[{device}] shard failed: {type(exc).__name__}: {exc}")
                    raise

    for outputs, rows, device_rejects in device_results:
        all_outputs.extend(outputs)
        manifest_rows.extend(rows)
        for reason, count in device_rejects.items():
            reject_histogram[reason] = reject_histogram.get(reason, 0) + count
    manifest_rows = sorted(manifest_rows, key=lambda row: row["seed"])
    all_outputs = [Path(row["output_path"]) for row in manifest_rows]
    if write_manifest_file:
        write_manifest(manifest_rows, OUTPUT_DIR)
    accepted = len(all_outputs)
    rejected = sum(reject_histogram.values())
    retried = sum(int(row.get("retry_attempts") or 0) for row in manifest_rows)
    scale_corrected_count = sum(1 for row in manifest_rows if str(row.get("scale_corrected")).lower() == "true")
    seamless_clone_used_count = sum(1 for row in manifest_rows if str(row.get("seamless_clone_used")).lower() == "true")
    fallback_alpha_paste_count = sum(1 for row in manifest_rows if str(row.get("fallback_alpha_paste")).lower() == "true")
    print(f"Generated {len(all_outputs)} images in {OUTPUT_DIR}")
    print("Smoke/evaluation counters:")
    print(f"  total attempts: {total_jobs}")
    print(f"  accepted: {accepted}")
    print(f"  rejected: {rejected}")
    print(f"  retried: {retried}")
    print(f"  scale_corrected_count: {scale_corrected_count}")
    print(f"  seamless_clone_used_count: {seamless_clone_used_count}")
    print(f"  fallback_alpha_paste_count: {fallback_alpha_paste_count}")
    print(f"  reject reasons histogram: {reject_histogram}")
    if total_jobs:
        estimated_before_scale_correction_accepts = max(0, accepted - scale_corrected_count)
        print(f"  accept rate before scale correction (estimated): {estimated_before_scale_correction_accepts / total_jobs:.3f}")
        print(f"  accept rate after scale correction: {accepted / total_jobs:.3f}")
        if accepted:
            print(f"  scale correction share of accepted: {scale_corrected_count / accepted:.3f}")
            print(f"  seamless clone success rate: {seamless_clone_used_count / accepted:.3f}")
    LAST_MANIFEST_ROWS = manifest_rows
    LAST_REJECT_HISTOGRAM = reject_histogram
    LAST_AUGMENTATION_SUMMARY = {
        "total_jobs": total_jobs,
        "accepted": accepted,
        "rejected": rejected,
        "accept_rate": accepted / total_jobs if total_jobs else 0.0,
        "reject_histogram": reject_histogram,
    }
    if return_manifest_rows:
        return all_outputs, manifest_rows
    return all_outputs


## 10.10 Quality-Guided Autotune

LAST_AUTOTUNE_RECOMMENDATIONS = {}


def summarize_quality_rows(rows):
    rows = rows or []
    accepted_rows = [row for row in rows if float(row.get("quality_score", 0.0) or 0.0) > 0.0]
    if not accepted_rows:
        return {
            "count": 0,
            "person_mean": 0.0,
            "scale_mean": 0.0,
            "background_mean": 0.0,
            "edge_mean": 0.0,
            "quality_mean": 0.0,
            "person_p10": 0.0,
            "scale_p10": 0.0,
            "background_p10": 0.0,
            "edge_p10": 0.0,
            "quality_p10": 0.0,
        }

    def values(key):
        return np.array([float(row.get(key, 0.0) or 0.0) for row in accepted_rows], dtype=np.float32)

    summary = {"count": len(accepted_rows)}
    for key in ("person_score", "scale_score", "background_score", "edge_score", "quality_score"):
        arr = values(key)
        short = key.replace("_score", "")
        summary[f"{short}_mean"] = round(float(arr.mean()), 4)
        summary[f"{short}_p10"] = round(float(np.percentile(arr, 10)), 4)
    return summary


def _bounded_value(name, proposed):
    if name not in EFFECTIVE_CONFIG:
        return proposed
    current = EFFECTIVE_CONFIG[name]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        return proposed
    if current == 0:
        return proposed
    max_ratio = float(AUTOTUNE_SETTINGS.get("max_adjustment_ratio", 1.35))
    lower = current / max_ratio
    upper = current * max_ratio
    bounded = min(max(float(proposed), lower), upper)
    if isinstance(current, int):
        return int(round(bounded))
    return round(bounded, 4)


def _blend_value(name, proposed):
    current = EFFECTIVE_CONFIG.get(name)
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        return proposed
    aggressiveness = float(AUTOTUNE_SETTINGS.get("aggressiveness", 0.60))
    blended = current + (float(proposed) - float(current)) * aggressiveness
    return _bounded_value(name, blended)


def _format_delta(old, new):
    if isinstance(old, (int, float)) and old != 0:
        return round((float(new) - float(old)) / abs(float(old)), 4)
    if isinstance(old, (int, float)):
        return round(float(new) - float(old), 4)
    return None


def _build_change_report(overrides, generation_delta=None):
    changes = {}
    for name, new_value in overrides.items():
        old_value = EFFECTIVE_CONFIG.get(name)
        changes[name] = {
            "old": old_value,
            "new": new_value,
            "delta": _format_delta(old_value, new_value),
        }
    if generation_delta:
        changes["generation_delta"] = {
            "old": None,
            "new": generation_delta,
            "delta": None,
        }
    return changes


def recommend_parameter_updates(rows=None, reject_histogram=None, summary=None):
    rows = LAST_MANIFEST_ROWS if rows is None else rows
    reject_histogram = LAST_REJECT_HISTOGRAM if reject_histogram is None else reject_histogram
    summary = LAST_AUGMENTATION_SUMMARY if summary is None else summary
    quality = summarize_quality_rows(rows)
    total_jobs = int((summary or {}).get("total_jobs", len(rows or [])))
    accepted = int((summary or {}).get("accepted", quality["count"]))
    accept_rate = accepted / max(1, total_jobs)
    rejects = reject_histogram or {}
    recommended = {}
    generation_delta = None

    if quality["count"] < AUTOTUNE_SETTINGS["min_samples"]:
        return {
            "quality_summary": quality,
            "accept_rate": round(accept_rate, 4),
            "reject_histogram": rejects,
            "recommended_overrides": {},
            "changes": {},
            "note": "Not enough accepted samples for stable autotune.",
        }

    if accept_rate < 0.30:
        recommended["CONTEXT_GENERATION_RETRIES"] = min(
            AUTOTUNE_SETTINGS["max_retry_budget"],
            int(CONTEXT_GENERATION_RETRIES) + 1,
        )

    low_person_rejects = sum(rejects.get(reason, 0) for reason in ("low_person_conf", "no_person_detected", "no_person_mask", "segmenter_failed"))
    if low_person_rejects >= max(2, total_jobs * 0.15) or quality["person_p10"] < 0.45:
        recommended["CONTEXT_PERSON_MIN_CONFIDENCE"] = max(0.08, CONTEXT_PERSON_MIN_CONFIDENCE - 0.02)
        recommended["MIN_RETRY_PERSON_CONFIDENCE"] = max(0.08, MIN_RETRY_PERSON_CONFIDENCE - 0.02)

    if rejects.get("bad_scale", 0) >= max(2, total_jobs * 0.12) or quality["scale_p10"] < 0.55:
        recommended["SCALE_CORRECTION_MAX_ATTEMPTS"] = min(4, int(SCALE_CORRECTION_MAX_ATTEMPTS) + 1)
        recommended["SCALE_CORRECTION_HEIGHT_STEP"] = min(0.12, SCALE_CORRECTION_HEIGHT_STEP + 0.02)

    if rejects.get("bad_person_depth_overlap", 0) >= max(1, total_jobs * 0.08):
        recommended["MAX_PERSON_PERSON_OVERLAP_RATIO"] = max(0.04, MAX_PERSON_PERSON_OVERLAP_RATIO - 0.02)
        recommended["OCCLUDED_PERSON_MAX_HEIGHT_RATIO"] = max(0.55, OCCLUDED_PERSON_MAX_HEIGHT_RATIO - 0.05)

    if quality["background_p10"] < 0.55:
        recommended["SEAMLESS_CLONE_MIN_MASK_AREA"] = max(0.003, SEAMLESS_CLONE_MIN_MASK_AREA - 0.001)
        recommended["LOCAL_BRIGHTNESS_STRENGTH"] = min(0.45, LOCAL_BRIGHTNESS_STRENGTH + 0.04)
        recommended["PERSON_COLOR_MATCH_STRENGTH"] = min(0.45, PERSON_COLOR_MATCH_STRENGTH + 0.04)

    if quality["edge_p10"] < 0.55:
        recommended["EDGE_HALO_COLOR_MATCH_STRENGTH"] = min(0.48, EDGE_HALO_COLOR_MATCH_STRENGTH + 0.04)
        recommended["PERSON_PASTE_FEATHER_RADIUS"] = min(0.60, PERSON_PASTE_FEATHER_RADIUS + 0.08)

    if quality["quality_mean"] < AUTOTUNE_SETTINGS["target_quality_score"]:
        generation_delta = {"strength": 0.01, "guidance": 0.10, "steps": 0}

    bounded = {name: _blend_value(name, value) for name, value in recommended.items()}
    changes = _build_change_report(bounded, generation_delta)

    return {
        "quality_summary": quality,
        "accept_rate": round(accept_rate, 4),
        "reject_histogram": rejects,
        "recommended_overrides": bounded,
        "generation_delta": generation_delta,
        "changes": changes,
    }



def json_safe_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)

def save_autotune_snapshot(report, before_config, after_config):
    if not AUTOTUNE_SETTINGS.get("save_snapshot", True):
        return None
    snapshot_dir = Path(AUTOTUNE_SETTINGS.get("snapshot_dir", "autotune_snapshots"))
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = snapshot_dir / f"autotune_snapshot_{timestamp}.json"
    payload = json_safe_value({
        "timestamp": timestamp,
        "run_preset": RUN_PRESET,
        "before_config": before_config,
        "after_config": after_config,
        "report": report,
    })
    snapshot_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return snapshot_path


def print_autotune_report(report):
    print("=== AUTOTUNE REPORT ===")
    print("Quality summary:", report.get("quality_summary"))
    print("Accept rate:", report.get("accept_rate"))
    print("Reject histogram:", report.get("reject_histogram"))
    note = report.get("note")
    if note:
        print("Note:", note)
    changes = report.get("changes", {})
    if not changes:
        print("No parameter changes recommended.")
        return
    for name, change in changes.items():
        old = change["old"]
        new = change["new"]
        delta = change["delta"]
        if isinstance(old, (int, float)) and isinstance(new, (int, float)) and delta is not None:
            print(f"  {name:35s}: {old:.4g} -> {new:.4g} ({delta:+.1%})")
        else:
            print(f"  {name:35s}: {old} -> {new}")



def _refresh_variant_generation_maps():
    global AUGMENTATION_VARIANT_WEIGHTS, VARIANT_STRENGTHS, VARIANT_GUIDANCE_SCALES, VARIANT_NUM_INFERENCE_STEPS
    AUGMENTATION_VARIANT_WEIGHTS = {name: cfg["weight"] for name, cfg in VARIANT_PROFILE.items()}
    VARIANT_STRENGTHS = {name: cfg["strength"] for name, cfg in VARIANT_PROFILE.items()}
    VARIANT_GUIDANCE_SCALES = {name: cfg["guidance"] for name, cfg in VARIANT_PROFILE.items()}
    VARIANT_NUM_INFERENCE_STEPS = {name: cfg["steps"] for name, cfg in VARIANT_PROFILE.items()}


def _bump_variant_generation(strength_delta=0.0, guidance_delta=0.0, step_delta=0):
    for profile in VARIANT_PROFILE.values():
        profile["strength"] = round(float(np.clip(profile["strength"] + strength_delta, 0.55, 0.86)), 4)
        profile["guidance"] = round(float(np.clip(profile["guidance"] + guidance_delta, 5.5, 8.5)), 4)
        profile["steps"] = int(np.clip(int(profile["steps"] + step_delta), 24, 48))
    _refresh_variant_generation_maps()

def apply_parameter_updates(report):
    updates = dict((report or {}).get("recommended_overrides", {}))
    generation_delta = (report or {}).get("generation_delta")
    if generation_delta:
        _bump_variant_generation(
            strength_delta=generation_delta.get("strength", 0.0),
            guidance_delta=generation_delta.get("guidance", 0.0),
            step_delta=generation_delta.get("steps", 0),
        )
    for name, value in updates.items():
        globals()[name] = value
        EFFECTIVE_CONFIG[name] = value
    LAST_AUTOTUNE_RECOMMENDATIONS.clear()
    LAST_AUTOTUNE_RECOMMENDATIONS.update(updates)
    if generation_delta:
        LAST_AUTOTUNE_RECOMMENDATIONS["generation_delta"] = generation_delta
    return dict(LAST_AUTOTUNE_RECOMMENDATIONS)


def autotune_from_last_run(apply=True, dry_run=True, save_snapshot=None):
    before_config = dict(EFFECTIVE_CONFIG)
    report = recommend_parameter_updates()
    if AUTOTUNE_SETTINGS.get("print_report", True):
        print_autotune_report(report)
    should_apply = bool(apply and not dry_run and AUTOTUNE_SETTINGS.get("enabled", True))
    if should_apply:
        applied = apply_parameter_updates(report)
        print("Applied runtime parameter updates:", applied)
    else:
        print("Dry run only; no runtime parameters were changed.")
    after_config = dict(EFFECTIVE_CONFIG)
    if save_snapshot is None:
        save_snapshot = AUTOTUNE_SETTINGS.get("save_snapshot", True)
    if save_snapshot:
        snapshot_path = save_autotune_snapshot(report, before_config, after_config)
        if snapshot_path:
            print("Saved autotune snapshot:", snapshot_path)
    return report

## 10.11 Reset Runtime Config

def reset_runtime_config():
    global EFFECTIVE_CONFIG, VARIANT_PROFILE
    EFFECTIVE_CONFIG = dict(BASE_EFFECTIVE_CONFIG)
    globals().update(EFFECTIVE_CONFIG)
    VARIANT_PROFILE.clear()
    VARIANT_PROFILE.update({name: dict(profile) for name, profile in BASE_VARIANT_PROFILE.items()})
    _refresh_variant_generation_maps()
    if "LAST_AUTOTUNE_RECOMMENDATIONS" in globals():
        LAST_AUTOTUNE_RECOMMENDATIONS.clear()
    print("Runtime config reset to notebook defaults.")
    return dict(EFFECTIVE_CONFIG)


# Run this cell, then call reset_runtime_config() whenever you want to undo runtime autotune changes.

def run_smoke(records, smoke_images=10, smoke_splits=None):
    smoke_splits = smoke_splits or TARGET_SPLITS
    generated_paths, manifest_rows = augment_dataset(
        records,
        variants=AUGMENTATION_VARIANTS,
        target_per_bucket=smoke_images,
        target_splits=smoke_splits,
        return_manifest_rows=True,
    )
    autotune_report = autotune_from_last_run(apply=True, dry_run=True)
    return generated_paths, manifest_rows, autotune_report


def export_outputs():
    import shutil
    export_base = Path("/kaggle/working/sd35_citypersons_augmented_export")
    if export_base.with_suffix(".zip").exists():
        export_base.with_suffix(".zip").unlink()
    shutil.make_archive(str(export_base), "zip", str(OUTPUT_DIR))
    print(f"Saved export: {export_base.with_suffix('.zip')}")
    return export_base.with_suffix(".zip")


if __name__ == "__main__":
    ensure_output_dirs()
    records = load_records()
    run_smoke(records)
