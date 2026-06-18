"""SD3.5 CityPersons augmentation: data scanning and previews."""

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

@dataclass
class ImageRecord:
    path: Path
    split: str
    bucket: str
    caption: str
    label_path: Optional[Path] = None
    weather: Optional[str] = None
    timeofday: Optional[str] = None
    scene: Optional[str] = None


def load_caption_map(csv_path):
    if not csv_path:
        return {}
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"Caption CSV not found: {csv_path}. Using default captions.")
        return {}
    caption_map = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            file_name = row.get("file_name") or row.get("filename") or row.get("image")
            caption = row.get("caption") or row.get("prompt")
            if file_name and caption:
                caption_map[Path(file_name).name] = caption
    return caption_map


def infer_split(path):
    parts = [part.lower() for part in Path(path).parts]
    if "train" in parts:
        return "train"
    if "valid" in parts or "val" in parts:
        return "val"
    if "test" in parts:
        return "test"
    return "train"


def find_label_path(image_path, label_dir=None):
    image_path = Path(image_path)
    split = infer_split(image_path)
    split_label_dir = LABEL_SPLIT_DIRS.get(split)
    sibling_label_dir = image_path.parent.parent / "labels" if image_path.parent.name == "images" else None
    candidates = [
        image_path.with_suffix(".json"),
        image_path.parent / f"{image_path.stem}.json",
        image_path.parent / f"{image_path.stem}.txt",
        sibling_label_dir / f"{image_path.stem}.txt" if sibling_label_dir else None,
        sibling_label_dir / f"{image_path.stem}.json" if sibling_label_dir else None,
        split_label_dir / f"{image_path.stem}.txt" if split_label_dir else None,
        split_label_dir / f"{image_path.stem}.json" if split_label_dir else None,
        image_path.parent / f"{image_path.stem}_gtBboxCityPersons.json",
        image_path.parent / f"{image_path.stem}_gtFine_polygons.json",
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


def build_caption(path, bucket, caption_map, metadata=None, include_weather=True):
    if Path(path).name in caption_map:
        return caption_map[Path(path).name]
    return BASE_CAPTION


def build_generation_prompt(record, variant):
    return PRESERVATION_PROMPT


def build_variant_negative_prompt(variant):
    return NEGATIVE_PROMPT


def is_source_image(path):
    path = Path(path)
    if path.suffix.lower() not in IMAGE_EXTS:
        return False
    name = path.name.lower()
    if any(token in name for token in ["mask", "label", "gtfine", "gtbbox", "instance", "polygon", "color"]):
        return False
    return True


def list_image_paths_fast(split_dir, max_images=None):
    split_dir = Path(split_dir)
    if not split_dir.exists():
        return []
    paths = []
    for path in sorted(split_dir.rglob("*")):
        if path.is_file() and is_source_image(path):
            paths.append(path)
            if max_images and len(paths) >= max_images:
                break
    return paths


def scan_dataset(split_dirs=DATASET_SPLIT_DIRS, caption_csv=CAPTION_CSV, max_images=MAX_TRAIN_IMAGES):
    caption_map = load_caption_map(caption_csv)
    records = []
    found_split_images = False
    for split, split_dir in split_dirs.items():
        image_paths = list_image_paths_fast(split_dir, max_images=None)
        if image_paths:
            found_split_images = True
        for image_path in image_paths:
            records.append(ImageRecord(
                path=image_path,
                split=split,
                bucket="urban_pedestrian_scene",
                caption=build_caption(image_path, "urban_pedestrian_scene", caption_map),
                label_path=find_label_path(image_path),
                weather=None,
                timeofday=None,
                scene="urban",
            ))
            if max_images and len(records) >= max_images:
                return records

    if not found_split_images:
        for image_path in sorted(DATASET_ROOT.rglob("*")):
            if not image_path.is_file() or not is_source_image(image_path):
                continue
            records.append(ImageRecord(
                path=image_path,
                split=infer_split(image_path),
                bucket="urban_pedestrian_scene",
                caption=build_caption(image_path, "urban_pedestrian_scene", caption_map),
                label_path=find_label_path(image_path),
                weather=None,
                timeofday=None,
                scene="urban",
            ))
            if max_images and len(records) >= max_images:
                break
    return records

def load_records():
    records = scan_dataset()
    print(f"Scanned {len(records)} CityPersons images from {DATASET_ROOT}")
    if not records:
        print("No images found. Check DATASET_ROOT and Kaggle dataset mount.")
    else:
        split_counts = {}
        for record in records:
            split_counts[record.split] = split_counts.get(record.split, 0) + 1
        print("split counts:", split_counts)
        for record in records[:10]:
            print(record.split, record.bucket, record.path)
    return records

def summarize_citypersons_records(records):
    split_counts = {}
    for record in records:
        split_counts[record.split] = split_counts.get(record.split, 0) + 1
    print("CityPersons split counts:", split_counts)
    print("scene buckets:", sorted({record.bucket for record in records}))

def preview_prompt_samples(records, variants=("add_single_pedestrian", "add_two_pedestrians", "add_occluded_pedestrian", "add_distant_pedestrian"), n=3):
    for record in records[:n]:
        print("\nimage:", record.path.name)
        print("label:", {"weather": record.weather, "timeofday": record.timeofday, "scene": record.scene})
        print("train caption:", record.caption)
        for variant in variants:
            print(f"{variant} prompt:", build_generation_prompt(record, variant))

def preview_records(records, n=6):
    if not records:
        print("No dataset records to preview yet.")
        return
    sample = records[:n]
    cols = min(3, len(sample))
    rows = math.ceil(len(sample) / cols)
    plt.figure(figsize=(4 * cols, 4 * rows))
    for i, record in enumerate(sample, 1):
        image = Image.open(record.path).convert("RGB")
        plt.subplot(rows, cols, i)
        plt.imshow(image)
        plt.title(f"{record.split}/{record.bucket}\n{record.path.name[:32]}")
        plt.axis("off")
    plt.tight_layout()
