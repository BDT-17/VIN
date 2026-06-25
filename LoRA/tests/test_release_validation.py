"""Release validation gates: hard-fail on the contract violations from the spec."""

import json
from pathlib import Path

import pandas as pd
import pytest

from LoRA.data.validate import validate_release, require_validated_release

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def _make_release(tmp_path, captions, splits, trigger="<vin_ped>", make_crops=True,
                  groups=None, clusters=None):
    rel = tmp_path / "rel"
    (rel / "lora_train" / "images").mkdir(parents=True, exist_ok=True)
    (rel / "lora_val" / "images").mkdir(parents=True, exist_ok=True)
    n = len(captions)
    groups = groups or [f"g{i}" for i in range(n)]
    clusters = clusters or [f"c{i}" for i in range(n)]
    rows = []
    for i, (cap, sp, g, cl) in enumerate(zip(captions, splits, groups, clusters)):
        crop = rel / f"crop_{i}.jpg"
        if make_crops:
            Image.new("RGB", (200, 400), (120, 120, 120)).save(crop)
        rows.append({"sample_id": f"s{i}", "image_id": f"img{i}", "instance_id": f"in{i}",
                     "source_id": "citypersons", "role": "lora_positive", "split": sp,
                     "group_id": g, "duplicate_cluster_id": cl,
                     "crop_path": str(crop), "caption": cap, "trigger_token": trigger})
    df = pd.DataFrame(rows)
    df.to_parquet(rel / "manifest.parquet", index=False)
    (rel / "release.json").write_text(json.dumps({
        "release_name": "t", "dataset_status": "exported", "trigger_token": trigger,
        "manifest_hash": "x"}), encoding="utf-8")
    return rel


def test_valid_release_passes_and_flips_status(tmp_path):
    rel = _make_release(tmp_path,
                        captions=["a photo of <vin_ped> pedestrian, full body",
                                  "a photo of <vin_ped> pedestrian, side view"],
                        splits=["train", "val"])
    report = validate_release(rel)
    assert report["valid"], report["errors"]
    assert require_validated_release(rel)["dataset_status"] == "validated"


def test_missing_trigger_token_fails(tmp_path):
    rel = _make_release(tmp_path,
                        captions=["a photo of pedestrian, full body",   # no trigger
                                  "a photo of <vin_ped> pedestrian"],
                        splits=["train", "val"])
    report = validate_release(rel)
    assert not report["valid"]
    assert any("trigger" in e for e in report["errors"])


def test_empty_val_fails(tmp_path):
    rel = _make_release(tmp_path,
                        captions=["a photo of <vin_ped> pedestrian"],
                        splits=["train"])
    report = validate_release(rel)
    assert not report["valid"]
    assert any("lora_val" in e for e in report["errors"])


def test_group_overlap_train_val_fails(tmp_path):
    rel = _make_release(tmp_path,
                        captions=["a photo of <vin_ped> pedestrian a",
                                  "a photo of <vin_ped> pedestrian b"],
                        splits=["train", "val"],
                        groups=["shared", "shared"])  # same group both sides
    report = validate_release(rel)
    assert not report["valid"]
    assert any("group_id overlap" in e for e in report["errors"])


def test_eval_leak_fails(tmp_path):
    rel = _make_release(tmp_path,
                        captions=["a photo of <vin_ped> pedestrian a",
                                  "a photo of <vin_ped> pedestrian b"],
                        splits=["train", "val"])
    eval_manifest = pd.DataFrame([{"image_id": "img0", "group_id": "g0", "eval_set": "inpaint_eval_v1"}])
    report = validate_release(rel, eval_manifest=eval_manifest)
    assert not report["valid"]
    assert any("leaked" in e for e in report["errors"])
