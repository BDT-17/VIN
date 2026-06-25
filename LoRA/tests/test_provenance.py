"""Provenance artifacts capture model/dataset/adapter identity."""

import json
from pathlib import Path

import pytest

from LoRA.train.provenance import (write_training_provenance, write_dataset_provenance,
                                   write_validation_prompts, sha256_file)
from LoRA.data.config import load_prompt_config


def test_sha256_file(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello")
    assert sha256_file(p) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_dataset_provenance(tmp_path):
    rel = tmp_path / "rel"
    rel.mkdir()
    (rel / "release.json").write_text(json.dumps({
        "release_name": "pedestrian_lora_v1", "dataset_status": "validated",
        "manifest_hash": "abc", "git_commit": "deadbeef",
        "train_count": 10, "val_count": 2}), encoding="utf-8")
    out = write_dataset_provenance(tmp_path, rel)
    data = json.loads(Path(out).read_text())
    assert data["release_name"] == "pedestrian_lora_v1"
    assert data["dataset_status"] == "validated"
    assert data["train_count"] == 10


def test_training_provenance_includes_versions_and_trigger(tmp_path):
    adapter = tmp_path / "pytorch_lora_weights.safetensors"
    adapter.write_bytes(b"fake-adapter")
    script = tmp_path / "train.py"
    script.write_text("# trainer")
    train_cfg = {"base_model_id": "stabilityai/stable-diffusion-3.5-medium",
                 "rank": 8, "learning_rate": 1e-4, "max_train_steps": 1000, "seed": 42}
    release = {"release_name": "pedestrian_lora_v1", "trigger_token": "<vin_ped>",
               "manifest_hash": "abc", "git_commit": "sha"}
    out = write_training_provenance(tmp_path, train_cfg, release, adapter, script)
    data = json.loads(Path(out).read_text())
    assert data["trigger_token"] == "<vin_ped>"
    assert data["base_model_id"] == "stabilityai/stable-diffusion-3.5-medium"
    assert data["adapter_sha256"] == sha256_file(adapter)
    assert "diffusers_version" in data
    assert (tmp_path / "adapter_sha256.txt").exists()


def test_validation_prompts_contains_trigger(tmp_path):
    prompts = load_prompt_config()
    out = write_validation_prompts(tmp_path, prompts)
    data = json.loads(Path(out).read_text())
    assert data["trigger_token"] == prompts.trigger_token
    assert all(prompts.trigger_token in p for p in data["prompts"])
