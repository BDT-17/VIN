"""SD3.5 LoRA training entrypoint.

Builds and runs the pinned Diffusers `train_dreambooth_lora_sd3.py` command in
caption mode. The Diffusers script still requires --instance_prompt even with
--caption_column, so it is always passed. Loss is monitored live and training
hard-fails on NaN/Inf.

Run output layout:
    <model_root>/<model_name>/run_NNN/
        adapter/pytorch_lora_weights.safetensors (+ .pt)
        checkpoints/checkpoint-*/
        training_config.json, train_command.json
        training_provenance.json, dataset_provenance.json
        adapter_verification.json, adapter_sha256.txt
        pip_freeze.txt, gpu_info.json, validation_prompts.json
"""

import json
import math
import re
import subprocess
import sys
from pathlib import Path

from ..data.config import load_train_config, load_prompt_config
from ..data.validate import require_validated_release
from .export_artifacts import find_adapter, export_pt, write_adapter_verification
from .provenance import (write_pip_freeze, write_gpu_info, write_validation_prompts,
                         write_training_provenance, write_dataset_provenance)


def _next_run_dir(model_root: Path, model_name: str) -> Path:
    base = Path(model_root) / model_name
    base.mkdir(parents=True, exist_ok=True)
    existing = [int(m.group(1)) for p in base.glob("run_*")
                if (m := re.match(r"run_(\d+)$", p.name))]
    n = (max(existing) + 1) if existing else 1
    run_dir = base / f"run_{n:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_command(release_dir: Path, run_dir: Path, train_cfg: dict,
                  prompts, train_script: Path) -> list:
    data_dir = Path(release_dir) / "lora_train"
    cmd = [
        "accelerate", "launch", str(train_script),
        "--pretrained_model_name_or_path", str(train_cfg["base_model_id"]),
        "--dataset_name", str(data_dir),
        "--image_column", train_cfg.get("image_column", "image"),
        "--caption_column", train_cfg.get("caption_column", "text"),
        # Diffusers SD3 LoRA script requires this even in caption mode:
        "--instance_prompt", train_cfg.get("instance_prompt", prompts.training_instance_prompt),
        "--validation_prompt", train_cfg.get("validation_prompt", prompts.validation_prompts[0]),
        "--output_dir", str(run_dir / "adapter"),
        "--resolution", str(int(train_cfg.get("resolution", 512))),
        "--train_batch_size", str(int(train_cfg.get("train_batch_size", 1))),
        "--gradient_accumulation_steps", str(int(train_cfg.get("gradient_accumulation_steps", 4))),
        "--learning_rate", str(float(train_cfg.get("learning_rate", 1e-4))),
        "--lr_scheduler", str(train_cfg.get("lr_scheduler", "constant")),
        "--lr_warmup_steps", str(int(train_cfg.get("lr_warmup_steps", 0))),
        "--max_train_steps", str(int(train_cfg.get("max_train_steps", 1000))),
        "--checkpointing_steps", str(int(train_cfg.get("checkpointing_steps", 250))),
        "--rank", str(int(train_cfg.get("rank", 8))),
        "--seed", str(int(train_cfg.get("seed", 42))),
        "--mixed_precision", str(train_cfg.get("mixed_precision", "fp16")),
    ]
    if train_cfg.get("gradient_checkpointing", True):
        cmd.append("--gradient_checkpointing")
    if train_cfg.get("use_8bit_adam", True):
        cmd.append("--use_8bit_adam")
    if train_cfg.get("train_text_encoder", False):
        cmd.append("--train_text_encoder")
    if float(train_cfg.get("lora_dropout", 0)) > 0:
        cmd += ["--lora_dropout", str(float(train_cfg["lora_dropout"]))]
    return cmd


def _monitor(cmd: list, run_dir: Path):
    """Run training, stream output, parse loss, hard-fail on NaN/Inf."""
    metrics_path = run_dir / "metrics.jsonl"
    step_re = re.compile(r"\b(\d+)/(\d+)")
    loss_re = re.compile(r"\bloss[=: ]+([0-9.eE+\-]+|nan|inf)", re.IGNORECASE)
    with open(metrics_path, "w", encoding="utf-8") as mf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            sm, lm = step_re.search(line), loss_re.search(line)
            if sm and lm:
                ls = lm.group(1).lower()
                loss = float("nan") if ls == "nan" else (float("inf") if "inf" in ls else float(ls))
                mf.write(json.dumps({"step": int(sm.group(1)), "loss": loss}) + "\n"); mf.flush()
                if not math.isfinite(loss):
                    proc.kill()
                    raise RuntimeError(f"Training diverged: non-finite loss at step {sm.group(1)}")
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)


def run_training(release_dir, work_dir, dry_run=False, train_config_path=None,
                 prompts_path=None, max_train_steps=None):
    release_dir = Path(release_dir)
    release = require_validated_release(release_dir)

    train_cfg = load_train_config(train_config_path)
    prompts = load_prompt_config(prompts_path)
    if max_train_steps is not None:
        train_cfg["max_train_steps"] = int(max_train_steps)

    repo_root = Path(__file__).resolve().parent.parent.parent
    train_script = repo_root / "LoRA" / train_cfg["train_script"]

    model_root = Path(work_dir) / "models"
    run_dir = _next_run_dir(model_root, train_cfg["model_name"])

    cmd = build_command(release_dir, run_dir, train_cfg, prompts, train_script)
    (run_dir / "train_command.json").write_text(json.dumps(cmd, indent=2), encoding="utf-8")
    (run_dir / "training_config.json").write_text(json.dumps(train_cfg, indent=2), encoding="utf-8")
    write_gpu_info(run_dir)
    write_pip_freeze(run_dir)
    write_validation_prompts(run_dir, prompts)
    write_dataset_provenance(run_dir, release_dir)

    if dry_run:
        print("DRY RUN — command:\n  " + " \\\n  ".join(cmd))
        return {"run_dir": run_dir, "command": cmd, "dry_run": True}

    if not train_script.exists():
        raise FileNotFoundError(
            f"Training script not vendored: {train_script}\n"
            "Place train_dreambooth_lora_sd3.py under LoRA/vendor/diffusers/<commit>/ "
            "(see LoRA/vendor/diffusers/VENDOR.md)."
        )

    _monitor(cmd, run_dir)

    adapter = find_adapter(run_dir)
    pt_path = export_pt(adapter, run_dir / "adapter" / train_cfg.get("pt_artifact_name",
                                                                     "pytorch_lora_weights.pt"))
    verification = write_adapter_verification(run_dir, adapter)
    write_training_provenance(run_dir, train_cfg, release, adapter, train_script)

    print(f"\nTRAINING COMPLETE -> {run_dir}")
    print(f"  adapter: {adapter}")
    print(f"  loadable: {verification['loadable']} ({verification['key_count']} keys)")
    return {"run_dir": run_dir, "adapter_path": adapter, "pt_path": pt_path,
            "verification": verification, "dry_run": False}
