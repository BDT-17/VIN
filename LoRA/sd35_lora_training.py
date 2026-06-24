"""SD3.5 LoRA training and artifact export helpers.

This module intentionally stays separate from the augmentation runner. The LoRA
folder remains usable as an inference/evaluation harness, while training can be
called explicitly from a notebook or Kaggle cell when a prepared captioned
training dataset is available.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from sd35_config import *


SUPPORTED_ADAPTER_NAMES = (
    "pytorch_lora_weights.safetensors",
    "pytorch_lora_weights.bin",
)


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_lora_adapter_file(output_dir):
    output_dir = Path(output_dir)
    for name in SUPPORTED_ADAPTER_NAMES:
        candidate = output_dir / name
        if candidate.exists():
            return candidate
    matches = sorted(output_dir.rglob("*lora*.safetensors")) + sorted(output_dir.rglob("*lora*.bin"))
    if matches:
        return matches[-1]
    raise FileNotFoundError(f"No LoRA adapter weights found under {output_dir}")


def export_lora_pt(adapter_path=None, output_dir=None, pt_name=None):
    """Export a trained LoRA adapter to a .pt artifact for model handoff."""
    output_dir = Path(output_dir or LORA_TRAINING_OUTPUT_DIR)
    adapter_path = Path(adapter_path) if adapter_path else find_lora_adapter_file(output_dir)
    pt_path = output_dir / (pt_name or LORA_TRAINING_PT_NAME)
    pt_path.parent.mkdir(parents=True, exist_ok=True)

    import torch

    if adapter_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("Install safetensors to export .safetensors LoRA weights to .pt") from exc
        state_dict = load_file(str(adapter_path), device="cpu")
    else:
        state_dict = torch.load(str(adapter_path), map_location="cpu")

    payload = {
        "format": "sd35_lora_state_dict",
        "base_model_id": SD35_MODEL_ID,
        "adapter_name": LORA_ADAPTER_NAME,
        "source_adapter": str(adapter_path),
        "state_dict": state_dict,
    }
    torch.save(payload, str(pt_path))
    return pt_path


def training_provenance(adapter_path=None, pt_path=None):
    adapter_path = Path(adapter_path) if adapter_path else None
    pt_path = Path(pt_path) if pt_path else None
    payload = {
        "training_name": LORA_TRAINING_NAME,
        "base_model_id": SD35_MODEL_ID,
        "resolution": LORA_TRAINING_RESOLUTION,
        "mixed_precision": LORA_TRAINING_MIXED_PRECISION,
        "train_batch_size": LORA_TRAINING_BATCH_SIZE,
        "gradient_accumulation_steps": LORA_TRAINING_GRADIENT_ACCUMULATION_STEPS,
        "gradient_checkpointing": LORA_TRAINING_GRADIENT_CHECKPOINTING,
        "use_8bit_adam": LORA_TRAINING_USE_8BIT_ADAM,
        "rank": LORA_TRAINING_RANK,
        "lora_dropout": LORA_TRAINING_DROPOUT,
        "learning_rate": LORA_TRAINING_LEARNING_RATE,
        "max_train_steps": LORA_TRAINING_MAX_TRAIN_STEPS,
        "target_modules": LORA_TRAINING_TARGET_MODULES,
        "trigger_token": LORA_TRIGGER_TOKEN,
        "prompt_prefix": LORA_PROMPT_PREFIX,
        "adapter_path": str(adapter_path) if adapter_path else "",
        "adapter_sha256": sha256_file(adapter_path) if adapter_path and adapter_path.exists() else "",
        "pt_path": str(pt_path) if pt_path else "",
        "pt_sha256": sha256_file(pt_path) if pt_path and pt_path.exists() else "",
        "python": sys.version,
    }
    try:
        import torch
        payload["torch_version"] = torch.__version__
        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        payload["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    except Exception as exc:
        payload["torch_error"] = str(exc)
    return payload


def write_training_provenance(output_dir=None, adapter_path=None, pt_path=None):
    output_dir = Path(output_dir or LORA_TRAINING_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "training_provenance.json"
    path.write_text(
        json.dumps(_json_safe(training_provenance(adapter_path=adapter_path, pt_path=pt_path)), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def write_training_config(output_dir=None):
    output_dir = Path(output_dir or LORA_TRAINING_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "training_config.json"
    payload = {
        "base_model": SD35_MODEL_ID,
        "mixed_precision": LORA_TRAINING_MIXED_PRECISION,
        "resolution": LORA_TRAINING_RESOLUTION,
        "train_batch_size": LORA_TRAINING_BATCH_SIZE,
        "gradient_accumulation_steps": LORA_TRAINING_GRADIENT_ACCUMULATION_STEPS,
        "gradient_checkpointing": LORA_TRAINING_GRADIENT_CHECKPOINTING,
        "use_8bit_adam": LORA_TRAINING_USE_8BIT_ADAM,
        "cache_latents": LORA_TRAINING_CACHE_LATENTS,
        "max_sequence_length": LORA_TRAINING_MAX_SEQUENCE_LENGTH,
        "train_text_encoder": LORA_TRAINING_TRAIN_TEXT_ENCODER,
        "rank": LORA_TRAINING_RANK,
        "lora_dropout": LORA_TRAINING_DROPOUT,
        "learning_rate": LORA_TRAINING_LEARNING_RATE,
        "target_modules": LORA_TRAINING_TARGET_MODULES,
        "pt_artifact_name": LORA_TRAINING_PT_NAME,
    }
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def build_diffusers_training_command(script_path=None, dataset_release=None):
    """Build a caption-aware SD3.5 LoRA training command.

    Args:
        script_path: Path to training script
        dataset_release: Path to validated dataset release (ImageFolder format)

    Returns:
        Command list for subprocess
    """
    script_path = Path(script_path or os.environ.get("SD35_LORA_TRAIN_SCRIPT", "train_dreambooth_lora_sd3.py"))

    # If dataset_release provided, use it; otherwise fall back to old config
    if dataset_release:
        data_dir = Path(dataset_release) / "lora_train"

        # Validate release before training
        from LoRA.data.validate import require_validated_release
        require_validated_release(Path(dataset_release))
    else:
        data_dir = Path(LORA_TRAINING_DATA_DIR)

    output_dir = Path(LORA_TRAINING_OUTPUT_DIR)

    command = [
        sys.executable,
        str(script_path),
        "--pretrained_model_name_or_path", str(SD35_MODEL_ID),
        "--output_dir", str(output_dir),
        "--resolution", str(int(LORA_TRAINING_RESOLUTION)),
        "--train_batch_size", str(int(LORA_TRAINING_BATCH_SIZE)),
        "--gradient_accumulation_steps", str(int(LORA_TRAINING_GRADIENT_ACCUMULATION_STEPS)),
        "--learning_rate", str(float(LORA_TRAINING_LEARNING_RATE)),
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0",
        "--max_train_steps", str(int(LORA_TRAINING_MAX_TRAIN_STEPS)),
        "--checkpointing_steps", str(int(LORA_TRAINING_CHECKPOINTING_STEPS)),
        "--rank", str(int(LORA_TRAINING_RANK)),
        "--seed", str(int(LORA_TRAINING_SEED)),
        "--mixed_precision", str(LORA_TRAINING_MIXED_PRECISION),
    ]

    # Use caption-aware training if metadata.jsonl exists
    metadata_path = data_dir / "metadata.jsonl"
    if metadata_path.exists():
        # Caption-per-image mode
        command.extend([
            "--dataset_name", str(data_dir),
            "--caption_column", "text",
            "--image_column", "file_name",
        ])
        print(f"Using caption-per-image training from: {metadata_path}")
    else:
        # Legacy instance_data_dir mode
        command.extend([
            "--instance_data_dir", str(data_dir),
            "--instance_prompt", str(LORA_TRAINING_VALIDATION_PROMPT),
        ])
        print(f"Using legacy instance_data_dir mode: {data_dir}")

    # Add optional flags
    if LORA_TRAINING_GRADIENT_CHECKPOINTING:
        command.append("--gradient_checkpointing")
    if LORA_TRAINING_USE_8BIT_ADAM:
        command.append("--use_8bit_adam")
    if LORA_TRAINING_TRAIN_TEXT_ENCODER:
        command.append("--train_text_encoder")
    if LORA_TRAINING_CACHE_LATENTS:
        command.append("--cache_latents")

    # Add dropout if configured
    if LORA_TRAINING_DROPOUT > 0:
        command.extend(["--lora_dropout", str(float(LORA_TRAINING_DROPOUT))])

    return command


def run_lora_training(script_path=None, dataset_release=None, dry_run=False):
    """Run SD3.5 LoRA training with caption-aware dataset support.

    Args:
        script_path: Path to training script
        dataset_release: Path to validated dataset release directory
        dry_run: If True, only prepare command without running

    Returns:
        Dictionary with training artifacts
    """
    output_dir = Path(LORA_TRAINING_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write training config
    config_path = write_training_config(output_dir=output_dir)

    # Build training command
    command = build_diffusers_training_command(
        script_path=script_path,
        dataset_release=dataset_release,
    )

    # Save command
    command_path = output_dir / "train_command.json"
    command_path.write_text(json.dumps(command, indent=2), encoding="utf-8")

    # Log dataset provenance if using release
    if dataset_release:
        release_meta_path = Path(dataset_release) / "release.json"
        if release_meta_path.exists():
            with open(release_meta_path) as f:
                release_meta = json.load(f)

            provenance = {
                "dataset_release": str(dataset_release),
                "release_name": release_meta.get("release_name"),
                "release_version": release_meta.get("release_version"),
                "dataset_status": release_meta.get("dataset_status"),
                "manifest_hash": release_meta.get("manifest_hash"),
                "git_commit": release_meta.get("git_commit"),
            }

            dataset_prov_path = output_dir / "dataset_provenance.json"
            with open(dataset_prov_path, 'w') as f:
                json.dump(provenance, f, indent=2)

            print(f"Dataset provenance saved: {dataset_prov_path}")

    if dry_run:
        print("=" * 60)
        print("TRAINING DRY RUN")
        print("=" * 60)
        print(f"Config: {config_path}")
        print(f"Command: {command_path}")
        print("\nCommand:")
        print(" \\\n  ".join(command))
        return {"command": command, "command_path": command_path, "config_path": config_path}

    print("=" * 60)
    print("STARTING LORA TRAINING")
    print("=" * 60)

    # Run training
    subprocess.run(command, check=True)

    # Export artifacts
    adapter_path = find_lora_adapter_file(output_dir)
    pt_path = export_lora_pt(adapter_path=adapter_path, output_dir=output_dir)
    provenance_path = write_training_provenance(
        output_dir=output_dir,
        adapter_path=adapter_path,
        pt_path=pt_path,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"LoRA adapter: {adapter_path}")
    print(f"LoRA .pt model: {pt_path}")
    print(f"Training provenance: {provenance_path}")

    return {
        "command": command,
        "adapter_path": adapter_path,
        "pt_path": pt_path,
        "provenance_path": provenance_path,
    }


if __name__ == "__main__":
    run_lora_training(dry_run=not bool(LORA_TRAINING_ENABLED))
