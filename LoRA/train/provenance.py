"""Training provenance: capture exactly what produced an adapter."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pkg_version(name: str) -> str:
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return ""


def write_pip_freeze(out_dir: Path) -> Path:
    out = Path(out_dir) / "pip_freeze.txt"
    try:
        res = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                             capture_output=True, text=True)
        out.write_text(res.stdout, encoding="utf-8")
    except Exception as exc:
        out.write_text(f"# pip freeze failed: {exc}\n", encoding="utf-8")
    return out


def write_gpu_info(out_dir: Path) -> Path:
    out = Path(out_dir) / "gpu_info.json"
    info = {"cuda_available": False}
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["torch_version"] = torch.__version__
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1)
    except Exception as exc:
        info["error"] = str(exc)
    out.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return out


def write_validation_prompts(out_dir: Path, prompts) -> Path:
    out = Path(out_dir) / "validation_prompts.json"
    out.write_text(json.dumps({
        "trigger_token": prompts.trigger_token,
        "instance_prompt": prompts.training_instance_prompt,
        "prompts": prompts.validation_prompts,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def write_training_provenance(out_dir: Path, train_cfg: dict, release: dict,
                              adapter_path: Path, train_script: Path) -> Path:
    out_dir = Path(out_dir)
    prov = {
        "base_model_id": train_cfg.get("base_model_id"),
        "release_name": release.get("release_name"),
        "trigger_token": release.get("trigger_token"),
        "manifest_hash": release.get("manifest_hash"),
        "dataset_git_commit": release.get("git_commit"),
        "rank": train_cfg.get("rank"),
        "learning_rate": train_cfg.get("learning_rate"),
        "max_train_steps": train_cfg.get("max_train_steps"),
        "seed": train_cfg.get("seed"),
        "diffusers_version": _pkg_version("diffusers"),
        "transformers_version": _pkg_version("transformers"),
        "accelerate_version": _pkg_version("accelerate"),
        "peft_version": _pkg_version("peft"),
        "train_script": str(train_script),
        "train_script_sha256": sha256_file(train_script) if Path(train_script).exists() else "",
        "adapter_sha256": sha256_file(adapter_path) if Path(adapter_path).exists() else "",
    }
    p = out_dir / "training_provenance.json"
    p.write_text(json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")

    if Path(adapter_path).exists():
        (out_dir / "adapter_sha256.txt").write_text(prov["adapter_sha256"], encoding="utf-8")
    return p


def write_dataset_provenance(out_dir: Path, release_dir: Path) -> Path:
    release = json.loads((Path(release_dir) / "release.json").read_text(encoding="utf-8"))
    p = Path(out_dir) / "dataset_provenance.json"
    p.write_text(json.dumps({
        "dataset_release": str(release_dir),
        "release_name": release.get("release_name"),
        "dataset_status": release.get("dataset_status"),
        "manifest_hash": release.get("manifest_hash"),
        "git_commit": release.get("git_commit"),
        "train_count": release.get("train_count"),
        "val_count": release.get("val_count"),
    }, indent=2), encoding="utf-8")
    return p
