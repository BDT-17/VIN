"""Stage 07 — export the ImageFolder + JSONL LoRA release.

Layout:
    <release>/lora_train/images/*.jpg + metadata.jsonl
    <release>/lora_val/images/*.jpg   + metadata.jsonl
    <release>/manifest.parquet
    <release>/release.json

metadata.jsonl rows: {"file_name": "images/<id>.jpg", "text": "<caption>"}
(loads via datasets.load_dataset('imagefolder', ...) -> columns image, text)
"""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from .config import SourcesConfig
from .schema import STATUS_EXPORTED


def _git_sha(repo_dir: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                             capture_output=True, text=True)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _copy_split(samples_df: pd.DataFrame, split: str, release_dir: Path) -> int:
    split_dir = release_dir / f"lora_{split}"
    img_dir = split_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = samples_df[samples_df["split"] == split]
    meta_lines = []
    n = 0
    for _, r in rows.iterrows():
        src = Path(r["crop_path"])
        if not src.exists():
            continue
        fname = f"{r['sample_id']}.jpg"
        shutil.copy2(src, img_dir / fname)
        meta_lines.append(json.dumps({"file_name": f"images/{fname}", "text": r["caption"]},
                                     ensure_ascii=False))
        n += 1
    (split_dir / "metadata.jsonl").write_text("\n".join(meta_lines) + ("\n" if meta_lines else ""),
                                              encoding="utf-8")
    return n


def run_export(samples_df: pd.DataFrame, work_dir: Path, config: SourcesConfig,
               repo_dir: Path = None) -> Path:
    release_dir = Path(work_dir) / "releases" / config.release_name
    release_dir.mkdir(parents=True, exist_ok=True)

    n_train = _copy_split(samples_df, "train", release_dir)
    n_val = _copy_split(samples_df, "val", release_dir)

    samples_df.to_parquet(release_dir / "manifest.parquet", index=False)
    manifest_hash = hashlib.sha256(
        (release_dir / "manifest.parquet").read_bytes()).hexdigest()

    trigger = samples_df["trigger_token"].iloc[0] if len(samples_df) else ""
    release = {
        "release_name": config.release_name,
        "dataset_status": STATUS_EXPORTED,
        "trigger_token": trigger,
        "caption_template_version": config.caption_config.template_version,
        "base_model_id": config.base_model_id,
        "manifest_hash": manifest_hash,
        "git_commit": _git_sha(repo_dir or Path.cwd()),
        "train_count": n_train,
        "val_count": n_val,
    }
    (release_dir / "release.json").write_text(json.dumps(release, indent=2), encoding="utf-8")
    print(f"  export: lora_train={n_train}, lora_val={n_val} -> {release_dir}")
    return release_dir
