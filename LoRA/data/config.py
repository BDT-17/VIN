"""Typed loaders for the YAML configs under LoRA/configs/."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _load_yaml(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---- sources.yaml -----------------------------------------------------------

@dataclass
class SourceDefinition:
    source_id: str
    kaggle_mount: str
    parser: str
    source_type: str
    group_strategy: str
    duplicate_priority: int
    lora_splits: List[str] = field(default_factory=list)
    eval_splits: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    temporal_sampling_fps: Optional[float] = None
    temporal_window_seconds: Optional[float] = None
    splits: Optional[Dict[str, str]] = None
    label_dirs: Optional[Dict[str, str]] = None
    sequence_dir: Optional[str] = None
    gt_dir: Optional[str] = None
    positive_dir: Optional[str] = None
    negative_dir: Optional[str] = None


@dataclass
class QualityThresholds:
    min_bbox_height_px: int = 96
    min_visible_ratio: float = 0.45
    min_quality_score: float = 0.70
    max_source_share: float = 0.50


@dataclass
class SplitConfig:
    train_ratio: float = 0.85
    val_ratio: float = 0.15
    split_seed: int = 42


@dataclass
class EvalConfig:
    eval_v1_ratio: float = 0.70
    final_holdout_ratio: float = 0.30
    eval_seed: int = 42


@dataclass
class CaptionConfig:
    trigger_token: str = "<vin_ped>"
    class_token: str = "pedestrian"
    template_version: str = "v1"
    min_caption_tokens: int = 5
    max_caption_tokens: int = 77


@dataclass
class ExportConfig:
    format: str = "imagefolder_jsonl"
    crop_context_ratio: float = 0.25
    crop_min_size: int = 128
    crop_max_size: int = 768


@dataclass
class SourcesConfig:
    release_name: str
    base_model_id: str
    sources: List[SourceDefinition]
    quality_thresholds: QualityThresholds
    split_config: SplitConfig
    eval_config: EvalConfig
    caption_config: CaptionConfig
    export_config: ExportConfig


def load_sources_config(path=None) -> SourcesConfig:
    raw = _load_yaml(path or CONFIGS_DIR / "sources.yaml")
    return SourcesConfig(
        release_name=raw["release_name"],
        base_model_id=raw.get("base_model_id", "stabilityai/stable-diffusion-3.5-medium"),
        sources=[SourceDefinition(**s) for s in raw.get("sources", [])],
        quality_thresholds=QualityThresholds(**raw.get("quality_thresholds", {})),
        split_config=SplitConfig(**raw.get("split_config", {})),
        eval_config=EvalConfig(**raw.get("eval_config", {})),
        caption_config=CaptionConfig(**raw.get("caption_config", {})),
        export_config=ExportConfig(**raw.get("export_config", {})),
    )


# ---- prompt_templates.yaml --------------------------------------------------

@dataclass
class PromptConfig:
    trigger_token: str
    class_token: str
    template_version: str
    caption_template: str
    training_instance_prompt: str
    validation_prompts: List[str]
    inpaint_prompt_template: str
    negative_prompt: str


def load_prompt_config(path=None) -> PromptConfig:
    raw = _load_yaml(path or CONFIGS_DIR / "prompt_templates.yaml")
    return PromptConfig(
        trigger_token=raw["trigger_token"],
        class_token=raw.get("class_token", "pedestrian"),
        template_version=raw.get("template_version", "v1"),
        caption_template=" ".join(raw["caption_template"].split()),
        training_instance_prompt=raw["training_instance_prompt"],
        validation_prompts=list(raw.get("validation_prompts", [])),
        inpaint_prompt_template=" ".join(raw["inpaint_prompt_template"].split()),
        negative_prompt=" ".join(raw["negative_prompt"].split()),
    )


# ---- lora_train.yaml --------------------------------------------------------

def load_train_config(path=None) -> dict:
    return _load_yaml(path or CONFIGS_DIR / "lora_train.yaml")
