"""Configuration loader for LoRA data pipeline."""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class QualityThresholds:
    """Quality filtering thresholds."""
    min_bbox_height_px: int = 96
    min_visible_ratio: float = 0.45
    min_quality_score: float = 0.70
    max_source_share: float = 0.50


@dataclass
class SplitConfig:
    """Dataset split configuration."""
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_seed: int = 42
    enforce_group_lock: bool = True


@dataclass
class CaptionConfig:
    """Caption generation configuration."""
    trigger_token: str = "<vin_ped>"
    template_version: str = "v1"
    min_caption_tokens: int = 5
    max_caption_tokens: int = 77


@dataclass
class ExportConfig:
    """Dataset export configuration."""
    format: str = "imagefolder_jsonl"
    crop_context_ratio: float = 0.25
    crop_min_size: int = 128
    crop_max_size: int = 768


@dataclass
class SourceDefinition:
    """Source dataset definition."""
    source_id: str
    kaggle_mount: str
    parser: str
    source_type: str
    allowed_roles: List[str]
    group_strategy: str
    duplicate_priority: int
    benchmark_lock: bool = False
    license_review_required: bool = False
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
class PipelineConfig:
    """Complete pipeline configuration."""
    release_name: str
    sources: List[SourceDefinition]
    quality_thresholds: QualityThresholds
    split_config: SplitConfig
    caption_config: CaptionConfig
    export_config: ExportConfig


def load_sources_config(config_path: Path) -> PipelineConfig:
    """Load sources.yaml configuration."""
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)

    sources = []
    for src in raw.get('sources', []):
        sources.append(SourceDefinition(**src))

    quality = QualityThresholds(**raw.get('quality_thresholds', {}))
    split = SplitConfig(**raw.get('split_config', {}))
    caption = CaptionConfig(**raw.get('caption_config', {}))
    export = ExportConfig(**raw.get('export_config', {}))

    return PipelineConfig(
        release_name=raw['release_name'],
        sources=sources,
        quality_thresholds=quality,
        split_config=split,
        caption_config=caption,
        export_config=export,
    )


def get_source_by_id(config: PipelineConfig, source_id: str) -> Optional[SourceDefinition]:
    """Get source definition by ID."""
    for src in config.sources:
        if src.source_id == source_id:
            return src
    return None


def get_benchmark_locked_sources(config: PipelineConfig) -> List[str]:
    """Get list of benchmark-locked source IDs."""
    return [src.source_id for src in config.sources if src.benchmark_lock]
