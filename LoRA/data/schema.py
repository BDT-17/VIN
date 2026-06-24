"""Data schema definitions for VIN LoRA pipeline.

All manifests use Parquet as source of truth. JSONL exports are
secondary artifacts for trainer compatibility only.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
from pathlib import Path


class SourceType(str, Enum):
    """Dataset source type."""
    PEDESTRIAN_DETECTION = "pedestrian_detection"
    VIDEO_TRACKING = "video_tracking"
    BINARY_CLASSIFICATION = "binary_classification"
    IMAGE_FOLDER = "image_folder"


class GroupStrategy(str, Enum):
    """Grouping strategy for split safety."""
    ORIGINAL_SCENE = "original_scene"
    SEQUENCE_TIME_WINDOW = "sequence_time_window"
    PERCEPTUAL_CLUSTER = "perceptual_cluster"
    NONE = "none"


class DataRole(str, Enum):
    """Sample role in the pipeline."""
    LORA_POSITIVE = "lora_positive"
    BACKGROUND = "background"
    DETECTOR_TRAIN_REAL = "detector_train_real"
    DETECTOR_VAL_REAL_FROZEN = "detector_val_real_frozen"
    DETECTOR_TEST_REAL_FROZEN = "detector_test_real_frozen"


class DedupeStatus(str, Enum):
    """Deduplication status."""
    UNIQUE = "unique"
    EXACT_DUPLICATE = "exact_duplicate"
    NEAR_DUPLICATE = "near_duplicate"
    TEMPORAL_NEIGHBOR = "temporal_neighbor"
    CROSS_SOURCE_MIRROR = "cross_source_mirror"


class Split(str, Enum):
    """Dataset split."""
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    EXCLUDED = "excluded"


@dataclass
class ImageRecord:
    """Image manifest record.

    Corresponds to normalized/images.parquet schema.
    """
    image_id: str
    source_id: str
    source_image_id: str
    raw_path: str
    width: int
    height: int
    sha256: str
    phash: str
    group_id: str
    original_split: Optional[str] = None
    frame_index: Optional[int] = None
    camera_domain: Optional[str] = None
    file_size_bytes: Optional[int] = None

    def validate(self) -> List[str]:
        """Validate record and return list of errors."""
        errors = []
        if not self.image_id:
            errors.append("image_id is required")
        if not self.source_id:
            errors.append("source_id is required")
        if self.width <= 0:
            errors.append(f"width must be positive, got {self.width}")
        if self.height <= 0:
            errors.append(f"height must be positive, got {self.height}")
        if not self.sha256 or len(self.sha256) != 64:
            errors.append(f"invalid sha256: {self.sha256}")
        if not self.phash:
            errors.append("phash is required")
        return errors


@dataclass
class InstanceRecord:
    """Instance manifest record.

    Corresponds to normalized/instances.parquet schema.
    """
    instance_id: str
    image_id: str
    class_name: str
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    visible_bbox_x: Optional[float] = None
    visible_bbox_y: Optional[float] = None
    visible_bbox_w: Optional[float] = None
    visible_bbox_h: Optional[float] = None
    track_id: Optional[int] = None
    occlusion_level: Optional[float] = None
    ignore_flag: bool = False
    confidence: Optional[float] = None
    annotation_origin: Optional[str] = None

    def validate(self, image_width: int, image_height: int) -> List[str]:
        """Validate instance against image dimensions."""
        errors = []
        if not self.instance_id:
            errors.append("instance_id is required")
        if not self.image_id:
            errors.append("image_id is required")
        if self.class_name not in {"pedestrian", "person", "ignore"}:
            errors.append(f"invalid class_name: {self.class_name}")
        if self.bbox_w <= 0:
            errors.append(f"bbox_w must be positive, got {self.bbox_w}")
        if self.bbox_h <= 0:
            errors.append(f"bbox_h must be positive, got {self.bbox_h}")
        if self.bbox_x < 0 or self.bbox_x >= image_width:
            errors.append(f"bbox_x {self.bbox_x} outside image width {image_width}")
        if self.bbox_y < 0 or self.bbox_y >= image_height:
            errors.append(f"bbox_y {self.bbox_y} outside image height {image_height}")
        if self.bbox_x + self.bbox_w > image_width:
            errors.append(f"bbox extends beyond image width")
        if self.bbox_y + self.bbox_h > image_height:
            errors.append(f"bbox extends beyond image height")
        return errors


@dataclass
class GroupRecord:
    """Group manifest record.

    Corresponds to groups.parquet schema.
    """
    image_id: str
    duplicate_cluster_id: str
    split_group_id: str
    dedupe_status: str
    canonical_image_id: str
    duplicate_type: Optional[str] = None
    similarity_score: Optional[float] = None


@dataclass
class SampleRecord:
    """LoRA sample record.

    Corresponds to curated/samples.parquet schema.
    """
    sample_id: str
    image_id: str
    instance_id: str
    role: str
    split: str
    crop_path: str
    crop_width: int
    crop_height: int
    bbox_height_ratio: float
    visible_ratio: float
    occlusion_level: Optional[float]
    source_id: str
    quality_score: float
    caption: str
    trigger_token: str
    duplicate_cluster_id: str
    split_group_id: str

    def validate(self) -> List[str]:
        """Validate sample record."""
        errors = []
        if not self.sample_id:
            errors.append("sample_id is required")
        if self.role not in [r.value for r in DataRole]:
            errors.append(f"invalid role: {self.role}")
        if self.split not in [s.value for s in Split]:
            errors.append(f"invalid split: {self.split}")
        if not self.caption:
            errors.append("caption is required")
        if self.trigger_token not in self.caption:
            errors.append(f"trigger token '{self.trigger_token}' not in caption")
        if not Path(self.crop_path).exists():
            errors.append(f"crop file does not exist: {self.crop_path}")
        if self.crop_width <= 0 or self.crop_height <= 0:
            errors.append(f"invalid crop dimensions: {self.crop_width}x{self.crop_height}")
        if not 0 <= self.quality_score <= 1:
            errors.append(f"quality_score must be in [0,1], got {self.quality_score}")
        return errors


@dataclass
class DatasetRelease:
    """Dataset release metadata."""
    release_name: str
    release_version: str
    dataset_status: str  # building, validating, validated, failed
    source_manifests: Dict[str, str]  # source_id -> manifest_hash
    filter_config_hash: str
    caption_template_version: str
    split_seed: int
    output_manifest_hash: str
    git_commit: Optional[str] = None
    created_at: Optional[str] = None
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if release is validated and ready for training."""
        return (
            self.dataset_status == "validated"
            and len(self.validation_errors) == 0
        )


@dataclass
class SourceConfig:
    """Source configuration from sources.yaml."""
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


# Parquet schema definitions for type consistency
IMAGES_PARQUET_SCHEMA = {
    "image_id": "string",
    "source_id": "string",
    "source_image_id": "string",
    "raw_path": "string",
    "width": "int32",
    "height": "int32",
    "sha256": "string",
    "phash": "string",
    "group_id": "string",
    "original_split": "string",
    "frame_index": "int32",
    "camera_domain": "string",
    "file_size_bytes": "int64",
}

INSTANCES_PARQUET_SCHEMA = {
    "instance_id": "string",
    "image_id": "string",
    "class_name": "string",
    "bbox_x": "float32",
    "bbox_y": "float32",
    "bbox_w": "float32",
    "bbox_h": "float32",
    "visible_bbox_x": "float32",
    "visible_bbox_y": "float32",
    "visible_bbox_w": "float32",
    "visible_bbox_h": "float32",
    "track_id": "int32",
    "occlusion_level": "float32",
    "ignore_flag": "bool",
    "confidence": "float32",
    "annotation_origin": "string",
}

GROUPS_PARQUET_SCHEMA = {
    "image_id": "string",
    "duplicate_cluster_id": "string",
    "split_group_id": "string",
    "dedupe_status": "string",
    "canonical_image_id": "string",
    "duplicate_type": "string",
    "similarity_score": "float32",
}

SAMPLES_PARQUET_SCHEMA = {
    "sample_id": "string",
    "image_id": "string",
    "instance_id": "string",
    "role": "string",
    "split": "string",
    "crop_path": "string",
    "crop_width": "int32",
    "crop_height": "int32",
    "bbox_height_ratio": "float32",
    "visible_ratio": "float32",
    "occlusion_level": "float32",
    "source_id": "string",
    "quality_score": "float32",
    "caption": "string",
    "trigger_token": "string",
    "duplicate_cluster_id": "string",
    "split_group_id": "string",
}
