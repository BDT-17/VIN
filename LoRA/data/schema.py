"""Record schemas and role constants for the LoRA data ETL.

Roles in the LoRA release:
    lora_positive   -> a captioned pedestrian crop used for LoRA train/val

Eval cases live in a separate subsystem (build_eval_cases) and never carry a
LoRA role, so the release can never leak the evaluation images.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional


# ---- roles / statuses -------------------------------------------------------

ROLE_LORA_POSITIVE = "lora_positive"

EVAL_SET_DEV = "inpaint_eval_v1"
EVAL_SET_FINAL = "final_inpaint_test_v1"

STATUS_EXPORTED = "exported"
STATUS_VALIDATED = "validated"

PEDESTRIAN = "pedestrian"


@dataclass
class ImageRecord:
    image_id: str
    source_id: str
    raw_path: str
    source_image_id: str
    original_split: str          # train | val | test (source folder)
    width: int
    height: int
    sha256: str = ""
    phash: str = ""
    group_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InstanceRecord:
    instance_id: str
    image_id: str
    class_name: str
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    visible_bbox_x: float = 0.0
    visible_bbox_y: float = 0.0
    visible_bbox_w: float = 0.0
    visible_bbox_h: float = 0.0
    track_id: str = ""
    occlusion_level: Optional[float] = None
    ignore_flag: bool = False
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SampleRecord:
    sample_id: str
    image_id: str
    instance_id: str
    source_id: str
    role: str                    # ROLE_LORA_POSITIVE
    split: str                   # train | val
    group_id: str
    crop_path: str = ""
    crop_width: int = 0
    crop_height: int = 0
    bbox_height_px: float = 0.0
    visible_ratio: float = 0.0
    quality_score: float = 0.0
    caption: str = ""
    trigger_token: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalCase:
    case_id: str
    image_path: str
    mask_path: str
    reference_path: str
    expected_bbox_xyxy: List[float] = field(default_factory=list)
    prompt_fields: dict = field(default_factory=dict)
    source_split: str = ""
    eval_set: str = EVAL_SET_DEV
    frozen: bool = True

    def to_dict(self) -> dict:
        return asdict(self)
