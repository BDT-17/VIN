"""Local config for the standalone AI Replace flow.

This file intentionally lives under inpaint/ so the rollback-era V5 pipeline
stays untouched.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AIReplaceConfig:
    AI_REPLACE_ENABLED: bool = True
    AI_REPLACE_FLOW: str = "pokecut_style_ai_replace"
    AI_REPLACE_MASK_SOURCE: str = "auto_bbox"

    AI_REPLACE_HARD_RESTORE_OUTSIDE_MASK: bool = True
    AI_REPLACE_SOFT_BORDER_PX: int = 4
    AI_REPLACE_MASK_EXPAND_PX: int = 12
    AI_REPLACE_MASK_BLUR_PX: int = 3

    AI_REPLACE_STRENGTH: float = 0.72
    AI_REPLACE_GUIDANCE_SCALE: float = 7.0
    AI_REPLACE_STEPS: int = 32
    AI_REPLACE_RESOLUTION: int = 512

    AI_REPLACE_OBJECT_CLASS: str = "person"
    AI_REPLACE_PROMPT: str = (
        "one realistic opaque pedestrian, full body, natural lighting, "
        "matching camera perspective, realistic street photo"
    )
    AI_REPLACE_NEGATIVE_PROMPT: str = (
        "transparent, ghost, silhouette, cropped, missing limbs, "
        "giant, closeup, floating, bad perspective, hard seam, "
        "halo, blurry, duplicate, text, watermark"
    )

    COLOR_TRANSFER_STRENGTH: float = 0.15
    MAX_COLOR_TRANSFER_STRENGTH: float = 0.25
    MAX_CORE_BLEND: float = 0.03
    MAX_SOFT_BOUNDARY_BLEND: float = 0.15
    MAX_FINE_EDGE_BLEND: float = 0.22

    SHADOW_ALPHA: float = 0.25
    SHADOW_BLUR: int = 15
    SHARPEN_STRENGTH: float = 0.15

    MIN_OBJECT_INSIDE_RATIO: float = 0.75
    MIN_OPACITY_SCORE: float = 0.75
    MIN_CONTRAST_SCORE: float = 0.55
    MAX_DETECTOR_CONF_DROP: float = 0.15
    MAX_OUTSIDE_MASK_DIFF: float = 1.0

    MODEL_ID: str = "runwayml/stable-diffusion-inpainting"
    TORCH_DTYPE: str = "float16"


DEFAULT_CONFIG = AIReplaceConfig()
