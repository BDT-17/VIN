"""
SD3.5 LoRA VIN — Prompt Engineering Module.

Bổ sung prompt gaps cho pipeline hiện tại:
- scene variety (weather, timeofday, location)
- per-variant anatomical/pose/clothing details
- SD3.5-specific negative prompts
- LoRA adapter guidance
- style consistency
"""

from sd35_config import LORA_ENABLED, LORA_TRIGGER_TOKEN, LORA_PROMPT_PREFIX

# ============================================================
# 1. SCENE CONTEXT — weather, timeofday, location variety
# ============================================================
# Pipeline hiện tại chỉ có "urban street photo" cho mọi scene.
# Bổ sung chi tiết scene context cho từng bucket.

SCENE_CONTEXT_MAP = {
    "urban_pedestrian_scene": {
        "day_sunny": "urban street scene, bright daylight, clear sky, sharp shadows",
        "day_overcast": "urban street scene, overcast daylight, soft diffuse lighting",
        "evening": "urban street scene, golden hour evening light, warm tones, long shadows",
        "night": "urban street scene, nighttime, artificial street lighting, dark shadows",
        "rainy": "urban street scene, rainy weather, wet road surface, reflections, overcast",
        "foggy": "urban street scene, foggy weather, low visibility, soft hazy atmosphere",
    },
    "crosswalk_scene": {
        "day_sunny": "pedestrian crosswalk, bright daylight, zebra crossing, clear visibility",
        "day_overcast": "pedestrian crosswalk, overcast daylight, zebra crossing markings",
        "evening": "pedestrian crosswalk, evening light, zebra stripes visible",
    },
    "intersection_scene": {
        "day_sunny": "street intersection, traffic lights, bright daylight, multiple road lanes",
        "day_overcast": "street intersection, overcast daylight, traffic lights visible",
        "evening": "street intersection, evening, traffic lights active, dimmer ambient",
    },
    "sidewalk_scene": {
        "day_sunny": "paved sidewalk beside road, bright daylight, storefronts, trees",
        "day_overcast": "paved sidewalk, overcast daylight, pedestrians walking",
        "evening": "sidewalk, evening, street lamps lighting the walkway",
    },
}


def build_scene_context(bucket: str, weather: str = None, timeofday: str = None) -> str:
    """Build rich scene description from bucket + metadata.
    
    Pipeline hiện tại thiếu: không dùng weather/timeofday từ record.
    """
    scene_map = SCENE_CONTEXT_MAP.get(bucket, {})
    if weather and timeofday:
        key = f"{timeofday}_{weather}"
        if key in scene_map:
            return scene_map[key]
        # fallback: use timeofday only
        for k, v in scene_map.items():
            if k.startswith(timeofday):
                return v
    # fallback to first available
    if scene_map:
        return next(iter(scene_map.values()))
    return "urban street scene"


# ============================================================
# 2. PER-VARIANT DETAILED PROMPTS — pose, clothing, anatomy
# ============================================================
# Pipeline hiện tại: variant prompts quá ngắn (e.g. "one clear full-body pedestrian")
# Thiếu: pose, clothing detail, spatial relation, anatomical correctness.

VARIANT_DETAILED_PROMPTS = {
    "add_single_pedestrian": (
        "one realistic full-body pedestrian, facing forward or walking naturally, "
        "wearing typical urban clothing (shirt, pants, shoes), "
        "clearly visible head, torso, both arms, both legs, two feet on ground, "
        "natural human proportions, no distortion, correct limb anatomy"
    ),
    "add_two_pedestrians": (
        "two separate full-body pedestrians walking side by side or in opposite directions, "
        "both with visible head, torso, arms, legs, two feet on ground each, "
        "clear gap between them, natural spacing, both wearing typical street clothing, "
        "correct anatomy for both, no merged bodies, no overlapping limbs"
    ),
    "add_small_group": (
        "three separate full-body pedestrians walking on the same sidewalk, "
        "visible gaps between each person, staggered positions, "
        "each with correct anatomy: head, torso, arms, legs, feet on ground, "
        "natural depth ordering, front person overlaps rear person slightly, "
        "wearing urban clothing, no fused bodies, no extra limbs"
    ),
    "add_occluded_pedestrian": (
        "one full-body pedestrian partially hidden behind a foreground object like a car, pole, or another person, "
        "visible upper body (head, torso, arms) or lower body (legs below knee) depending on occluder position, "
        "the occluded parts are genuinely blocked not missing, context object is correctly in front, "
        "realistic occlusion boundary, no transparency, no ghosting through occluder"
    ),
    "add_distant_pedestrian": (
        "one distant full-body pedestrian seen from far away, small in frame, "
        "reduced but visible detail: head as small oval, torso, legs moving, "
        "consistent with perspective scale, matching depth in scene, "
        "no oversizing, no blurring into blob, correct proportion at distance"
    ),
    "add_near_pedestrian": (
        "one near full-body pedestrian close to camera, large in frame, "
        "clear clothing texture and fabric detail, visible facial features, "
        "head, torso, both arms with hands, both legs with shoes, both feet planted, "
        "dominant foreground presence, correct anatomy at close range, no cropping at frame edge"
    ),
}


# ============================================================
# 3. STYLE / QUALITY PROMPTS — ổn định chất lượng output
# ============================================================
# Pipeline hiện tại không có style prefix/suffix để ổn định chất lượng.

STYLE_PREFIX = (
    "high quality, realistic street photography, traffic camera view, "
    "sharp focus, natural lighting, authentic urban texture"
)

STYLE_SUFFIX = (
    "professional grade, realistic proportions, natural pose, "
    "sharp edges on person, no motion blur on person, correct depth placement"
)


# ============================================================
# 4. SD3.5-SPECIFIC NEGATIVE PROMPTS
# ============================================================
# Pipeline hiện tại thiếu:
# - SD3.5-specific artifacts (checkerboard, oil painting, plastic skin)
# - LoRA overfitting issues
# - anatomical details

SD35_NEGATIVE_PROMPTS = {
    # SD3.5 artifact patterns
    "sd35_artifacts": (
        "checkerboard pattern, grid artifact, oil painting texture, "
        "plastic skin, waxy face, airbrushed, over-smooth, "
        "vector art, illustration, cartoon, 3d render, cgi, "
        "oversaturated, blown out highlights, crushed blacks"
    ),
    # LoRA-specific issues
    "lora_artifacts": (
        "duplicate person, repeated texture, pattern repetition, "
        "color bleeding from LoRA, overfitted artifacts, "
        "training data watermark, training data artifact, "
        "content from training image, copied training composition"
    ),
    # Anatomical corrections
    "anatomy_negatives": (
        "extra limb, missing limb, amputee, deformed hand, "
        "twisted body, broken spine, contortion, impossible pose, "
        "floating limb disconnected from body, disproportionate limb, "
        "giant hand, tiny head, elongated neck, no neck, "
        "melted face, asymmetric face, disfigured face"
    ),
    # Street scene negatives
    "scene_negatives": (
        "indoor setting, bedroom, office, kitchen, studio background, "
        "green screen, plain background, abstract background, "
        "park, forest, nature, beach, mountain, rural area"
    ),
}

# Negative prompt when LoRA is enabled (combines all)
LORA_ENHANCED_NEGATIVE = (
    "ghost person, transparent human, translucent body, semi-transparent pedestrian, shadow person, "
    "silhouette, outline drawing, wireframe human, invisible body, "
    "floating limbs, partial body, missing legs, missing arms, cropped person, "
    "faceless person, dark blob, black silhouette, faded pedestrian, "
    "giant, closeup, floating, bad perspective, hard seam, merged people, fused bodies, "
    # LoRA-specific
    "duplicate person, repeated texture, pattern repetition, color bleeding from LoRA, "
    "overfitted artifacts, training data watermark, "
    # SD3.5-specific
    "checkerboard pattern, grid artifact, oil painting texture, "
    "plastic skin, waxy face, airbrushed, over-smooth, "
    # Anatomy
    "extra limb, missing limb, amputee, deformed hand, twisted body, "
    "broken spine, contortion, impossible pose, disproportionate limb, "
    "giant hand, tiny head, elongated neck, "
    # Scene
    "indoor setting, studio background, abstract background, green screen"
)


# ============================================================
# 5. LORA ADAPTER GUIDANCE PROMPT
# ============================================================
# Khi LoRA active, thêm guidance để model dùng adapter đúng cách.

LORA_GUIDANCE = (
    "Apply the pedestrian appearance style from the LoRA adapter, "
    "maintain correct human anatomy and proportions, "
    "blend naturally with scene lighting and perspective, "
    "avoid over-applying LoRA style, keep diversity in clothing color and pose"
)


# ============================================================
# 6. BUILDER FUNCTION — tổng hợp prompt hoàn chỉnh
# ============================================================

def build_enhanced_prompt(
    variant: str,
    bucket: str = "urban_pedestrian_scene",
    weather: str = None,
    timeofday: str = None,
    lora_enabled: bool = None,
    use_style_prefix: bool = True,
    use_style_suffix: bool = True,
) -> str:
    """Build complete prompt with all enhancements.
    
    Args:
        variant: add_single_pedestrian etc.
        bucket: scene bucket name
        weather: sunny, overcast, rainy, foggy
        timeofday: day, evening, night
        lora_enabled: override LORA_ENABLED config
        use_style_prefix: include STYLE_PREFIX
        use_style_suffix: include STYLE_SUFFIX
    
    Returns:
        Full positive prompt string
    """
    if lora_enabled is None:
        lora_enabled = LORA_ENABLED
    
    # Scene context
    scene = build_scene_context(bucket, weather, timeofday)
    
    # LoRA trigger prefix
    lora_prefix = ""
    if lora_enabled:
        parts = [str(v).strip() for v in (LORA_TRIGGER_TOKEN, LORA_PROMPT_PREFIX) if str(v).strip()]
        lora_prefix = ", ".join(parts) + ", " if parts else ""
    
    # Variant detail
    variant_detail = VARIANT_DETAILED_PROMPTS.get(variant, "")
    
    # Style
    style = f"{STYLE_PREFIX}. " if use_style_prefix else ""
    suffix = f" {STYLE_SUFFIX}" if use_style_suffix else ""
    
    # Assemble
    prompt = (
        f"{scene}. "
        f"{style}"
        f"{lora_prefix}{variant_detail}. "
        f"{suffix}"
    ).strip()
    
    # Clean up double spaces / punctuation
    prompt = prompt.replace("  ", " ").replace("..", ".").replace(", .", ".")
    if not prompt.endswith("."):
        prompt += "."
    
    return prompt


def build_enhanced_negative(variant: str = None, lora_enabled: bool = None) -> str:
    """Build negative prompt with variant-specific additions."""
    if lora_enabled is None:
        lora_enabled = LORA_ENABLED
    
    base = LORA_ENHANCED_NEGATIVE if lora_enabled else (
        "ghost person, transparent human, translucent body, semi-transparent pedestrian, shadow person, "
        "silhouette, outline drawing, wireframe human, invisible body, floating limbs, partial body, "
        "missing legs, missing arms, cropped person, faceless person, dark blob, black silhouette, "
        "faded pedestrian, giant, closeup, floating, bad perspective, hard seam, merged people, fused bodies"
    )
    
    variant_negatives = {
        "add_two_pedestrians": ", only one person, single pedestrian, not enough people",
        "add_small_group": ", only one or two people, not enough people in group",
        "add_occluded_pedestrian": (
            ", person fully visible with no occlusion, "
            "person floating in front of occluder, person painted over occluding object"
        ),
        "add_distant_pedestrian": ", closeup person, oversized distant figure, person too large for distance",
        "add_near_pedestrian": ", far away person, undersized foreground figure, person too small for foreground",
    }
    
    extra = variant_negatives.get(variant, "")
    return f"{base}{extra}"


# ============================================================
# 7. COMPATIBILITY WRAPPER — để dùng với pipeline hiện tại
# ============================================================

def wrap_existing_prompts(variant: str, existing_prompt: str, existing_negative: str) -> tuple:
    """Wrap existing pipeline prompts with enhanced versions.
    
    Trả về (enhanced_prompt, enhanced_negative) để thay thế trong pipeline.
    Pipeline hiện tại không cần sửa code, chỉ cần gọi wrapper này.
    """
    enhanced = build_enhanced_prompt(variant)
    enhanced_neg = build_enhanced_negative(variant)
    return enhanced, enhanced_neg
