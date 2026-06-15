"""Add-it CityPersons augmentation: configuration.

All Add-it-specific parameters live here.  Parent pipeline config
(dataset paths, YOLO thresholds, placement rules, etc.) is imported
from the existing ``sd35_config`` / ``sd35_utils`` modules via sys.path.
"""

from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Make the parent ``notebooks/`` directory importable so we can reuse
# sd35_config, sd35_data, sd35_utils, sd35_evaluation, sd35_model.
# ---------------------------------------------------------------------------
_PARENT_DIR = str(Path(__file__).resolve().parent.parent)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

# Re-export everything from the parent config so downstream code can do
# ``from addit_config import *`` and get both parent and Add-it symbols.
from sd35_config import *  # noqa: F401,F403

# ═══════════════════════════════════════════════════════════════════════════
# 1. Weighted Extended-Attention
# ═══════════════════════════════════════════════════════════════════════════
ADDIT_ATTENTION_SCHEDULE = "cosine"       # cosine | linear | constant
ADDIT_W_SOURCE_START     = 0.70           # w_source at step 0 (noisy)
ADDIT_W_SOURCE_END       = 0.05           # w_source at final step (clean)
ADDIT_W_SELF_START       = 0.20           # w_self at step 0
ADDIT_W_SELF_END         = 0.85           # w_self at final step
# Text weight = 1 - w_source - w_self (implicit, clamped ≥ 0.05)
ADDIT_ATTENTION_LAYER_RANGE = (0.0, 1.0)  # fraction of layers to inject (0=first, 1=all)

# ═══════════════════════════════════════════════════════════════════════════
# 2. Noise Structure Transfer
# ═══════════════════════════════════════════════════════════════════════════
ADDIT_STRUCTURE_STRENGTH  = 0.72          # like img2img strength (0=full noise, 1=source only)
ADDIT_NOISE_BLEND_RATIO   = 0.85          # blend source noise vs pure random

# ═══════════════════════════════════════════════════════════════════════════
# 3. Subject-Guided Latent Blending
# ═══════════════════════════════════════════════════════════════════════════
ADDIT_BLEND_FEATHER_LATENT  = 3           # Gaussian blur radius for latent mask edges
ADDIT_BLEND_START_RATIO     = 0.0         # blending active from this step ratio
ADDIT_BLEND_END_RATIO       = 0.80        # blending stops at this step ratio
ADDIT_MASK_DILATION_LATENT  = 2           # dilate insertion mask in latent px
ADDIT_MASK_EXPANSION_RATIO  = 1.15        # expand bbox by this ratio for mask

# ═══════════════════════════════════════════════════════════════════════════
# 4. Generation Defaults (can be overridden per-variant)
# ═══════════════════════════════════════════════════════════════════════════
ADDIT_NUM_INFERENCE_STEPS = 36
ADDIT_GUIDANCE_SCALE      = 7.0
ADDIT_SEED                = 42

# Per-variant overrides (mirroring parent VARIANT_PROFILE structure).
ADDIT_VARIANT_OVERRIDES = {
    "add_single_pedestrian":   {"strength": 0.72, "guidance": 6.8, "steps": 36},
    "add_two_pedestrians":     {"strength": 0.74, "guidance": 6.9, "steps": 36},
    "add_small_group":         {"strength": 0.76, "guidance": 7.0, "steps": 38},
    "add_occluded_pedestrian": {"strength": 0.74, "guidance": 6.8, "steps": 36},
    "add_distant_pedestrian":  {"strength": 0.68, "guidance": 6.6, "steps": 34},
    "add_near_pedestrian":     {"strength": 0.76, "guidance": 7.3, "steps": 38},
}

# ═══════════════════════════════════════════════════════════════════════════
# 5. Prompt Templates
# ═══════════════════════════════════════════════════════════════════════════
ADDIT_SOURCE_PROMPT = "urban street photo, city scene, natural lighting"

ADDIT_TARGET_PROMPTS = {
    "add_single_pedestrian":   "urban street photo with one clear full-body pedestrian walking on the sidewalk",
    "add_two_pedestrians":     "urban street photo with two separate full-body pedestrians walking on the sidewalk",
    "add_small_group":         "urban street photo with three separate full-body pedestrians in a small group on the sidewalk",
    "add_occluded_pedestrian": "urban street photo with a partly occluded full-body pedestrian behind a foreground object",
    "add_distant_pedestrian":  "urban street photo with a distant clear full-body pedestrian",
    "add_near_pedestrian":     "urban street photo with a near larger full-body pedestrian walking",
}

ADDIT_NEGATIVE_PROMPT = (
    "cropped, missing head, missing legs, thin body, giant, closeup, "
    "floating, ghost, bad perspective, hard seam, overlap, "
    "merged people, fused bodies, blurry, low quality"
)

# ═══════════════════════════════════════════════════════════════════════════
# 6. Output & Debug
# ═══════════════════════════════════════════════════════════════════════════
ADDIT_OUTPUT_DIR       = Path("/kaggle/working/addit_citypersons")
ADDIT_DEBUG_DIR        = Path("/kaggle/working/addit_debug")
ADDIT_SAVE_DEBUG       = True
ADDIT_DEBUG_MAX_ITEMS  = 10
ADDIT_SAVE_STEP_VIS    = False   # save latent at every N steps (expensive)
ADDIT_STEP_VIS_INTERVAL = 5

# ═══════════════════════════════════════════════════════════════════════════
# 7. Retry & Validation (reuse parent thresholds, override here if needed)
# ═══════════════════════════════════════════════════════════════════════════
ADDIT_MAX_RETRIES = 3
ADDIT_RETRY_SEED_STEP = 9973     # seed += this per retry
