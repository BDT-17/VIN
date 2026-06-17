"""Add-it CityPersons augmentation: configuration.

All Add-it-specific parameters live here.  Parent pipeline config
(dataset paths, YOLO thresholds, placement rules, etc.) is imported
from the existing ``sd35_config`` / ``sd35_utils`` modules via sys.path.
"""

from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Make the repository root importable so we can reuse
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
ADDIT_WEIGHTED_EXTENDED_ATTENTION = True   # Enable extended attention
ADDIT_ATTENTION_SCHEDULE = "cosine"       # cosine | linear | constant
ADDIT_W_SOURCE_START     = 0.42           # lower source pull so the edit is not copied away
ADDIT_W_SOURCE_END       = 0.02           # keep late denoising focused on the generated person
ADDIT_W_SELF_START       = 0.38           # stronger target latent stream for visible insertion
ADDIT_W_SELF_END         = 0.88           # preserve target detail late in denoising
# Text weight = 1 - w_source - w_self (implicit, clamped ≥ 0.05)
ADDIT_ATTENTION_LAYER_RANGE = (0.0, 1.0)  # fraction of layers to inject (0=first, 1=all)

# ═══════════════════════════════════════════════════════════════════════════
# 2. Noise Structure Transfer
# ═══════════════════════════════════════════════════════════════════════════
ADDIT_STRUCTURE_STRENGTH  = 0.82          # stronger edit budget inside the insertion region
ADDIT_NOISE_BLEND_RATIO   = 0.72          # less source-correlated noise so new people emerge

# ═══════════════════════════════════════════════════════════════════════════
# 3. Subject-Guided Latent Blending
# ═══════════════════════════════════════════════════════════════════════════
ADDIT_BLEND_FEATHER_LATENT  = 3           # tighter edge; less source overwrite inside the person region
ADDIT_BLEND_START_RATIO     = 0.0         # blending active from this step ratio
ADDIT_BLEND_END_RATIO       = 1.0        # blending stops at this step ratio (higher preserves background longer)
ADDIT_MASK_DILATION_LATENT  = 5           # give legs/feet enough latent space to form
ADDIT_MASK_EXPANSION_RATIO  = 1.60        # larger edit island; final YOLO cutout still preserves BG
ADDIT_FINAL_PIXEL_COMPOSITE = True        # restore original pixels outside insertion region after VAE decode
ADDIT_FINAL_COMPOSITE_MODE  = "bbox"      # bbox = exact background outside bbox; silhouette = tighter person mask
ADDIT_FINAL_COMPOSITE_FEATHER_PX = 0      # 0 preserves outside pixels exactly; >0 softens the boundary
ADDIT_FINAL_PERSON_CUTOUT = True          # after Add-it generation, cut detected person and paste onto source
ADDIT_PERSON_CUTOUT_CONF = 0.15           # YOLO-seg confidence for extracting generated person
ADDIT_PERSON_CUTOUT_MASK_THRESHOLD = 48    # keep weak lower-body mask pixels without falling back to bbox paste
ADDIT_PERSON_CUTOUT_DILATE_PX = 2          # keep accessories/feet while still preserving background outside the mask
ADDIT_PERSON_CUTOUT_FEATHER_PX = 0.45      # very thin edge blend so the person does not look like a sticker
ADDIT_PERSON_CUTOUT_EDGE_MIN_ALPHA = 24    # alpha below this becomes background
ADDIT_PERSON_CUTOUT_EDGE_FULL_ALPHA = 96   # alpha above this keeps generated person fully
ADDIT_PERSON_CUTOUT_FALLBACK_TO_BBOX = False  # never paste the whole insert bbox when segmentation fails
ADDIT_PERSON_CUTOUT_CONTRAST_BOOST = 1.18  # make added person less washed out before masked paste
ADDIT_PERSON_CUTOUT_SHARPNESS_BOOST = 1.35 # sharpen only generated person pixels; source BG is untouched
ADDIT_REQUIRE_PERSON_CUTOUT = True         # reject/retry if no generated person mask is extracted
ADDIT_MIN_PERSON_CUTOUT_AREA_RATIO = 0.00035  # reject tiny masks that are unlikely to be useful people
ADDIT_MIN_PERSON_CUTOUT_MAE_255 = 3.0      # reject outputs that are effectively identical to source
ADDIT_MIN_PERSON_CUTOUT_SHARPNESS = 4.0    # reject very blurry generated-person cutouts

# Multi-GPU Add-it runtime. The notebook loads SD3.5 on GPU 0, then moves the
# MM-DiT transformer to GPU 1 when available. VAE/text stay on GPU 0.
ADDIT_USE_TWO_GPUS = True
ADDIT_PRIMARY_DEVICE = "cuda:0"
ADDIT_TRANSFORMER_DEVICE = "cuda:1"

# ═══════════════════════════════════════════════════════════════════════════
# 4. Generation Defaults (can be overridden per-variant)
# ═══════════════════════════════════════════════════════════════════════════
ADDIT_NUM_INFERENCE_STEPS = 36
ADDIT_GUIDANCE_SCALE      = 7.8
ADDIT_SEED                = 42
ADDIT_FALLBACK_TO_NATIVE_IMG2IMG = True  # keep smoke runs useful if custom denoise fails
ADDIT_ADAPTIVE_RETRY_ENABLED = True      # adapt prompt/params after each rejection reason
ADDIT_ADAPTIVE_MAX_STRENGTH_DELTA = 0.14
ADDIT_ADAPTIVE_MAX_GUIDANCE_DELTA = 1.20
ADDIT_ADAPTIVE_MAX_EXTRA_STEPS = 12

# Per-variant overrides (mirroring parent VARIANT_PROFILE structure).
ADDIT_VARIANT_OVERRIDES = {
    "add_single_pedestrian":   {"strength": 0.82, "guidance": 7.8, "steps": 42},
    "add_two_pedestrians":     {"strength": 0.84, "guidance": 7.9, "steps": 42},
    "add_small_group":         {"strength": 0.86, "guidance": 8.0, "steps": 44},
    "add_occluded_pedestrian": {"strength": 0.84, "guidance": 7.8, "steps": 42},
    "add_distant_pedestrian":  {"strength": 0.78, "guidance": 7.6, "steps": 40},
    "add_near_pedestrian":     {"strength": 0.86, "guidance": 8.3, "steps": 44},
}

# ═══════════════════════════════════════════════════════════════════════════
# 5. Prompt Templates
# ═══════════════════════════════════════════════════════════════════════════
ADDIT_SOURCE_PROMPT = "urban street photo, city scene, natural lighting"

ADDIT_TARGET_PROMPTS = {
    "add_single_pedestrian":   "urban street photo with one newly added, clearly visible, sharp full-body pedestrian walking on the sidewalk, distinct foreground silhouette, visible legs and feet",
    "add_two_pedestrians":     "urban street photo with two newly added, clearly visible, separate full-body pedestrians walking on the sidewalk, visible legs and feet",
    "add_small_group":         "urban street photo with three newly added, clearly visible, separate full-body pedestrians in a small group on the sidewalk",
    "add_occluded_pedestrian": "urban street photo with a newly added, clearly visible partly occluded full-body pedestrian behind a foreground object, visible body and feet",
    "add_distant_pedestrian":  "urban street photo with a newly added distant but clearly visible full-body pedestrian, detectable silhouette",
    "add_near_pedestrian":     "urban street photo with a newly added near larger sharp full-body pedestrian walking, clear clothing, visible legs and feet",
}

ADDIT_NEGATIVE_PROMPT = (
    "cropped, missing head, missing legs, thin body, giant, closeup, "
    "floating, ghost, bad perspective, hard seam, overlap, "
    "merged people, fused bodies, blurry, low quality, faded, transparent, "
    "low contrast, blended into background, unchanged image, no new person"
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
