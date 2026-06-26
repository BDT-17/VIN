"""Add-it configuration — faithful re-implementation (paper §3, App A.1).

Every constant traces to a specific place in Tewel et al., ICLR 2025
(arXiv:2411.07232).  The paper targets FLUX.1-dev; this port runs on SD3.5
Medium, so FLUX-specific *integer* timesteps and *absolute* layer indices are
re-expressed as **fractions** of SD3.5's schedule/stack.  Each such constant is
flagged ``SD3.5-ADAPT`` and explained in ``docs/addit_paper_fidelity.md``.

This file deliberately does NOT import the parent ``sd35_config`` placement /
detector / variant machinery: Add-it decides object placement *itself* (its
affordance thesis), so there is no insertion-bbox, no YOLO cutout-paste, and no
``find_insertion_region``.  Only the few model/runtime constants Add-it needs
(model id, resolution, device) are pulled in.
"""

from pathlib import Path
import sys

# Make the repo root importable for the handful of shared runtime constants.
_PARENT_DIR = str(Path(__file__).resolve().parent.parent)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

# Pull ONLY what Add-it needs from the parent config (model id, resolution,
# device, T5 flag).  We avoid ``import *`` so none of the augmentation-flow
# placement/validation knobs leak into this faithful flow.
from sd35_config import (  # noqa: F401
    SD35_MODEL_ID,
    MODEL_BACKEND,
    RESOLUTION,
    TRAIN_DEVICE,
    USE_T5,
    SEED,
)

# ═══════════════════════════════════════════════════════════════════════════
# 0. Backbone / runtime
# ═══════════════════════════════════════════════════════════════════════════
# SD3.5-ADAPT: the paper uses FLUX.1-dev.  We run SD3.5 Medium (the repo's
# model).  The mechanism is identical; only the schedule/layer constants below
# are re-derived.
ADDIT_BACKBONE = "sd35"                       # informational; loader builds SD3.5
ADDIT_NUM_INFERENCE_STEPS = 30                # paper: 30 denoising steps (App A.1)
ADDIT_GUIDANCE_SCALE = 5.0                    # SD3.5 CFG (FLUX uses distilled guidance; SD3.5 needs real CFG)
ADDIT_SEED = 42

# Single-GPU by default; the loop can temporarily move the transformer to a
# second GPU when present (kept from the original scaffold, optional).
ADDIT_USE_TWO_GPUS = False
ADDIT_PRIMARY_DEVICE = "cuda:0"
ADDIT_TRANSFORMER_DEVICE = "cuda:1"

# ═══════════════════════════════════════════════════════════════════════════
# 1. Weighted Extended Self-Attention  (paper §3.2, eq. 3)
# ═══════════════════════════════════════════════════════════════════════════
ADDIT_EXTENDED_ATTENTION = True

# γ balancing.  Paper: solve f(γ)=A_source−A_target=0 with a root-solver; the
# validated average is γ≈1.05.  γ multiplies the TARGET keys; source keys = 1,
# prompt keys = 1.
ADDIT_GAMMA_MODE = "auto"                     # "auto" (root-solver) | "fixed"
ADDIT_GAMMA_FIXED = 1.05                      # paper's validated default
ADDIT_GAMMA_SOURCE = 1.0                      # γ_s — held at 1 (paper)
ADDIT_GAMMA_SEARCH_LO = 0.6                   # root-solver bracket
ADDIT_GAMMA_SEARCH_HI = 1.6
ADDIT_GAMMA_SOLVE_ITERS = 12
ADDIT_GAMMA_SOLVE_EVERY = 5                   # re-solve γ every N steps (probe is extra forwards)

# Extended-attention layer window.
# SD3.5-ADAPT: paper applies extended attention in FLUX multi-stream blocks up
# to t=670 and single-stream up to t=340.  SD3.5 has ONE block type (24 dual-
# stream blocks), so the per-layer split is collapsed to "all layers"; the
# *timestep* gating is handled by ADDIT_ATTENTION_T_FRAC below.
ADDIT_ATTENTION_LAYER_FRAC = (0.0, 1.0)       # fraction of the 24-block stack

# SD3.5-ADAPT: paper stops extended attention partway through denoising
# (t=670/1000 ≈ 0.67 of the way from full-noise).  We gate by remaining-noise
# fraction: inject while σ_t ≥ this (i.e. early/mid denoising), matching the
# paper's "apply early, release late" schedule.
ADDIT_ATTENTION_SIGMA_MIN = 0.34              # ≈ 340/1000; inject while σ_t ≥ 0.34

# ═══════════════════════════════════════════════════════════════════════════
# 2. Structure Transfer  (paper §3.3, App A.1)
# ═══════════════════════════════════════════════════════════════════════════
# Paper: start denoising from the source noised to a FIXED high t_struct.
#   t_struct = 933 for generated images, 867 for real images (out of 1000).
# SD3.5-ADAPT: expressed as a fraction of num_train_timesteps (≈ σ at start).
ADDIT_T_STRUCT_FRAC_REAL = 0.867              # 867/1000 — real source images
ADDIT_T_STRUCT_FRAC_GEN = 0.933               # 933/1000 — generated source images
ADDIT_SOURCE_IS_REAL = True                   # CityPersons frames are real photos

# ═══════════════════════════════════════════════════════════════════════════
# 3. Subject-Guided Latent Blending  (paper §3.4, App A.1)
# ═══════════════════════════════════════════════════════════════════════════
ADDIT_LATENT_BLENDING = True

# Paper: blend at a single timestep t_blend = 500 / 1000.
ADDIT_T_BLEND_FRAC = 0.50                     # 500/1000

# Subject-attention capture window.
# SD3.5-ADAPT: paper's mask layers are FLUX [13,14,18, single-23, single-33].
# Those sit at relative depths ≈ {0.68, 0.74, 0.95} of the multi-stream stack
# and ≈ {0.60, 0.87} of the single-stream stack.  Mapped onto SD3.5's 24 blocks
# we take the corresponding mid-to-deep fractions:
ADDIT_MASK_LAYER_FRACS = (0.55, 0.60, 0.68, 0.75, 0.90)

# Capture subject attention over these step ratios (the paper aggregates over
# "specific timesteps and layers"; we aggregate over the mid denoising window
# leading up to t_blend, where subject structure is well-formed).
ADDIT_SUBJECT_CAPTURE_START_FRAC = 0.20       # start ratio (0=first step)
ADDIT_SUBJECT_CAPTURE_END_FRAC = 0.55         # end ratio (~ at t_blend)

# Point sampling for SAM-2 (paper App A.1).
ADDIT_MASK_MAX_POINTS = 4                     # up to 4 points
ADDIT_MASK_POINT_REL_THRESH = 0.35            # stop below 0.35 · p_max
ADDIT_MASK_POINT_EXCLUDE_FRAC = 0.10          # neighborhood excluded between picks

# Mask post-processing.
ADDIT_MASK_DILATE_PX = 4                      # small grow so shadows/contact stay inside M
ADDIT_MASK_FEATHER_PX = 2                     # soft blend edge

# After decode, optionally enforce pixel-exact source outside the refined mask
# (paper's latent blend keeps the background latent; this hardens it in pixels).
ADDIT_FINAL_PIXEL_COMPOSITE = True

ADDIT_USE_SAM2 = True                         # soft: falls back to Otsu mask if unavailable

# ═══════════════════════════════════════════════════════════════════════════
# 4. Prompting  (paper §3 + App A.7 — TARGET prompts, not instructions)
# ═══════════════════════════════════════════════════════════════════════════
# Add-it conditions on a *target prompt* describing the edited image, plus a
# *subject token* (the single noun naming the added object).  The model decides
# where to place it.  No insertion bbox, no per-variant placement profiles.
ADDIT_DEFAULT_TARGET_PROMPT = (
    "a photo of a city street with a pedestrian walking"
)
ADDIT_DEFAULT_SUBJECT_TOKEN = "pedestrian"

# SD3.5 needs real CFG, so a light negative prompt is allowed (FLUX-dev used
# distilled guidance with no negative).  Keep it minimal — placement is the
# model's job, not the prompt's.
ADDIT_NEGATIVE_PROMPT = "low quality, blurry, distorted, deformed"

# Optional example prompt/subject pairs for the demo/eval (free-placement).
ADDIT_EXAMPLE_PROMPTS = [
    ("a photo of a city street with a pedestrian walking", "pedestrian"),
    ("a city sidewalk with a person standing near the curb", "person"),
    ("an urban road with two pedestrians crossing", "pedestrians"),
]

# ═══════════════════════════════════════════════════════════════════════════
# 5. Output & debug
# ═══════════════════════════════════════════════════════════════════════════
ADDIT_OUTPUT_DIR = Path("/kaggle/working/addit_faithful")
ADDIT_DEBUG_DIR = Path("/kaggle/working/addit_debug")
ADDIT_SAVE_DEBUG = True
ADDIT_DEBUG_MAX_ITEMS = 20
ADDIT_SAVE_MASK_VIS = True                    # save the attention map + refined mask
