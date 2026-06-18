"""SD3.5 CityPersons augmentation: configuration."""

from pathlib import Path

# User-facing settings: change these between runs.
RUN_PRESET = "smoke"  # smoke | quality | batch

USER_CONFIG = {
    "DATASET_ROOT_CANDIDATES": [
        Path('/kaggle/input/datasets/kyoru4444/mot17-02-fcrnn/MOT17-02-FRCNN'),
        Path('/kaggle/input/datasets/kyoru4444/mot17-02-fcrnn/MOT17-02-FRCNN/img1'),
        Path('/kaggle/input/datasets/nguyenaabcxyzeric/cityperson'),
        Path('/kaggle/input/datasets/nguyenaabcxyzeric/CityPerson'),
        Path('/kaggle/input/datasets/nguyenaabcxyzeric/cityperson/test/images'),
        Path('/kaggle/input/datasets/nguyenaabcxyzeric/CityPerson/test/images'),
        Path('/kaggle/input/datasets/muttahirulislam/citypersons-dataset-with-bg-image/yolo_dir/yolo_dir'),
        Path('/kaggle/input/citypersons-dataset-with-bg-image/yolo_dir/yolo_dir'),
        Path('/kaggle/input/citypersons-dataset-with-bg-image'),
        Path('/kaggle/input/datasets/samyamine23/cityperson'),
        Path('/kaggle/input/cityperson'),
        Path('/kaggle/input/citypersons'),
        Path('/kaggle/input/city-persons'),
        Path('/kaggle/input/city-persons-2-0'),
        Path('/kaggle/input/city-persons-20'),
        Path('/kaggle/input/dataset'),
        Path('/kaggle/working/Dataset'),
    ],
    "MODEL_BACKEND": "sd35",
    "SD35_MODEL_ID": "stabilityai/stable-diffusion-3.5-medium",
    "OUTPUT_DIR": Path("/kaggle/working/sd35_citypersons_scale_corrected"),
    "RESOLUTION": 512,
    "MAX_TRAIN_IMAGES": 500,
    "USE_T5": False,
    "TRAIN_DEVICE": "cuda:0",
    "USE_ALL_GPUS_FOR_AUGMENTATION": True,
    "AUGMENTATION_DEVICES": None,
    "USE_MODEL_CPU_OFFLOAD": True,
    "AUGMENTATION_VARIANTS": [
        'add_single_pedestrian',
        'add_two_pedestrians',
        'add_small_group',
        'add_occluded_pedestrian',
        'add_distant_pedestrian',
        'add_near_pedestrian',
    ],
    "SEED": 42,
}

PARAMETER_OVERRIDES = {
    "TARGET_SPLITS": ["test"],
    # Keep this short. Put manual one-off changes here, e.g.:
    # "CONTEXT_GENERATION_RETRIES": 4,
    # "MAX_PERSON_PERSON_OVERLAP_RATIO": 0.18,
}

AUTOTUNE_SETTINGS = {
    "enabled": True,
    "min_samples": 30,
    "target_accept_rate": 0.55,
    "target_quality_score": 0.72,
    "aggressiveness": 0.60,
    "max_adjustment_ratio": 1.35,
    "max_retry_budget": 5,
    "save_snapshot": True,
    "snapshot_dir": "/kaggle/working/autotune_snapshots",
    "print_report": True,
}

RUN_PRESETS = {
    "smoke": {
        "AUGMENTATIONS_PER_BUCKET": 5,
        "TARGET_SPLITS": ["train"],
        "SAVE_PATCH_DEBUG": True,
        "PATCH_DEBUG_MAX_ITEMS": 10,
        "CONTEXT_GENERATION_RETRIES": 1,
    },
    "quality": {
        "AUGMENTATIONS_PER_BUCKET": 25,
        "TARGET_SPLITS": ["train", "val"],
        "SAVE_PATCH_DEBUG": True,
        "PATCH_DEBUG_MAX_ITEMS": 24,
        "CONTEXT_GENERATION_RETRIES": 3,
    },
    "batch": {
        "AUGMENTATIONS_PER_BUCKET": 200,
        "TARGET_SPLITS": ["train", "val"],
        "SAVE_PATCH_DEBUG": True,
        "PATCH_DEBUG_MAX_ITEMS": 24,
        "CONTEXT_GENERATION_RETRIES": 3,
    },
}

# Variant profile: weights and per-variant SD3.5 generation settings.
VARIANT_PROFILE = {'add_single_pedestrian': {'weight': 0.24, 'strength': 0.72, 'guidance': 6.8, 'steps': 36},
 'add_two_pedestrians': {'weight': 0.18, 'strength': 0.74, 'guidance': 6.9, 'steps': 36},
 'add_small_group': {'weight': 0.16, 'strength': 0.76, 'guidance': 7.0, 'steps': 38},
 'add_occluded_pedestrian': {'weight': 0.18, 'strength': 0.74, 'guidance': 6.8, 'steps': 36},
 'add_distant_pedestrian': {'weight': 0.14, 'strength': 0.68, 'guidance': 6.6, 'steps': 34},
 'add_near_pedestrian': {'weight': 0.1, 'strength': 0.76, 'guidance': 7.3, 'steps': 38}}
AUGMENTATION_VARIANT_WEIGHTS = {name: cfg["weight"] for name, cfg in VARIANT_PROFILE.items()}
VARIANT_STRENGTHS = {name: cfg["strength"] for name, cfg in VARIANT_PROFILE.items()}
VARIANT_GUIDANCE_SCALES = {name: cfg["guidance"] for name, cfg in VARIANT_PROFILE.items()}
VARIANT_NUM_INFERENCE_STEPS = {name: cfg["steps"] for name, cfg in VARIANT_PROFILE.items()}

# Domain configuration groups. Keep behavior-preserving threshold changes here.

DATASET_CONFIG = {'SCENE_BUCKETS': ['urban_pedestrian_scene'],
 'IMAGE_SUBDIR': 'images',
 'CAPTION_CSV': None,
 'MATCH_LABEL_JSON': False,
 'PATCH_PERSON_CLASS_IDS': {0},
 'PATCH_VEHICLE_CLASS_IDS': {2, 5, 7}}

PLACEMENT_CONFIG = {'BACKGROUND_PRESERVATION_MODE': 'context_person_composite',
 'PATCH_CONTEXT_RATIO': 0.18,
 'PATCH_MIN_SIZE': 128,
 'PATCH_FEATHER_RADIUS': 3,
 'PATCH_MAX_PLACEMENT_TRIES': 180,
 'PATCH_ROAD_Y_RANGE': (0.68, 0.90),
 'INSERTION_EDGE_MARGIN': 16,
 'INSERTION_OVERLAP_PENALTY': 3.0,
 'ALLOW_PERSON_VEHICLE_OVERLAP': True,
 'ALLOW_PERSON_PERSON_OVERLAP': True,
 'MAX_PERSON_PERSON_OVERLAP_RATIO': 0.08,
 'OCCLUDED_PERSON_MAX_HEIGHT_RATIO': 0.68,
 'OCCLUDED_PERSON_MAX_FOOT_Y_DELTA': 0,
 'PERSON_OVERLAP_MIN_FRONT_HEIGHT_RATIO': 1.18,
 'PERSON_OVERLAP_FRONT_LAYER_BONUS': 0.0,
 'MAX_VEHICLE_OVERLAP_RATIO': 0.22,
 'VEHICLE_OVERLAP_FRONT_LAYER_BONUS': 0.08,
 'INSERTION_CENTER_BIAS': 0.0,
 'PLACEMENT_SLOT_BIAS': 1.45,
 'PLACEMENT_SLOT_XS': (0.2, 0.36, 0.52, 0.68, 0.82),
 'PLACEMENT_SLOT_YS': (0.68, 0.74, 0.8, 0.86, 0.91),
 'PLACEMENT_SLOT_JITTER': 0.06,
 'SMART_PLACEMENT_VERSION': 'v2',
 'USE_SEMANTIC_PLACEMENT': True,
 'SEMANTIC_SEGMENTATION_MODEL_ID': 'nvidia/segformer-b0-finetuned-cityscapes-1024-1024',
 'VALID_PLACEMENT_LABELS': {'terrain', 'sidewalk', 'road'},
 'AVOID_PLACEMENT_LABELS': {'bicycle',
                            'building',
                            'bus',
                            'car',
                            'fence',
                            'motorcycle',
                            'person',
                            'pole',
                            'rider',
                            'sky',
                            'traffic light',
                            'traffic sign',
                            'train',
                            'truck',
                            'wall'},
 'MIN_FOOT_SUPPORT': 0.48,
 'MAX_FOOT_AVOID_SUPPORT': 0.05,
 'MIN_BODY_VALID_SUPPORT': 0.14,
 'MAX_BODY_AVOID_SUPPORT': 0.16,
 'REQUIRE_SEMANTIC_PLACEMENT': False,
 'MIN_ACCEPTED_PLACEMENT_SCORE': -1000.0,
 'SEMANTIC_FOOT_WEIGHT': 6.5,
 'SEMANTIC_AVOID_PENALTY': 9.0}

SCALE_CONFIG = {'PERSPECTIVE_SCALE_NEAR': 1.14,
 'PERSPECTIVE_SCALE_FAR': 0.50,
 'USE_REFERENCE_PERSON_SCALE': True,
 'REFERENCE_SCALE_MIN_PERSON_HEIGHT': 24,
 'REFERENCE_SCALE_MAX_PERSON_HEIGHT': 180,
 'REFERENCE_SCALE_BLEND': 0.56,
 'REFERENCE_SCALE_MAX_Y_DISTANCE': 70,
 'CAR_HEIGHT_TO_PERSON_HEIGHT_RATIO': 1.05,
 'CAR_HEIGHT_TO_PERSON_HEIGHT_MIN_RATIO': 0.9,
 'CAR_HEIGHT_TO_PERSON_HEIGHT_MAX_RATIO': 1.1,
 'NEAR_PERSON_SCALE_MULTIPLIER': 1.6,
 'PERSON_TARGET_SCALE_MULTIPLIER': 1.6,
 'CAR_REFERENCE_SCALE_BLEND': 0.7,
 'CAR_REFERENCE_MAX_Y_DISTANCE': 120,
 'CAR_REFERENCE_MIN_HEIGHT': 18,
 'CAR_REFERENCE_MAX_HEIGHT': 130,
 'REFERENCE_SCALE_MIN_FACTOR': 0.7,
 'REFERENCE_SCALE_MAX_FACTOR': 1.75,
 'REFERENCE_SCALE_MIN_SAMPLES': 2,
 'REFERENCE_SCALE_MIN_Y_GAP': 24,
 'REFERENCE_SCALE_MIN_SLOPE': 0.06,
 'REFERENCE_SCALE_MAX_SLOPE': 1.1,
 'PERSON_ASPECT_RATIO': 0.36,
 'USE_PERSPECTIVE_SCALE_CORRECTION': True,
 'SCALE_CORRECTION_SOFT_MIN': 0.62,
 'SCALE_CORRECTION_SOFT_MAX': 2.2,
 'SCALE_CORRECTION_HARD_MIN': 0.35,
 'SCALE_CORRECTION_HARD_MAX': 3.2,
 'SCALE_CORRECTION_BORDERLINE_RETRY': True,
 'SCALE_CORRECTION_BORDERLINE_MIN_CONF': 0.48,
 'SCALE_CORRECTION_BORDERLINE_MIN_MASK_AREA_RATIO': 0.0015,
 'MIN_BORDER_MARGIN_RATIO': 0.018,
 'ALLOW_PARTIAL_SMALL_GROUP': True,
 'FLEXIBLE_SCALE_INPAINT': True,
 'SCALE_ENVELOPE_HEIGHT_MULT': 1.16,
 'SCALE_ENVELOPE_WIDTH_MULT': 1.3,
 'GROUP_PERSON_GAP_RATIO': 0.1,
 'GROUP_PERSON_HEIGHT_JITTER': 0.1,
 'GROUP_SCALE_JITTER_ENABLED': False,
 'PERSPECTIVE_MONOTONIC_SCALE_ENABLED': True,
 'PERSPECTIVE_MONOTONIC_MIN_RATIO': 1.04}

YOLO_EVAL_CONFIG = {'CONTEXT_PERSON_SEGMENTATION_MODEL': 'yolov8m-seg.pt',
 'CONTEXT_PERSON_MIN_CONFIDENCE': 0.12,
 'MIN_RETRY_PERSON_CONFIDENCE': 0.28,
 'MIN_PERSON_CONF_BY_VARIANT': {'add_single_pedestrian': 0.22,
                                'add_two_pedestrians': 0.17,
                                'add_small_group': 0.15,
                                'add_occluded_pedestrian': 0.12,
                                'add_distant_pedestrian': 0.11,
                                'add_near_pedestrian': 0.22},
 'MIN_GHOST_PERSON_MASK_AREA_RATIO': 0.002,
 'MIN_GHOST_PERSON_CONTRAST_255': 12.0,
 'CONTEXT_PERSON_MASK_THRESHOLD': 0.40,
 'MIN_PERSON_DET_TARGET_HEIGHT_RATIO': 0.62,
 'MAX_PERSON_TOP_OFFSET_RATIO': 0.42,
 'MAX_PERSON_BOTTOM_GAP_RATIO': 0.3,
 'MIN_PERSON_MASK_DET_HEIGHT_RATIO': 0.62,
 'MIN_PERSON_MASK_TARGET_HEIGHT_RATIO': 0.48,
 'MIN_PERSON_MASK_VERTICAL_BAND_COVERAGE': 0.18,
 'MIN_PERSON_MASK_ASPECT_RATIO': 1.35,
 'MAX_PERSON_MASK_ASPECT_RATIO': 6.0,
 'PERSON_MASK_DILATE_FOR_ACCESSORIES': 2,
 'PERSON_MASK_ERODE_PIXELS': 0,
 'PERSON_MASK_TRIM_FRINGE_PIXELS': 2,
 'PERSON_PASTE_HARD_THRESHOLD': 132,
 'PERSON_PASTE_FEATHER_RADIUS': 0.08,
 'ACCESSORY_KEEP_COMPONENTS': 12,
 'KEEP_EXTRA_GENERATED_PEOPLE': True,
 'MAX_EXTRA_GENERATED_PEOPLE': 2,
 'EXTRA_PERSON_MIN_SCORE_DELTA': 1.35,
 'PERSON_CROP_MASK_PADDING': 4,
 'MAX_GENERATED_PERSON_SHARPNESS_STD': 9.0,
 'ACCESSORY_MIN_COMPONENT_AREA_RATIO': 0.002,
 'FILL_PERSON_MASK_HOLES': True}

MASK_CONFIG = {'HUMAN_MASK_PADDING': 10,
 'HUMAN_MASK_BLUR_RADIUS': 2,
 'BBOX_MASK_PADDING': 8,
 'BBOX_MASK_BLUR_RADIUS': 3,
 'BBOX_MASK_RADIUS': 8}

COMPOSITING_CONFIG = {'USE_SEAMLESS_CLONE': False,
 'SEAMLESS_CLONE_MODE': 'mixed',
 'SEAMLESS_CLONE_MIN_MASK_AREA_RATIO': 0.0006,
 'SEAMLESS_FOREGROUND_PRESERVE_STRENGTH': 0.92,
 'SEAMLESS_FOREGROUND_CORE_ERODE': 0,
 'SEAMLESS_EDGE_ONLY_BLEND': True,
 'HARMONIZATION_COLOR_STRENGTH': 0.18,
 'HARMONIZATION_BRIGHTNESS_STRENGTH': 0.16,
 'HARMONIZATION_CONTRAST_STRENGTH': 0.12,
 'HARMONIZATION_SATURATION_STRENGTH': 0.10,
 'HARMONIZATION_NOISE_STRENGTH': 0.18,
 'HARMONIZATION_BLUR_SIGMA': 0.30,
 'HARMONIZATION_BLUR_STRENGTH': 0.08,
 'PERSON_TONE_FILTER_ENABLED': True,
 'PERSON_TONE_FILTER_STRENGTH': 0.08,
 'PERSON_TONE_FILTER_CORE_STRENGTH': 0.02,
 'PERSON_TONE_FILTER_MAX_COLOR_SHIFT': 5.0,
 'PERSON_TONE_FILTER_MAX_BRIGHTNESS_SHIFT': 3.0,
 'CONTACT_SHADOW_ENABLED': True,
 'CONTACT_SHADOW_OPACITY_NEAR': 46,
 'CONTACT_SHADOW_OPACITY_FAR': 16,
 'DRAW_INSERTION_GUIDE': True,
 'PERSON_GENERATION_ONLY_MODE': True,
 'CONTEXT_PERSON_GENERATION_PIPELINE': 'img2img',
 'PERSON_GENERATION_NEUTRAL_STRENGTH': 0.55,
 'PERSON_GENERATION_CONTEXT_DARKEN': 0.03,
 'INSERTION_GUIDE_ALPHA': 0.44,
 'INSERTION_GUIDE_BLUR': 0.25,
 'CONTEXT_CROP_EXPAND': 3.4,
 'CONTEXT_CROP_MIN_SIZE': 192,
 'CONTEXT_INPAINT_MASK_PADDING': 46,
 'COLOR_MATCH_PERSON_TO_SCENE': True,
 'COLOR_MATCH_STRENGTH': 0.45,
 'COLOR_MATCH_CONTEXT_PAD': 18,
 'FOREGROUND_HARMONIZATION_CORE_ALPHA': 0.02,
 'FOREGROUND_HARMONIZATION_EDGE_ERODE': 5,
 'TEXTURE_MATCH_PERSON_TO_SCENE': True,
 'TEXTURE_MATCH_STRENGTH': 0.06,
 'TEXTURE_MATCH_MIN_BLUR': 0.10,
 'TEXTURE_MATCH_MAX_BLUR': 0.28,
 'TEXTURE_MATCH_CONTEXT_PAD': 24,
 'PERSON_DETAIL_ENHANCE_ENABLED': True,
 'PERSON_DETAIL_SHARPNESS_BOOST': 1.32,
 'PERSON_DETAIL_CONTRAST_BOOST': 1.08,
 'PERSON_DETAIL_CORE_ERODE': 3,
 'EDGE_HALO_NEUTRALIZE': True,
 'EDGE_HALO_COLOR_MATCH_STRENGTH': 0.18,
 'EDGE_HALO_WIDTH': 1,
 'EDGE_HALO_MIN_ALPHA': 0.16,
 'EDGE_HALO_MAX_ALPHA': 0.72,
 'EDGE_LOCAL_BG_RADIUS': 7.0,
 'EDGE_HORIZON_BG_BLEND': 0.45,
 'EDGE_HORIZON_BAND_RATIO': 0.10,
 'EDGE_BG_CONTEXT_PAD': 18,
 'CONTEXT_TARGET_BBOX_EXPAND_FOR_DETECTION': 2.2,
 'OCCLUSION_AWARE_COMPOSITE': True,
 'OCCLUSION_MASK_BBOX_PADDING': 3,
 'OCCLUSION_MASK_BLUR_RADIUS': 1.2,
 'MIN_OCCLUDER_OVERLAP_RATIO': 0.03,
 'VEHICLE_OCCLUDER_MAX_FOOT_Y_DELTA': 18,
 'OCCLUDER_MIN_HEIGHT_RATIO': 0.35,
 'MAX_OCCLUSION_REMOVED_MASK_RATIO': 0.55}

VALIDATION_CONFIG = {'MIN_CORRECTED_PERSON_AREA_RATIO': 0.00045,
 'MIN_CORRECTED_ASPECT_RATIO': 1.35,
 'MAX_CORRECTED_ASPECT_RATIO': 3.6,
 'HEAD_MARGIN_RATIO': 0.18,
 'FOOT_MARGIN_RATIO': 0.12,
 'SIDE_MARGIN_RATIO': 0.2,
 'MIN_PERSON_CONF': 0.35,
 'MIN_PERSON_HEIGHT_RATIO': 0.06,
 'MAX_PERSON_HEIGHT_RATIO': 0.62,
 'MIN_PERSON_ASPECT_RATIO': 1.6,
 'MAX_PERSON_ASPECT_RATIO': 4.5,
 'MAX_BOTTOM_Y_RATIO': 0.92,
 'REJECT_IF_MASK_TOUCHES_BORDER': True,
 'MIN_GENERATED_HEIGHT_RATIO': 0.75,
 'MAX_GENERATED_HEIGHT_RATIO': 2.05,
 'PERSON_BORDER_REJECT_PIXELS': 3,
 'MIN_MASK_BBOX_HEIGHT_RATIO': 0.50,
 'STRICT_EARLY_PERSON_SCALE_FILTER': True,
 'FINAL_SCALE_VALIDATION_ENABLED': True,
 'MIN_ACCEPTED_SINGLE_HEIGHT_RATIO': 0.09,
 'MIN_ACCEPTED_NEAR_HEIGHT_RATIO': 0.16,
 'MIN_ACCEPTED_DISTANT_HEIGHT_RATIO': 0.08,
 'MIN_ACCEPTED_FOREGROUND_HEIGHT_RATIO': 0.12,
 'MIN_ACCEPTED_MASK_OPAQUE_RATIO': 0.24,
 'FINAL_MIN_SCALE_RATIO': 0.76,
 'FINAL_MAX_SCALE_RATIO': 1.85,
 'MIN_ACCEPTED_PERSPECTIVE_HEIGHT_RATIO_FAR': 0.08,
 'MIN_ACCEPTED_PERSPECTIVE_HEIGHT_RATIO_NEAR': 0.17,
 'CONTEXT_MIN_GENERATED_MASK_DIFF': 0.003,
 'CONTEXT_MIN_PERSON_TARGET_OVERLAP': 0.12,
 'CONTEXT_MIN_PERSON_MASK_AREA_RATIO': 0.00045,
 'MAX_MASK_OUTSIDE_INSERTION_RATIO': 0.36,
 'CONTEXT_MIN_FINAL_PERSON_DIFF': 0.01,
 'FINAL_COMPOSITE_MIN_MAE_255': 1.0,
 'FINAL_COMPOSITE_MIN_MAE_255_SEAMLESS': 0.75,
 'POST_PASTE_RETRY_MIN_MASK_AREA_RATIO': 0.0015}

RETRY_CONFIG = {'CONTEXT_PERSON_FALLBACK_TO_BBOX_INPAINT': False,
 'CONTEXT_GENERATION_RETRIES': 4,
 'EARLY_STOP_SCALE_UNRECOVERABLE_STREAK': 2,
 'DISTANT_EXTRA_RETRIES': 1,
 'SMALL_GROUP_EXTRA_RETRIES': 1,
 'NEAR_MAX_RETRIES': 2}

GENERATION_CONFIG = {'AUGMENTATION_STRENGTH': 0.72, 'GUIDANCE_SCALE': 7.2, 'NUM_INFERENCE_STEPS': 36}

DEBUG_CONFIG = {'DEBUG_SCALE_VISUALIZATION': False}

EDGE_HARMONIZATION_CONFIG = {
 'EDGE_HARMONIZATION_ENABLED': True,
 'EDGE_FEATHER_RADIUS': 1,
 'EDGE_BLUR_RADIUS': 0.25,
 'EDGE_COLOR_MATCH_STRENGTH': 0.06,
 'EDGE_BAND_WIDTH': 2,
 'POISSON_BLEND_ENABLED': False,
}

ADDIT_CONFIG = {
 'ADDIT_CONCEPT_ENABLED': True,
 'ADDIT_WEIGHTED_EXTENDED_ATTENTION': False,
 'ADDIT_STRUCTURE_TRANSFER': False,
 'ADDIT_SUBJECT_GUIDED_BLEND_PROXY': False,
 'ADDIT_BLEND_CONTEXT_DILATE': 0,
 'ADDIT_BLEND_EDGE_RADIUS': 0.6,
 'ADDIT_BLEND_SHADOW_EXTENSION': 0.22,
 'ADDIT_BLEND_SHADOW_BLUR': 5.0,
 'ADDIT_BLEND_SHADOW_ALPHA': 0.24,
}

CONFIG_GROUPS = [
    DATASET_CONFIG,
    PLACEMENT_CONFIG,
    SCALE_CONFIG,
    YOLO_EVAL_CONFIG,
    MASK_CONFIG,
    COMPOSITING_CONFIG,
    VALIDATION_CONFIG,
    RETRY_CONFIG,
    GENERATION_CONFIG,
    DEBUG_CONFIG,
    EDGE_HARMONIZATION_CONFIG,
    ADDIT_CONFIG,
]

def flatten_config(*groups):
    config = {}
    for group in groups:
        config.update(group)
    return config

def cfg(name, default=None):
    return EFFECTIVE_CONFIG.get(name, default)

if RUN_PRESET not in RUN_PRESETS:
    raise ValueError(f"Unknown RUN_PRESET={RUN_PRESET!r}. Choose one of {sorted(RUN_PRESETS)}.")

EFFECTIVE_CONFIG = flatten_config(USER_CONFIG, *CONFIG_GROUPS, RUN_PRESETS[RUN_PRESET], PARAMETER_OVERRIDES)
BASE_EFFECTIVE_CONFIG = dict(EFFECTIVE_CONFIG)
BASE_VARIANT_PROFILE = {name: dict(profile) for name, profile in VARIANT_PROFILE.items()}
globals().update(EFFECTIVE_CONFIG)

def looks_like_roboflow_citypersons_root(path):
    path = Path(path)
    return (
        (path / "train" / "images").exists()
        and ((path / "valid" / "images").exists() or (path / "val" / "images").exists())
    )


def looks_like_mot_sequence_root(path):
    path = Path(path)
    return (path / "img1").exists() and ((path / "gt").exists() or (path / "det").exists())


def normalize_dataset_root_candidate(path):
    path = Path(path)
    candidates = [path]
    if path.name == "img1":
        candidates.append(path.parent)
    if path.name == "images" and path.parent.name in {"train", "valid", "val", "test"}:
        candidates.append(path.parent.parent)
    if path.name in {"train", "valid", "val", "test"}:
        candidates.append(path.parent)
    for parent in path.parents:
        if parent.name.lower() in {"cityperson", "citypersons"}:
            candidates.append(parent)
            break
    for candidate in candidates:
        if looks_like_roboflow_citypersons_root(candidate) or looks_like_mot_sequence_root(candidate):
            return candidate
    return path


def resolve_dataset_root(candidates):
    for path in candidates:
        root = normalize_dataset_root_candidate(path)
        if looks_like_roboflow_citypersons_root(root) or looks_like_mot_sequence_root(root):
            return Path(root)
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for path in sorted(kaggle_input.rglob("data.yaml")):
            root = path.parent
            if looks_like_roboflow_citypersons_root(root):
                return root
    return next((Path(path) for path in candidates if Path(path).exists()), Path(candidates[0]))


# Derived paths.
DATASET_ROOT = resolve_dataset_root(DATASET_ROOT_CANDIDATES)
IMAGE_ROOT = DATASET_ROOT
LABEL_ROOT = DATASET_ROOT
if looks_like_mot_sequence_root(DATASET_ROOT):
    VALID_SPLIT_NAME = "test"
    DATASET_SPLIT_DIRS = {"test": DATASET_ROOT / "img1"}
    LABEL_SPLIT_DIRS = {"test": DATASET_ROOT / "gt"}
else:
    VALID_SPLIT_NAME = "valid" if (DATASET_ROOT / "valid" / "images").exists() else "val"
    DATASET_SPLIT_DIRS = {
        "train": DATASET_ROOT / "train" / "images",
        "val": DATASET_ROOT / VALID_SPLIT_NAME / "images",
        "test": DATASET_ROOT / "test" / "images",
    }
    LABEL_SPLIT_DIRS = {
        "train": DATASET_ROOT / "train" / "labels",
        "val": DATASET_ROOT / VALID_SPLIT_NAME / "labels",
        "test": DATASET_ROOT / "test" / "labels",
    }
METRICS_DIR = Path("/kaggle/working/metrics")
METRICS_CSV_PATH = METRICS_DIR / "augmentation_metrics.csv"
METRICS_SUMMARY_PATH = METRICS_DIR / "augmentation_metrics_summary.csv"
METRICS_PLOT_PATH = METRICS_DIR / "augmentation_metrics_by_variant.png"
PATCH_DEBUG_DIR = OUTPUT_DIR / "patch_debug"
EDGE_DEBUG_DIR = OUTPUT_DIR / "edge_harmonization_debug"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BASE_CAPTION = "urban street photo"
PRESERVATION_PROMPT = "clear full body, separated silhouette, grounded feet, natural scale"

SCENE_PROMPTS = {
    "urban_pedestrian_scene": "urban street photo",
}

VARIANT_PROMPTS = {
    "add_single_pedestrian": "one clear full-body pedestrian",
    "add_two_pedestrians": "two separate full-body pedestrians with visible gap",
    "add_small_group": "three separate full-body pedestrians with visible gaps",
    "add_occluded_pedestrian": "partly occluded full-body pedestrian behind foreground object",
    "add_distant_pedestrian": "distant clear full-body pedestrian",
    "add_near_pedestrian": "near larger full-body pedestrian",
}

NEGATIVE_PROMPT = "cropped, missing head, missing legs, thin body, giant, closeup, floating, ghost, bad perspective, hard seam, overlap, merged people, fused bodies"


def ensure_output_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PATCH_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    EDGE_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "output_dir": OUTPUT_DIR,
        "metrics_dir": METRICS_DIR,
        "patch_debug_dir": PATCH_DEBUG_DIR,
        "edge_debug_dir": EDGE_DEBUG_DIR,
    }
