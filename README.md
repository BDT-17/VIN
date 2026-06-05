# CityPersons Pedestrian Augmentation With SD3.5

This repository contains the V3 Context-Person Composite pipeline for generating pedestrian augmentations on CityPersons street-scene images using Stable Diffusion 3.5 Medium and YOLOv8m segmentation.

The key design choice is architectural separation: SD3.5 is allowed to generate candidate pedestrians, but the final image always keeps the original background. Only the segmented pedestrian pixels are composited back into the source frame.

## Files

- `sd3.5-agumentation.ipynb` - Main Kaggle notebook for generation, segmentation, compositing, validation, logging, and metrics.
- `report-5-6-26.tex` - LaTeX source for the technical report.
- `CityPersons_Report.pdf` - Compiled technical report.

## Pipeline Summary

The V3 pipeline runs the following stages for each sample:

1. Smart placement selects a target region using road Y-range, perspective scale, placement slots, jitter, and overlap constraints.
2. A faint human insertion guide is painted at the target location.
3. SD3.5 Medium runs img2img at `448x448`, with T5 disabled for VRAM efficiency.
4. YOLOv8m-seg segments candidate pedestrians at confidence `0.12`.
5. Detection and mask quality filters reject weak, cropped, ghost-like, misplaced, or wrongly scaled candidates.
6. Adaptive retry changes strength, guidance, prompt terms, and context based on the reject reason.
7. The accepted person mask is pasted onto the original image, preserving the original background.
8. Appearance harmonisation adjusts color, brightness, contrast, saturation, noise, sharpness, edge halo, and contact shadow.
9. The notebook saves the final composite, comparison image, debug strip, manifest, and metrics.

## Core Configuration

```text
Pipeline: Context-Person Composite (V3)
Backbone: Stable Diffusion 3.5 Medium
Segmenter: YOLOv8m-seg
Hardware: Kaggle T4 x2
Resolution: 448x448
Dataset: CityPersons in YOLO format
```

Important settings:

```python
MODEL_BACKEND = "sd35"
SD35_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
BACKGROUND_PRESERVATION_MODE = "context_person_composite"
CONTEXT_PERSON_SEGMENTATION_MODEL = "yolov8m-seg.pt"
CONTEXT_PERSON_MIN_CONFIDENCE = 0.12
CONTEXT_GENERATION_RETRIES = 3
EDGE_HALO_NEUTRALIZE = True
FINAL_SCALE_VALIDATION_ENABLED = True
```

## Augmentation Variants

The notebook supports six generation variants:

| Variant | Strength | Guidance | Steps | Weight |
|---|---:|---:|---:|---:|
| Single pedestrian | 0.72 | 6.8 | 36 | 0.24 |
| Two pedestrians | 0.74 | 6.9 | 36 | 0.18 |
| Small group | 0.76 | 7.0 | 38 | 0.16 |
| Occluded pedestrian | 0.74 | 6.8 | 36 | 0.18 |
| Distant pedestrian | 0.68 | 6.6 | 34 | 0.14 |
| Near pedestrian | 0.76 | 7.3 | 38 | 0.10 |

Variant-specific confidence thresholds are used to avoid over-rejecting difficult cases:

```python
MIN_PERSON_CONF_BY_VARIANT = {
    "add_single_pedestrian": 0.30,
    "add_two_pedestrians": 0.25,
    "add_small_group": 0.22,
    "add_occluded_pedestrian": 0.22,
    "add_distant_pedestrian": 0.18,
    "add_near_pedestrian": 0.32,
}
```

## Quality Filters

The report highlights these quality gates:

- Early scale gate on YOLO detection: height ratio `[0.68, 1.32]`.
- Final scale gate after paste: height ratio `[0.76, 1.30]`.
- Per-variant minimum height ratios: single `0.075`, near `0.160`, distant `0.065`.
- Ghost-person contrast: mean pixel difference under mask must exceed `12.0` on a `0-255` scale.
- Minimum segmented mask area: `0.00045` of the image.
- Final composite contrast threshold: `0.010`.
- Vertical band coverage above `0.24` to ensure head, torso, and legs are present.
- Person-mask aspect ratio `[1.35, 5.2]`.
- Border contact rejected if mask is within `6 px` of the image edge.
- Mask outside detection bbox must stay below `0.22`.
- Multi-person variants must produce the requested count.

## Adaptive Retry

The notebook retries up to 3 times per sample. Retry behavior depends on rejection reason:

- Ghost or low contrast: increase strength and guidance, add prompts for solid visible silhouettes.
- Not enough persons: increase strength and guidance, emphasize requested count and separated silhouettes.
- Too large for perspective: decrease strength and guidance, add mid-distance wording.
- Cropped body: decrease strength and expand context, require full body visibility.

Samples that exhaust all retries are discarded.

## Smoke Test Result

The updated report documents a 10-image smoke test using multi-person variants only:

```text
Accepted outputs: 4 / 10
Acceptance rate: 40%
```

Accepted examples:

- `bremen_000209` two-pedestrian
- `aachen_000123` two-pedestrian
- `bremen_000109` two-pedestrian
- `bremen_000186` small-group

Dominant reject reasons:

- `low_person_conf`
- `too_large_for_perspective`
- `floating_or_bad_ground`

The report identifies a linked failure chain: scale retry reduces strength and guidance after an over-large person, then later attempts fall below the confidence threshold.

## Technical Findings

The report’s main findings are:

- Background preservation cannot be guaranteed by prompting alone; compositing only person pixels onto the original background is the reliable solution.
- A loose scale envelope with a faint insertion guide works better than strict bounding-box constraints.
- Ghost detection through pixel contrast is required, otherwise transparent pedestrians can be silently accepted.
- Reason-specific retry is better than fixed escalation.
- Multi-person count validation is necessary; generating one person for a two-person prompt is a scenario failure, not an acceptable fallback.

## Next Steps

Recommended next experiments:

1. Run a 200-image production batch across all six variants on the train split.
2. Inspect debug strips for ghost, scale, and multi-person failures.
3. If `low_person_conf` exceeds 30% after scale retry, widen confidence on later retry attempts, for example accept `0.28` instead of `0.35`.
4. If multi-person count failure exceeds 40%, revisit small-group insertion guide geometry.
5. Enable and review post-hoc metrics: SSIM, PSNR, histogram distance, brightness delta, and contrast delta per variant.

Target quality gate:

```text
Acceptance rate >= 80%
Background SSIM >= 0.86
```

If acceptance remains below 60% after parameter sweeps, the report recommends continuing with pipeline-level refinement: confidence scheduling, retry policy tuning, insertion-guide geometry, and stricter variant-wise quality analysis.

## How To Run

1. Open `sd3.5-agumentation.ipynb` in Kaggle.
2. Enable GPU, preferably dual T4.
3. Attach the CityPersons YOLO-format dataset.
4. Add Hugging Face secret `HF_TOKEN` if SD3.5 access requires authentication.
5. Run cells from top to bottom.
6. Review outputs under:

```text
/kaggle/working/sd35_citypersons_augmented
```

Important output folders/files include:

- final augmented images
- side-by-side comparison images
- five-panel debug strips for early samples
- manifest CSV
- metric CSV and summaries

## Recommendation

Use this notebook as the current V3 research baseline, not as a fully validated production generator yet. The architecture is strong and solves background preservation correctly, but the latest smoke test shows that acceptance rate is still 40% on difficult multi-person cases. The next milestone is a 200-image run with parameter refinement toward the report’s `80%` acceptance and `0.86` background SSIM targets.
