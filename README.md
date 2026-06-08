# CityPersons Pedestrian Augmentation With SD3.5

Research notebook for generating pedestrian augmentations on CityPersons-style street scenes using **Stable Diffusion 3.5 Medium**, **YOLOv8m-seg**, and a background-preserving compositing pipeline.

The current working entrypoint is:

```text
sd3.5-agumentation-scale-correction-clean.ipynb
```

Older notebooks are treated as backups/archive material. The clean notebook is the version that contains the current pipeline, configuration layout, quality scoring, autotune flow, and multi-GPU augmentation support.

## Goal

The project augments pedestrian detection data by inserting realistic new pedestrians into existing street-scene images while preserving the original background geometry.

This is not a prompt-only image generation workflow. SD3.5 is used to generate candidate people, then the pipeline extracts only the generated pedestrian pixels and composites them back onto the original frame.

## Main Pipeline

```text
CityPersons / YOLO records
        |
        v
Placement candidate selection
        |
        v
SD3.5 img2img / inpaint candidate generation
        |
        v
YOLOv8m-seg person extraction
        |
        v
Scale correction + grounding checks
        |
        v
Edge blending + color/brightness harmonization
        |
        v
Occlusion-aware compositing
        |
        v
YOLO validation + retry policy
        |
        v
Augmented image + comparison pair + manifest row
```

## What Is New In The Clean Notebook

- SDXL code has been removed; the pipeline is SD3.5-only.
- The long augmentation section has been split into smaller notebook sections.
- Config is grouped through default config dictionaries, run presets, and short `PARAMETER_OVERRIDES`.
- Smoke/quality/batch presets are supported through `RUN_PRESETS`.
- Scale correction is applied after generation, using detected person geometry and foot anchoring.
- Person-person overlap is depth-aware: a person may overlap another person only when the occluded person is smaller and plausibly behind.
- Compositing includes edge blending, color harmonization, local brightness matching, shadow synthesis, and foreground occlusion handling.
- Quality scoring combines person, scale, background, and edge scores:

```python
quality_score = (
    0.45 * person_score
    + 0.25 * scale_score
    + 0.20 * background_score
    + 0.10 * edge_score
)
```

- Quality-guided autotune reports recommended runtime parameter changes.
- Autotune now defaults to dry-run behavior and saves before/after config snapshots.
- Runtime config can be reset from the reset cell.
- Multi-GPU augmentation is enabled when multiple CUDA devices are available.

## Notebook Sections

The notebook keeps the experiment flow in this order:

```text
1. Install
2. Runtime Check
3. HF Login
4. Configuration
5. Imports / Prompts
6. Dataset Scanner
7. Preview
8. Image Preprocessing
10. Img2Img Augmentation Pipeline
11. Recommended First Run
12. Metrics
```

Section 10 is internally split into smaller blocks for pipeline loading, placement, masks, scale correction, generation, YOLO evaluation, retry policy, manifest writing, autotune, and runtime reset.

## Configuration Pattern

The notebook uses:

```python
USER_CONFIG
RUN_PRESETS
DATASET_CONFIG
PLACEMENT_CONFIG
SCALE_CONFIG
YOLO_EVAL_CONFIG
MASK_CONFIG
COMPOSITING_CONFIG
VALIDATION_CONFIG
RETRY_CONFIG
GENERATION_CONFIG
DEBUG_CONFIG
PARAMETER_OVERRIDES
```

The intended workflow is:

1. Choose `RUN_PRESET`.
2. Keep default config stable.
3. Put short one-off changes in `PARAMETER_OVERRIDES`.
4. Run a smoke test.
5. Review the autotune report and snapshot.
6. Apply changes manually or rerun autotune with dry-run disabled.

## Autotune Behavior

Autotune is intentionally conservative.

- Default mode is dry-run.
- Minimum accepted sample count is 30.
- Recommended changes are bounded by `max_adjustment_ratio`.
- Changes are blended through `aggressiveness`.
- Snapshot JSON files are saved in `autotune_snapshots/`.
- The reset cell can restore the notebook defaults during the current runtime.

To inspect recommendations without applying them:

```python
autotune_report = autotune_from_last_run(apply=True, dry_run=True)
```

To apply runtime changes after review:

```python
autotune_report = autotune_from_last_run(apply=True, dry_run=False)
```

To restore runtime defaults:

```python
reset_runtime_config()
```

## Outputs

Typical outputs include:

- augmented images;
- side-by-side comparison images;
- optional patch/debug images;
- manifest metadata;
- rejection histograms;
- quality metrics;
- autotune snapshots.

## Core Stack

```text
Dataset format: CityPersons-style / YOLO labels
Generator: Stable Diffusion 3.5 Medium
Segmenter: YOLOv8m-seg
Default resolution: 448x448
Primary runtime: Kaggle GPU
Multi-GPU: automatic CUDA device sharding when available
```

## Current Status

This is an active research baseline, not a finalized production augmentation system.

The current bottlenecks are compositing quality, geometry, scale, occlusion ordering, validation policy, and reproducibility. The clean notebook is now the source of truth for continued experiments.
