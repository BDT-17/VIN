# CityPersons Pedestrian Augmentation With SD3.5

Research repo for generating pedestrian augmentations on CityPersons street-scene images using **Stable Diffusion 3.5 Medium**, **YOLOv8m-seg**, and a **Context-Person Composite** pipeline.

The central idea is to let SD3.5 generate candidate pedestrians, then discard generated background pixels and paste only segmented pedestrian pixels back onto the original frame.

## Files

- `sd3.5-agumentation.ipynb` - Original augmentation notebook kept as backup.
- `sd3.5-agumentation-scale-correction.ipynb` - Current research notebook with scale correction, blending, retry policy, and metadata logging.
- `docs/key-concepts.md` - Short project concept note.
- `docs/model-selection.md` - Model choice notes.
- `docs/report-5-6-26.tex` - Technical report source.
- `docs/CityPersons_Report.pdf` - Compiled report.

## Key Concepts

- **Background preservation**: the source image remains the final background; generated background is not trusted.
- **Context-Person Composite**: generation, segmentation, scale correction, and compositing are separated into explicit stages.
- **Perspective-aware scale correction**: detected pedestrians are resized after generation based on foot/ground position.
- **Foot-anchored placement**: resized pedestrians keep their foot point fixed to reduce floating and grounding errors.
- **YOLO mask extraction**: YOLOv8m-seg detects and segments generated pedestrians before compositing.
- **Edge integration**: mask cleanup, harmonization, seamlessClone, alpha fallback, and foreground preservation reduce pasted-looking edges.
- **Adaptive retry**: failed attempts are retried based on reason, such as low confidence, missing people, bad scale, floating, or weak final composite.
- **Multi-person validation**: two-person and small-group variants use per-person correction and count checks.

## Core Stack

```text
Dataset: CityPersons, YOLO format
Generator: Stable Diffusion 3.5 Medium
Segmenter: YOLOv8m-seg
Resolution: 448x448
Pipeline: Context-Person Composite
Runtime: Kaggle GPU
```

## Research Status

This is an active research baseline. Current work focuses on compositing quality, perspective scale, foot grounding, retry policy, and edge blending rather than changing the base generative model.
