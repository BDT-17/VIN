# Model Selection For Background-Preserving Object Insertion

## Selected Model

**Stable Diffusion 3.5 Medium (BF16/FP16)** is the current generation model for this background-preserving object insertion pipeline.

The model is used for **inference/img2img augmentation**. It generates candidate objects in scene context, then a detector/segmenter extracts only the target object pixels for compositing back onto the original image.

The current reference setup uses YOLOv8m-seg for pedestrian extraction, but the model-selection criteria are broader than CityPersons:

- candidate object realism;
- prompt adherence;
- controllable img2img behavior;
- stable generation under retry;
- practical runtime on Kaggle T4 16GB;
- compatibility with detector-guided segmentation and validation;
- ability to work as one stage in a larger augmentation pipeline, not as a full-scene image replacement model.

## 1. Task Context

The project is not a free text-to-image generator. It is a dataset augmentation pipeline:

```text
source image + target object instruction
-> generate candidate object in context
-> extract target object only
-> preserve original background
-> validate and log the augmented sample
```

This changes model selection. The best model is not necessarily the most photorealistic open-ended generator. The best model is the one that produces controllable candidate objects often enough, within the memory and retry budget, while fitting a segmentation-composite workflow.

## 2. Why SD3.5 Medium BF16/FP16

### 2.1 Strong Enough For Object Candidates

SD3.5 Medium is strong enough to generate realistic candidate objects in local scene context. For the current pedestrian reference task, it can generate full-body people with plausible clothing, lighting, and perspective when prompt, strength, guidance, and retry policy are tuned.

For other object classes, the same advantage applies when the object is visually common enough for the base model and the prompt can describe it clearly.

### 2.2 Practical On Kaggle T4

The pipeline has to run more than a single image generation call. A full augmentation run may include:

- SD3.5 img2img;
- detector/segmenter inference;
- optional depth estimation;
- compositing and edge harmonization;
- validation;
- adaptive retry;
- debug and manifest output.

SD3.5 Medium is a practical compromise for Kaggle T4 16GB when combined with:

- FP16 inference;
- T5 disabled by default;
- model CPU offload;
- VAE slicing/tiling;
- attention slicing;
- resolution 512;
- small per-sample batches.

### 2.3 Better Fit Than Heavier Models For Batch Augmentation

Heavier models can produce better individual images, but batch data augmentation needs throughput, repeatability, and integration with validation.

The current choice favors:

- many attempts under controlled cost;
- stable retry behavior;
- easier notebook deployment;
- enough quality for detector-training data;
- compatibility with object-only composite.

### 2.4 FP16/BF16 Preferred Over NF4 For This Baseline

NF4 or 4-bit inference can reduce VRAM, but the baseline prioritizes stable output quality and consistent img2img behavior.

FP16/BF16 is preferred because:

- outputs are generally more stable;
- quantization artifacts are less likely;
- comparisons across prompt/strength/guidance are cleaner;
- edge and texture artifacts are easier to diagnose.

NF4 remains a fallback option if the target environment is more constrained.

## 3. Why The Pipeline Does Not Trust Generated Background

Even when a model follows the prompt, img2img can alter the source image outside the intended object area. For augmentation, those edits can corrupt labels and change the data distribution.

The architecture avoids this by design:

1. generate candidate image or context crop;
2. segment target object pixels;
3. discard generated background;
4. paste only the target object into the original image;
5. validate background preservation and object detectability.

This is the key reason the model does not need to be perfect at background preservation by itself.

## 4. Comparison With Alternatives

| Model | Why it is not the default |
|---|---|
| SD3 Medium BF16 | Older than SD3.5 Medium, weaker prompt adherence and generation quality. |
| SD3 Medium NF4 | Useful for low VRAM, but less stable for quality-focused img2img augmentation. |
| SD3.5 Medium NF4 | Possible fallback, but FP16/BF16 is preferred for baseline quality and consistency. |
| SD3.5 Large Turbo | Larger and less ideal for a retry-heavy notebook pipeline with segmentation and metrics. |
| FLUX.1-dev FP16 | Very strong image quality, but heavier runtime/VRAM and less practical for this current Kaggle batch workflow. |
| FLUX FP8/NF4 | Useful for experiments, but adds workflow complexity and has not replaced SD3.5 Medium for the current object-insertion pipeline. |

## 5. Generalization Beyond Pedestrians

Changing the object class does not automatically require changing the base generation model.

Before changing models, update:

- prompt variants;
- detector/segmenter;
- object placement rules;
- scale/anchor rules;
- validation metrics;
- rejection thresholds.

Only consider switching models when the current generator systematically fails to produce the target object class, even after prompt and pipeline tuning.

## 6. Conclusion

SD3.5 Medium BF16/FP16 is the current default because it balances generation quality, controllability, memory use, and notebook practicality.

For this project, the architecture matters as much as the generator: background preservation is achieved by **object-only segmentation and compositing**, not by trusting the diffusion model to leave the whole scene unchanged.
