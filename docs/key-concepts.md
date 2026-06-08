# Key Concepts

This project builds a research pipeline for pedestrian augmentation on CityPersons-style street scenes. The working source of truth is:

```text
sd3.5-agumentation-scale-correction-clean.ipynb
```

## 1. Background-Preserving Augmentation

The original image is treated as the trusted scene. SD3.5 is allowed to generate candidate pedestrians, but the generated background is not trusted.

The pipeline therefore:

1. generates a candidate image;
2. detects and segments generated pedestrians;
3. extracts only pedestrian pixels;
4. pastes them back onto the original frame.

This keeps buildings, roads, signs, vehicles, and scene layout closer to the source data distribution.

## 2. Context-Person Composite

The core architecture is **Context-Person Composite**.

Instead of asking the diffusion model to solve the whole augmentation problem, the notebook separates the task into explicit stages:

```text
placement -> generation -> segmentation -> scale correction -> compositing -> validation -> retry
```

This makes the pipeline easier to debug because each failure can be assigned to a concrete reason such as bad scale, no person detected, low confidence, poor edge quality, or invalid overlap.

## 3. Foot-Anchored Scale Correction

Generated pedestrians often have plausible appearance but wrong physical scale. The notebook corrects this after generation.

The important idea is:

- estimate the expected person height from ground/foot position;
- resize the detected pedestrian crop;
- keep the foot point anchored during resize;
- validate that the final person is not floating or implausibly tall/short.

This is especially important for CityPersons because pedestrian height strongly depends on perspective depth.

## 4. Occlusion-Aware Placement

Street scenes frequently contain foreground vehicles, poles, signs, and other pedestrians. A new person should not simply be pasted on top of everything.

The clean notebook includes occlusion-aware logic:

- foreground occluder masks can remove hidden parts of the generated person;
- vehicle/person overlap is checked before acceptance;
- person-person overlap is allowed only when the occluded person is smaller and plausibly behind;
- invalid depth ordering is rejected as `bad_person_depth_overlap`.

This is a key difference between a visual demo and a detection-dataset augmentation pipeline.

## 5. Edge And Appearance Integration

The project uses multiple compositing steps to reduce pasted-looking artifacts:

- mask cleanup;
- edge feathering;
- color harmonization;
- local brightness matching;
- edge halo color matching;
- shadow synthesis;
- `seamlessClone` when the mask and scene support it;
- alpha fallback when seamless cloning is not reliable.

The goal is not perfect image editing. The goal is to produce training images where the added pedestrian is realistic enough and does not corrupt the original annotation context.

## 6. YOLO-Guided Validation

YOLOv8m-seg is used as a practical validator and extractor.

It supports:

- person mask extraction;
- confidence filtering;
- expected person count checks;
- final augmented image validation;
- rejection reason logging;
- manifest metadata.

The pipeline does not assume every generated image is useful. It rejects weak samples and retries under a controlled budget.

## 7. Quality Score

The clean notebook computes a compact quality score:

```python
quality_score = (
    0.45 * person_score
    + 0.25 * scale_score
    + 0.20 * background_score
    + 0.10 * edge_score
)
```

The component scores reflect:

- **person_score**: confidence and detectability of the inserted pedestrian;
- **scale_score**: perspective-aware size plausibility;
- **background_score**: preservation of the original scene;
- **edge_score**: visual integration around the pasted mask.

This score is used for analysis and autotune guidance, not as a claim of absolute image quality.

## 8. Quality-Guided Autotune

Autotune reduces manual parameter tuning but is intentionally conservative.

Current safeguards:

- default mode is `dry_run=True`;
- minimum accepted sample count is 30;
- recommended changes are bounded by `max_adjustment_ratio`;
- updates are softened through `aggressiveness`;
- before/after config snapshots are saved automatically;
- runtime defaults can be restored with `reset_runtime_config()`.

The intended workflow is:

1. run a smoke or quality batch;
2. inspect quality summary, accept rate, and reject histogram;
3. review the autotune report;
4. decide whether to apply the recommended runtime changes.

## 9. Multi-GPU Execution

The notebook can shard augmentation jobs across available CUDA devices.

When `USE_ALL_GPUS_FOR_AUGMENTATION=True`, the device resolver returns all visible CUDA devices, for example:

```python
["cuda:0", "cuda:1"]
```

If only one GPU exists, it behaves like a normal single-GPU run. If CUDA is unavailable, it falls back to CPU.

## 10. Reproducibility Principle

Autotune is useful for exploration, but it can reduce reproducibility if changes are applied silently.

For research runs:

- keep `dry_run=True` while exploring;
- save and inspect autotune snapshots;
- record `RUN_PRESET`, `PARAMETER_OVERRIDES`, seed, manifest, and snapshot path;
- only apply autotune changes intentionally.

This keeps experiment reports traceable.

## 11. Current Limitations

The notebook is still large because it keeps the full research pipeline inside one file.

Known limitations:

- some visual quality issues still need manual inspection;
- YOLO-based validation is practical but not a perfect human-quality metric;
- smoke tests can be noisy and should not be over-trusted;
- threshold tuning should be based on enough samples and saved snapshots;
- the notebook is suitable for experimentation but may later benefit from moving helper code into a small Python module.

## 12. Research Focus

The current research focus is not changing the base model. The main work is improving:

- placement policy;
- perspective scale;
- foot grounding;
- occlusion ordering;
- blending and harmonization;
- retry policy;
- quality metrics;
- reproducible parameter tuning.
