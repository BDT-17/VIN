# Key Concepts

This project studies a general data augmentation task:

```text
add target object(s) to existing images while preserving the original background
```

The current implementation uses CityPersons pedestrian insertion as the reference experiment. Pedestrians are a hard and useful test case, but they are not the conceptual boundary of the project.

## 1. Background-Preserving Augmentation

The original image is treated as the trusted scene. The diffusion model may generate candidate objects, but its generated background is not trusted.

The pipeline therefore:

1. generates a candidate image or context crop;
2. detects and segments the newly generated target object;
3. extracts only target-object pixels;
4. pastes those pixels back onto the original frame;
5. validates that the output changed only where the new object should appear.

This keeps roads, buildings, shelves, tables, signs, vehicles, vegetation, or any other original scene context closer to the source data distribution.

## 2. Context-Object Composite

The core architecture is **Context-Object Composite**.

Instead of asking the diffusion model to both create the object and preserve every background pixel, the pipeline separates the task into explicit stages:

```text
placement -> generation -> segmentation -> scale correction -> compositing -> validation -> retry
```

In the current code this mode is still named `context_person_composite`, because the first benchmark is pedestrian insertion. The design generalizes to other object classes when the detector, prompt set, placement policy, and validation rules are swapped.

## 3. Object-Aware Scale Correction

Generated objects often look plausible but have the wrong physical scale. The pipeline corrects this after generation when a usable object mask exists.

For pedestrians, the implementation estimates expected height from depth/ground position, resizes the crop, keeps the feet anchored, and rejects objects that are implausibly tall, short, floating, or cropped.

For other object categories, the same concept should be adapted:

- estimate expected size from perspective, depth, known reference objects, or domain priors;
- anchor the object to the right physical contact point, such as feet, wheels, base, shadow, or table contact;
- resize the segmented object and mask together;
- reject outputs outside the plausible size envelope.

## 4. Placement Policy

Object insertion needs a placement policy, not just a prompt.

The current pedestrian implementation chooses candidate boxes using road/sidewalk priors, existing person/vehicle boxes, semantic masks, depth ordering, and overlap checks.

For another dataset, placement should encode domain-specific validity:

- cars belong on roads or parking areas;
- products belong on shelves, tables, or hands;
- traffic signs belong near poles or roadside regions;
- animals need plausible ground contact and occlusion;
- medical or industrial objects may need strict anatomical or mechanical constraints.

Bad placement can corrupt a training dataset even when the inserted object looks visually realistic.

## 5. Occlusion-Aware Compositing

The inserted object should respect foreground objects in the original image.

The current pipeline can build occluder masks from existing persons and vehicles, then remove hidden regions from the new object before final paste. The general rule is:

```text
new object should not blindly cover trusted foreground evidence
```

For a new domain, foreground classes and depth rules need to be redefined.

## 6. Edge And Appearance Integration

The project uses multiple compositing steps to reduce pasted-looking artifacts:

- mask cleanup;
- generated-background fringe trimming;
- edge feathering;
- local color transfer;
- brightness and contrast matching;
- saturation matching;
- texture/noise matching;
- local boundary color matching;
- optional contact shadow;
- alpha paste as the current default;
- optional seamless clone support, disabled by default because it can create halo artifacts.

The goal is not perfect photo editing. The goal is to produce training images where the added object is realistic enough and does not corrupt the original annotation context.

## 7. Detector-Guided Validation

A detector/segmenter is used as both extractor and validator.

In the reference implementation, YOLOv8m-seg extracts pedestrian masks and checks confidence, count, geometry, mask area, and final composite quality.

For a new object class, replace this with the best available validator:

- class-specific detector;
- segmentation model;
- open-vocabulary detector;
- domain classifier;
- geometric rules;
- downstream task metrics.

The pipeline does not assume every generated image is useful. It rejects weak samples and retries under a controlled budget.

## 8. Quality Score

The clean notebook computes a compact quality score:

```python
quality_score = (
    0.45 * object_score
    + 0.25 * scale_score
    + 0.20 * background_score
    + 0.10 * edge_score
)
```

In the current code, `object_score` is named `person_score`.

The component scores reflect:

- **object_score**: confidence and detectability of the inserted object;
- **scale_score**: size plausibility;
- **background_score**: preservation of the original scene;
- **edge_score**: visual integration around the pasted mask.

This score is used for analysis and autotune guidance, not as a claim of absolute image quality.

## 9. Quality-Guided Autotune

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

## 10. Multi-GPU Execution

The notebook can shard augmentation jobs across available CUDA devices.

When `USE_ALL_GPUS_FOR_AUGMENTATION=True`, the device resolver returns all visible CUDA devices, for example:

```python
["cuda:0", "cuda:1"]
```

If only one GPU exists, it behaves like a normal single-GPU run. If CUDA is unavailable, it falls back to CPU.

## 11. Reproducibility Principle

Autotune is useful for exploration, but it can reduce reproducibility if changes are applied silently.

For research runs:

- keep `dry_run=True` while exploring;
- save and inspect autotune snapshots;
- record `RUN_PRESET`, `PARAMETER_OVERRIDES`, seed, manifest, and snapshot path;
- only apply autotune changes intentionally.

This keeps experiment reports traceable.

## 12. Current Limitations

The codebase still contains pedestrian-specific names because CityPersons is the active reference experiment.

Known limitations:

- new object classes require detector, prompt, placement, and validation changes;
- visual quality still needs manual inspection;
- detector-based validation is practical but not a perfect human-quality metric;
- smoke tests can be noisy and should not be over-trusted;
- threshold tuning should be based on enough samples and saved snapshots;
- the notebook is suitable for experimentation but should be modularized further for a general object-insertion toolkit.

## 13. Research Focus

The current research focus is improving the pipeline architecture rather than changing the base model.

Main work areas:

- target-object placement policy;
- perspective and object scale;
- object grounding/contact;
- occlusion ordering;
- mask quality;
- blending and harmonization;
- detector-guided retry policy;
- quality metrics;
- reproducible parameter tuning;
- adapting the pipeline beyond the pedestrian/CityPersons reference case.
