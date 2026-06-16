# Pipeline Version History

This document records how the pipeline evolved toward the current goal:

```text
build a data augmentation pipeline that adds target objects to images while preserving the original background
```

CityPersons pedestrian insertion is the reference experiment, but the lessons apply to object insertion more generally.

---

## Executive Summary

The project went through five major pipeline versions:

- **V1 Full Image Img2Img**: simple and capable of adding objects, but rewrites too much of the background.
- **V2 Patch Blend**: preserves most of the image, but object generation inside small patches is unreliable.
- **V3 Human/Object-Shaped Mask Inpaint**: constrains edits, but overly tight masks do not give the model enough room.
- **V4 BBox Inpaint**: gives the model more canvas, but scale, color, and pasted-boundary artifacts remain.
- **V5 Context-Object Composite**: current direction. Generate in context, segment only the new object, discard generated background, then composite onto the original image.

The main conclusion is:

```text
do not ask one diffusion pass to both create the object and preserve the dataset image.
Generate the object, segment it, and keep the original background as source of truth.
```

---

## 1. Problem Definition

The task is not free image editing. It is augmentation for downstream datasets.

The pipeline must:

- add target object(s) to existing images;
- preserve the original background and layout;
- keep the inserted object realistic enough for model training;
- maintain plausible scale, placement, grounding, and occlusion;
- reject weak generations instead of saving every output;
- record enough metadata to audit the augmentation run.

In the current implementation, the target object is a pedestrian and the reference dataset is CityPersons. For a different task, replace the detector, placement policy, prompts, scale policy, and validation rules.

---

## 2. V1 - Full Image Img2Img

### Architecture

```text
Source image
-> Resize / crop
-> SD3.5 img2img over the whole image
-> Save augmented image
```

### What Worked

- Simple to implement.
- The model can generate visually plausible objects.
- No mask or composite logic needed.

### What Failed

- The background changes globally.
- Roads, buildings, vehicles, lighting, and camera viewpoint can drift.
- The output may become a new image rather than the original image plus one object.

### Lesson

Prompting the model to "preserve background" is not enough. Background preservation must be enforced architecturally.

---

## 3. V2 - Patch Blend

### Architecture

```text
Source image
-> Select insertion patch
-> SD3.5 img2img on patch
-> Paste patch or insertion area back
```

### What Worked

- Pixels outside the patch are preserved.
- The changed region is smaller and easier to inspect.

### What Failed

- Small patches often lack enough context for reliable object generation.
- The model may only beautify texture instead of adding a clear object.
- Patch boundaries can become visible.
- Object scale is still weakly controlled.

### Lesson

Restricting the edit area helps, but generation quality suffers when the model does not get enough semantic context.

---

## 4. V3 - Tight Object-Shaped Inpaint

### Architecture

```text
Source image
-> Choose insertion region
-> Create tight object-shaped mask
-> Inpaint inside mask
-> Composite masked pixels back
```

### What Worked

- Background outside the mask is preserved.
- Inpainting is closer to the real task than full-image img2img.

### What Failed

- A tight human/object mask gives the model too little room.
- Generated objects can look like shadows, ghosts, or partial bodies.
- The model cannot create natural boundaries, clothing, accessories, or contact shadows.

### Lesson

A more detailed mask is not always better. Diffusion models often need extra canvas around the object.

---

## 5. V4 - BBox Inpaint

### Architecture

```text
Source image
-> Choose insertion bbox
-> Create larger rounded/bbox mask
-> Inpaint inside bbox
-> Composite masked result back
```

### What Worked

- The model has more room to create a clear object.
- Full object generation improves compared with tight masks.
- Background outside the bbox is preserved.

### What Failed

- Background inside the bbox can still be rewritten.
- Object scale can be wrong.
- Inserted object color and lighting may not match the scene.
- Mask edges and pasted outlines remain visible.

### Lesson

Letting the model generate inside a larger region improves object quality, but the generated background still cannot be trusted.

---

## 6. V5 - Context-Object Composite

### Architecture

```text
Original image
-> Choose valid insertion region
-> Generate candidate in context
-> Detect/segment target object
-> Correct scale and mask
-> Harmonize color, texture, edge, and shadow
-> Paste only object pixels into original image
-> Validate and retry
```

### Why It Is The Current Direction

V5 separates two responsibilities:

- **Diffusion model**: generate a plausible target object in context.
- **Original image**: remain the trusted background.

The generated background is discarded. Only the segmented target object is pasted onto the original image.

### Current Pedestrian-Specific Implementation

In code, V5 is still named `context_person_composite`. It includes:

- YOLOv8m-seg person extraction;
- depth-based expected height;
- foot-anchored scale correction;
- foreground occlusion logic for persons and vehicles;
- local color/brightness/texture matching;
- edge harmonization;
- detector-guided validation and retry;
- manifest and quality metrics.

### General Object-Insertion Interpretation

For another target object, the same V5 pattern becomes:

```text
context generation
-> target-object segmentation
-> object-specific scale and anchor correction
-> object-only paste
-> object-specific validation
```

---

## 7. Cross-Version Comparison

| Version | Background preservation | Object quality | Scale control | Main issue |
|---|---:|---:|---:|---|
| V1 Full Img2Img | Low | Medium/High | Low | Rewrites the whole image |
| V2 Patch Blend | Medium | Low/Medium | Low | Patch lacks context |
| V3 Tight Mask Inpaint | High | Low | Low | Mask too restrictive |
| V4 BBox Inpaint | Medium/High | Medium/High | Medium | Generated bbox background leaks in |
| V5 Context-Object Composite | High | High if detector succeeds | Medium/High | Needs class-specific detector and validation |

---

## 8. Technical Insights

### 8.1 Prompting Alone Cannot Guarantee Background Preservation

Diffusion denoising can alter any visible part of the image. If the dataset background must remain faithful, the pipeline must enforce preservation through segmentation and compositing.

### 8.2 Object Generation And Background Preservation Should Be Separate

The strongest design is:

```text
generate candidate object
-> extract object
-> discard generated background
-> paste object onto original image
```

### 8.3 Scale Needs Geometry, Not Just Text

Prompts like "correct scale" are weak. Scale should use depth, perspective, reference objects, anchor points, or domain priors.

### 8.4 Validation Is Part Of Generation

The pipeline should expect failure:

- no object detected;
- wrong class;
- low confidence;
- wrong scale;
- partial/cropped object;
- ghost-like object;
- bad edge quality;
- invalid occlusion;
- background corruption.

Reject/retry logic is more practical than searching for one perfect parameter set.

### 8.5 Generalization Requires Class-Specific Rules

The architecture is general, but each object class needs its own:

- valid placement areas;
- detector/segmenter;
- scale envelope;
- anchor point;
- occluder logic;
- acceptance thresholds.

---

## 9. Next Steps

1. Rename code-level concepts from `person` to `object` where practical, without breaking the current benchmark.
2. Add a target-class abstraction for prompts, detector IDs, scale policy, and validation.
3. Support more detector/segmenter backends, including open-vocabulary options.
4. Improve mask quality with stronger segmentation or matting.
5. Add dataset adapters beyond CityPersons.
6. Measure downstream model performance, not only visual quality.

---

## Conclusion

The project has moved from "make diffusion edit an image" toward a more reliable data augmentation architecture:

```text
generate object candidates, keep the original background, validate everything
```

That is the core task for this repo: **background-preserving object insertion for dataset augmentation**.
