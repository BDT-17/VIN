# Project Key Concepts

- **Goal**: Generate realistic pedestrian augmentations for CityPersons street scenes while preserving the original background.

- **Core Pipeline**: Stable Diffusion 3.5 Medium generates candidate pedestrians; YOLOv8m-seg extracts only pedestrian pixels; the pipeline composites those pixels back onto the original image.

- **Background Preservation**: The original frame is treated as ground truth. Generated background pixels are discarded to avoid scene distortion.

- **Context-Person Composite**: The project separates generation, segmentation, scale correction, and compositing instead of relying on prompt-only image editing.

- **Perspective Scale Correction**: Detected pedestrians are resized after generation based on foot/ground position so far, mid, and near pedestrians have plausible heights.

- **Foot-Anchored Placement**: Resizing keeps the pedestrian foot point fixed, reducing floating people and bad ground contact.

- **Edge Integration**: Mask cleanup, appearance harmonization, seamlessClone, alpha fallback, and foreground preservation are used to reduce pasted-looking edges without making the person disappear into the background.

- **Adaptive Retry Policy**: Failed generations are retried based on rejection reason, including no person detected, low confidence, bad scale, missing people, floating, or poor final composite quality.

- **Multi-Person Handling**: Two-person and small-group variants require per-person detection, per-person scale correction, and count validation.

- **Research Focus**: Current bottlenecks are geometry, scale, grounding, blending, and validation policy, not changing the base model.

- **Main Outputs**: Augmented images, side-by-side comparison pairs, debug strips, manifest metadata, and quality metrics.

- **Current Status**: This is a research baseline for iterative tuning, not a finalized production augmentation generator.
