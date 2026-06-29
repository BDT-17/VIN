# Shared metrics — comparing the two insertion flows

The repo has two object-insertion flows. They were each measuring quality
differently, so their numbers could **not** be compared head-to-head:

| Flow | Native metric module | Native scoring style |
|------|----------------------|----------------------|
| V5 augmentation | `sd35_metrics.py` | fused `affordance_score` + placement/scale/occlusion gates |
| LoRA inpaint | `LoRA/inference/inpaint_metrics.py` | component-only (no fused score) |

`shared_metrics.py` (repo root) adds a **common intersection** every flow can
compute, so the two become directly comparable while each keeps its own extra
metrics as extensions.

## The shared metric set

Computed by `shared_metrics.compute_shared_metrics(...)`:

| Metric | Meaning | Direction |
|--------|---------|-----------|
| `person_detected` / `person_confidence` / `person_count` | object detected in result (injected detector) | higher better |
| `object_added` / `inclusion_count_delta` / `source_person_count` | did a NEW object appear vs the source? (the "Inclusion" analogue) | higher better |
| `scale_ratio` / `scale_error` / `detected_height` / `expected_height` | detected vs expected object height | error lower better; ratio neutral |
| `bg_mae` / `bg_ssim` | background preservation OUTSIDE the object region | mae lower / ssim higher |

Directions live in `shared_metrics.SHARED_METRIC_DIRECTIONS` and drive the
paired comparison.

## Design choices

- **Dependency-injected detector.** `shared_metrics` imports only `numpy` +
  `Pillow`. The detector is a callable
  `detector(image) -> [{"bbox_xyxy", "conf", "cls"}]`, so the module imports
  anywhere and no flow gains a torch/YOLO dependency it didn't already have.
  Without a detector, detection-based metrics are skipped (0 / None) — never
  faked.
- **Object region, three ways.** Background preservation needs to know what to
  exclude. Priority: explicit **mask** (LoRA has one) → explicit **bbox** (V5's
  insert bbox) → **detected boxes** (fallback when there is no fixed region).
  Same metric, each flow supplies what it has.
- **Honest scale.** `scale_ratio`/`scale_error` are `None` when no expected
  height is given (no fake 1.0) — matching V5's existing convention.
- **No fused score in the shared layer.** The shared set is component metrics
  only; V5's fused `affordance_score` stays in `sd35_metrics` as a V5 extension.

## How each flow calls it

- **LoRA**: `inpaint_metrics.compute_shared_case_metrics(reference, result,
  source, mask, ...)` — reference is the PIPE target (background = source),
  object region is the inpaint mask. Additive to the existing
  `compute_case_metrics`; the LoRA contract test is unchanged.
- **V5**: `sd35_metrics.compute_shared_case_metrics(result, source, detector,
  expected_height, insert_bbox)` — reference is the original source (V5
  composites the object onto the untouched background), object region is the
  planned insert bbox. Additive; the manifest/affordance schema is unchanged.

## Cross-flow comparison

`shared_metrics.paired_shared(rows_a, rows_b, label_a=..., label_b=...)` matches
rows by `(case_id, seed)` and reports, per metric, each side's mean, the delta
(B−A), and whether B improved given the metric's direction. Run the same eval
cases through both flows, collect the shared rows, and pair them.

`shared_metrics.summarize_shared(rows)` gives per-metric means for a single
flow's run.

## What is NOT in the shared layer (kept per-flow)

- V5: `placement_score`, `affordance_score`, `occlusion_score`, edge/boundary
  scores, the fused gate logic.
- LoRA: `outside_mask_mae/ssim` (mask-relative), `edge_seam_score`,
  `person_inside_mask_ratio`.

## Tests

`tests/test_shared_metrics.py` (10 tests, numpy/PIL + a fake detector — no torch)
covers person detection + class filter, inclusion delta, scale none/ratio/error,
background identical/outside-change/inside-change, full-schema completeness, and
paired-direction logic.
