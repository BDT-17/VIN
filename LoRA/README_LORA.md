# LoRA flow

> **Concept (text→image) LoRA — simplest flow.** If the goal is just "train one
> model that learns to GENERATE images like the dataset" (no editing, no
> background preservation), use the concept flow: a plain DreamBooth-style LoRA
> trained on the PIPE `add a person` subset's *finished* photos (`target_img`).
> No mask, no source image, no `input_proj`, no segment/paste — ONE artifact
> (`pytorch_lora_weights.safetensors`) and inference is a straight `pipe(prompt)`.
> Files: `train/train_concept_lora.py`, `train/concept_lora_dataset.py`,
> `inference/sd35_concept_runner.py`, `inference/run_concept_eval.py`,
> `configs/concept_lora_train.yaml`, `notebooks/concept_01_all_in_one.ipynb`.
> The mask-free EDIT flow below is a separate, more complex direction.
>
> **Concept + segment-paste (100% bg preserve).** To add the generated person to a
> real background while keeping the original byte-exact, `concept_02_segment_paste.ipynb`
> reuses the YOLO-seg + composite step: the concept model generates a person (from
> the prompt, blind to the original), YOLOv8-seg cuts it out, and it is pasted onto
> the original (`segment_paste.generate_and_paste_concept`). Background outside the
> person is byte-exact (set `feather_px=0` for a strictly byte-exact seam). The
> person's placement/scale come from the generation — steer them with the prompt.

> **Active direction (edit) — mask-free EDIT LoRA + segment-and-paste.** The goal is a
> model that adds a person to a scene while preserving 100% of the original
> background and matching the photo's vibe. Base SD3.5 already makes good
> pedestrians, so a plain concept LoRA adds little; the value is in *edit*
> behaviour learned from before/after pairs (PIPE).
>
> The pipeline (2026-06 pivot):
> 1. **Generate** — a mask-free LoRA (+ `input_proj.pt`) conditioned on the
>    source image + a plain "add a person" instruction generates a FULL image
>    that adds a person fitting the scene. No mask, no background preservation at
>    this stage — the model decides where/scale/pose. (`train/train_maskfree_edit.py`,
>    `inference/sd35_maskfree_runner.py`.)
> 2. **Segment** — YOLOv8-seg cuts the generated person out
>    (`inference/person_detector.py::load_person_segmenter`).
> 3. **Paste** — the segmented person is composited back onto the ORIGINAL
>    background at the same coordinates, feathered + colour-matched so it isn't a
>    sticker (`inference/segment_paste.py`). Background outside the person is
>    byte-exact by construction (composite, not hard-restore).
>
> A single-pair overfit spike (`spike_maskfree_conditioning.ipynb`) confirmed the
> mask-free 32→16 `input_proj` conditioning is wired correctly.

```text
A. Data ETL + Train
   raw datasets / PIPE pairs -> train mask-free adapter -> model artifacts
       (pytorch_lora_weights.safetensors + input_proj.pt)

B. Generate -> Segment -> Paste (test / augment)
   original image
   -> mask-free generate (full image, person added freely)
   -> YOLOv8-seg the person
   -> paste onto the original background (feather + colour-match)
```

There is **no** scale correction, semantic placement, harmonization network, or
autotune here. Background preservation is achieved by the segment-and-paste
composite, not by a learned objective.

## Layout

```text
LoRA/
  configs/
    sources.yaml             # sources, quality thresholds, split + eval ratios
    prompt_templates.yaml    # trigger token + caption / validation prompts
    maskfree_edit_train.yaml # mask-free EDIT training hyperparameters
    inpaint_eval.yaml        # PIPE eval config
  data/                      # ETL: ingest -> ... -> export -> validate; build_eval_cases_pipe
    parsers/{yolo,mot,classification}.py
  train/                     # train_maskfree_edit, maskfree_edit_dataset, spike, export_artifacts, provenance
  inference/                 # sd35_maskfree_runner, person_detector, segment_paste, inpaint_metrics, report
  notebooks/                 # concept_01 (train), concept_02 (segment-paste inference), data_01 (ETL)
  tests/
  vendor/diffusers/<commit>/train_dreambooth_lora_sd3.py   # pinned trainer
```

Notebooks hold **no** ETL/train/eval logic — they only call `LoRA.data`,
`LoRA.train`, `LoRA.inference`. Import style is clean package imports
(`from LoRA.data.pipeline import run_full_etl`); add the repo root to `sys.path`.

### Notebooks

| Notebook | Purpose |
|---|---|
| `concept_01_all_in_one.ipynb` | **Concept (text→image) LoRA**: train on PIPE person photos → generate → contact sheet. |
| `concept_02_segment_paste.ipynb` | **Concept inference**: load a trained concept adapter → generate → YOLO-seg → paste person into the original (100% bg preserved). |
| `data_01_build_lora_release.ipynb` | Build the LoRA dataset release via the ETL pipeline. |

The mask-free EDIT flow's notebooks were removed; its code remains (legacy) under
`train/train_maskfree_edit.py`, `inference/sd35_maskfree_runner.py`, etc., runnable
directly if that direction is revisited.

## A. Data ETL (`notebooks/data_01_build_lora_release.ipynb`)

```text
00 ingest -> 01 normalize -> 02 dedupe/group -> 03 build eval cases
-> 04 filter/crop -> 05 caption -> 06 split -> 07 export -> 08 validate
```

- **Sources** (`configs/sources.yaml`): CityPersons (YOLO), MOT17-02 (MOT),
  Human Detection (classification). Per source, `lora_splits` feed the LoRA
  release and `eval_splits` are frozen (never train LoRA).
- **Crops** keep pedestrian + 25% context (no transparent cutouts).
- **Split** is group-aware (scene / sequence-window / dedupe cluster) so train
  and val never share a scene.

`validate_release` hard-fails on: empty train/val, unreadable crop,
duplicate-cluster or group overlap between train/val, or any eval image/group
leaking into the release.

## B. Mask-free train (legacy — `train/train_maskfree_edit.py`)

Trains on **PIPE** ([`paint-by-inpaint/PIPE`](https://huggingface.co/datasets/paint-by-inpaint/PIPE)),
which provides **real** before/after pairs: `source_img` is the object-erased
background and `target_img` is the real photo. PIPE is an object-ADDITION
dataset, so every pair is an "add object" edit and `Instruction_Class` holds the
object category. The mask-free loader (`train/maskfree_edit_dataset.py`) filters
**strictly on `Instruction_Class`** to person classes (`_is_person_class`, whole-word,
class field only — the free-text caption is never matched, so a non-person class
whose caption mentions a person does not leak in). No mask is used or fed to the
model; a `|target − source|` diff is used only to drop no-op pairs.

The model is conditioned on (source latent, instruction) via the 32→16
`input_proj`, trained with IP2P-style classifier-free guidance (drop text /
image / both at small probability). Plain natural-language instructions only —
the mask-free flow **dropped the `<vin_ped>` trigger token**.

Run output (`models/<model_name>/run_NNN/`):
`adapter/pytorch_lora_weights.safetensors` + `adapter/input_proj.pt` (BOTH
required at inference), `training_provenance.json`, `checkpoints/`, etc.

## C. Generate → segment → paste (legacy — `inference/sd35_maskfree_runner.py`)

1. `inference/sd35_maskfree_runner.py` loads base SD3.5 + LoRA + `input_proj.pt`
   and runs the full denoise loop, reconstructing the edit conditioning each step
   with two-scale CFG (`s_image` = source adherence, `s_text` = instruction).
2. `person_detector.load_person_segmenter` (YOLOv8n-seg, COCO class 0) returns a
   per-person `{mask, bbox, conf}` from the generated frame.
3. `segment_paste.composite_persons` pastes the person onto the original with a
   feathered alpha + reinhard colour-match (optional OpenCV Poisson). The
   background outside the person is preserved byte-exact.

`segment_paste.generate_and_paste` runs all three for one image and returns
`(composite, generated, info)` so a `original | generated | composite` sheet can
be built. **Known tradeoff:** the person is lit by the *generated* scene, so a
paste can look like a sticker — raise `s_image`, increase `color_match`, or
enable Poisson to mitigate.

## Metrics

`inference/inpaint_metrics.py` (component metrics only, no fused score):
`outside_mask_mae`, `outside_mask_ssim`, `person_detected`, `person_confidence`,
`person_inside_mask_ratio`, `expected/detected_height`, `scale_ratio`,
`edge_seam_score`, `runtime_seconds`, `cuda_peak_mb`. The person metrics require
a detector — `person_detector.load_person_detector` (YOLOv8n) supplies it;
without it `person_*` read 0/None.

## Tests

`pytest LoRA/tests/` — release validation gates, caption contract, inpaint metric
contract, PIPE eval builder + person filter, segment-paste compositor, provenance.
(Requires `pandas`, `pyarrow`, `pillow`, `numpy`, `pyyaml`.)
