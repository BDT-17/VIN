# LoRA flow

> **Concept (text→image) LoRA.** The goal is "train one model that learns to
> GENERATE images like the dataset". A plain DreamBooth-style SD3.5 LoRA trained
> on the PIPE `add a person` subset's *finished* photos (`target_img`). No mask,
> no source image, no `input_proj` — ONE artifact
> (`pytorch_lora_weights.safetensors`) and inference is a straight `pipe(prompt)`.
> Files: `train/train_concept_lora.py`, `train/concept_lora_dataset.py`,
> `inference/sd35_concept_runner.py`, `inference/run_concept_eval.py`,
> `configs/concept_lora_train.yaml`, `notebooks/concept_01_all_in_one.ipynb`.
>
> **Concept + segment-paste (100% bg preserve).** To add the generated person to a
> real background while keeping the original byte-exact,
> `notebooks/concept_02_segment_paste.ipynb` reuses a YOLO-seg + composite step:
> the concept model generates a person (from the prompt, blind to the original),
> YOLOv8-seg cuts it out, and it is pasted onto the original
> (`segment_paste.generate_and_paste_concept`). Background outside the person is
> byte-exact (set `feather_px=0` for a strictly byte-exact seam). The person's
> placement/scale come from the generation — steer them with the prompt.

```text
A. Train
   PIPE person photos (target_img) -> train concept LoRA -> pytorch_lora_weights.safetensors

B. Generate -> Segment -> Paste (test / augment)
   prompt
   -> concept generate (full image with a person)
   -> YOLOv8-seg the person
   -> paste onto the original background (feather + colour-match)
```

## Layout

```text
LoRA/
  configs/
    sources.yaml             # ETL sources, quality thresholds, split + eval ratios
    prompt_templates.yaml    # trigger token + caption / validation prompts (ETL)
    concept_lora_train.yaml  # concept (text->image) training hyperparameters
    lora_train.yaml          # legacy release-trainer config (train_sd35_lora)
  data/                      # ETL: ingest -> ... -> export -> validate; build_eval_cases_pipe; list_images
    parsers/{yolo,mot,classification}.py
  train/                     # concept_lora_dataset, train_concept_lora, vae_utils, provenance
                             #   (legacy: train_sd35_lora + export_artifacts)
  inference/                 # sd35_concept_runner, run_concept_eval, segment_paste, person_detector
  notebooks/                 # concept_01 (train), concept_02 (segment-paste inference), data_01 (ETL)
  tests/
  vendor/diffusers/<commit>/train_dreambooth_lora_sd3.py   # pinned trainer (legacy release flow)
```

Notebooks hold **no** ETL/train/eval logic — they only call `LoRA.data`,
`LoRA.train`, `LoRA.inference`. Import style is clean package imports
(`from LoRA.train.train_concept_lora import run_training`); add the repo root to
`sys.path`.

### Notebooks

| Notebook | Purpose |
|---|---|
| `concept_01_all_in_one.ipynb` | **Concept (text→image) LoRA**: train on PIPE person photos → generate → contact sheet. |
| `concept_02_segment_paste.ipynb` | **Concept inference**: load a trained concept adapter → generate → YOLO-seg → paste person into the original (100% bg preserved). |
| `data_01_build_lora_release.ipynb` | Build a VIN dataset release via the ETL pipeline (independent data tool). |

## A. Concept train (`notebooks/concept_01_all_in_one.ipynb`)

Trains on **PIPE** ([`paint-by-inpaint/PIPE`](https://huggingface.co/datasets/paint-by-inpaint/PIPE)).
PIPE is an object-ADDITION dataset; `target_img` is the real photo that already
contains the added object. The concept loader (`train/concept_lora_dataset.py`)
keeps **only** `target_img`, filtered **strictly on `Instruction_Class`** to person
classes (`_is_person_class`, whole-word, class field only — the free-text caption
is never matched, so a non-person class whose caption mentions a person does not
leak in). The `|target − source|` diff is used only to drop degenerate pairs; it
is never fed to the model.

Captions are derived from the PIPE instruction with the leading imperative verb
stripped ("add a man in a hat" → "a photo of a man in a hat"), so the text
encoder learns image→text alignment. An optional `trigger_token` can be prepended
(`configs/concept_lora_train.yaml`).

Standard SD3 flow-matching loss, LoRA rank 16 on attention, ONE artifact:
`models/<model_name>/run_NNN/adapter/pytorch_lora_weights.safetensors` +
`training_provenance.json` (`requires_input_proj == false`).

## B. Generate → segment → paste (`notebooks/concept_02_segment_paste.ipynb`)

1. `inference/sd35_concept_runner.py` loads base SD3.5 + LoRA and generates from a
   text prompt (`generate(prompt)`) — no source image.
2. `person_detector.load_person_segmenter` (YOLOv8-seg, COCO class 0) returns a
   per-person `{mask, bbox, conf}` from the generated frame. For a clean cut it
   uses `retina_masks=True` (native-resolution, crisp silhouette) — pick a larger
   weight (`yolov8x-seg.pt`) for the sharpest masks.
3. `segment_paste.composite_persons` pastes the person onto the original with a
   feathered alpha + reinhard colour-match (optional OpenCV Poisson). `erode_px`
   shrinks the mask inward to drop the YOLO background halo (a tighter cut);
   `feather_px` keeps the seam soft (0 = hard/byte-exact). The background outside
   the person is preserved byte-exact.

`segment_paste.generate_and_paste_concept` runs all three for one image and
returns `(composite, generated, info)` so an `original | generated | composite`
sheet can be built. **Known tradeoffs:** the person is lit by the *generated*
scene (raise `color_match` or enable Poisson to mitigate), and its placement/scale
come from the generation (the concept model is blind to the original — steer with
the prompt).

## C. Data ETL (`notebooks/data_01_build_lora_release.ipynb`) — independent

```text
00 ingest -> 01 normalize -> 02 dedupe/group -> 03 build eval cases
-> 04 filter/crop -> 05 caption -> 06 split -> 07 export -> 08 validate
```

Builds a VIN dataset release from CityPersons (YOLO), MOT17-02 (MOT), Human
Detection (classification) — a separate data tool, **not used** by the PIPE
concept flow above. The legacy release-trainer (`train/train_sd35_lora.py` +
`vendor/.../train_dreambooth_lora_sd3.py`, config `lora_train.yaml`) consumes this
release; `validate_release` hard-fails on empty train/val, unreadable crop,
duplicate-cluster or group overlap between train/val, or eval leakage.

## Tests

`pytest LoRA/tests/` — release validation gates, caption contract, concept caption
builder, PIPE eval builder + person filter, segment-paste compositor, provenance.
(Requires `pandas`, `pyarrow`, `pillow`, `numpy`, `pyyaml`.)
