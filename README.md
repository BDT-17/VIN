# VIN — Pedestrian Data Augmentation & LoRA Toolkit

Research toolkit built around **Stable Diffusion 3.5 Medium** for generating and
training on synthetic pedestrian data, with **CityPersons** as the reference
benchmark. The repository hosts four independent flows that share the same SD3.5
foundation but serve different goals.

> The central design rule across every flow: **the source image is the trusted
> background; generated background pixels are not trusted.**

---

## Repository layout

| Path | Flow | Purpose | Status |
|------|------|---------|--------|
| `sd35_*.py`, `sd35_run.ipynb`, `sd3.5-…clean.ipynb` | **V5 augmentation** | Insert objects into existing images while preserving the background | Smoke-runnable research baseline |
| `LoRA/` | **LoRA training** | Build a validated dataset release, then train + export an SD3.5 LoRA adapter | Active focus; data pipeline + training ready |
| `inpaint/` | **AI Replace** | Pokecut-style inpaint with hard background restoration | Background-preservation smoke flow |
| `addit(experimental)/` | **ADDIT** | Concept-injection experiments | Experimental, not validated |
| `docs/` | — | Design notes, model selection, workflow diagrams | — |
| `tests/` | — | Root augmentation tests (`LoRA/tests/` holds the LoRA tests) | — |

> **Note — two `sd35_config.py`:** the repo root and `LoRA/` each have their own
> copy of the `sd35_*.py` modules (the LoRA folder is a divergent copy of the
> augmentation flow plus LoRA-specific code). They are **not** interchangeable —
> see [Import gotchas](#import-gotchas).

---

## 1. V5 augmentation flow (root)

Adds new objects to existing images in local scene context, segments only the
newly generated pixels, corrects scale/placement/occlusion, and composites the
object back onto the **original** background.

```text
Dataset scan -> placement proposal -> context img2img generation
-> target-object segmentation -> perspective/scale correction
-> edge/color/shadow blending -> object-only alpha composite
-> detector validation + retry -> image outputs + manifest
```

**Key modules**

- `sd35_config.py` — presets, generation params, placement/validation thresholds
- `sd35_data.py` — dataset scan and preview
- `sd35_utils.py` — preprocessing, placement, masks, scale/depth helpers
- `sd35_model.py` — SD3.5 pipeline loading
- `sd35_pipeline.py` — generation, paste, edge correction, compositing
- `sd35_evaluation.py` — detector/segmenter validation and retry policy
- `sd35_edge_harmonization.py` — boundary-only edge harmonization
- `sd35_runner.py` — job building, runner, manifest, autotune, export

**Run on Kaggle** (Internet ON, GPU enabled):

- `sd35_run.ipynb` — clone-based runner. Clones/pulls this repo into
  `/kaggle/working/VIN` and imports the root modules.
- `sd3.5-agumentation-scale-correction-clean.ipynb` — self-contained notebook
  that writes the modules to `/kaggle/working` via `%%writefile`, then runs them.

Cell order: install deps → clone/update → imports → runtime check → HF login →
dataset scan → smoke run → export (optional).

**Safety gates** (recent hardening): augmentation refuses `test`/`val`/`valid`
splits (`augment_dataset` raises on benchmark contamination); semantic placement
and total-affordance rejection are hard gates; `scale_score` is `None` (not a
false `1.0`) when `expected_height` is unavailable, and the metrics summary
reports `scale_coverage_rate`.

---

## 2. LoRA training flow (`LoRA/`)

End-to-end: **raw datasets → build dataset release → validate → train LoRA →
export adapter**. All logic lives in Python modules; the notebook only calls them.

**Entry points**

- `LoRA/sd35_run.ipynb` — the Kaggle LoRA runner (recommended). Clones the repo,
  resolves dataset mounts, builds + validates a release, then runs smoke/full
  training and verifies the adapter.
- `LoRA/data/` — the data contract: `sources.yaml`, parsers (YOLO/MOT/
  classification), dedupe, splits, captions, crops, export, validate, metrics.
- `LoRA/sd35_lora_training.py` — training/export entrypoint (builds the Diffusers
  `accelerate launch` command, monitors loss with NaN/Inf hard-fail, exports the
  `.safetensors` + `.pt` adapter and provenance).

**Run on Kaggle** (`LoRA/sd35_run.ipynb`):

1. **Datasets** — attach via *Add Data*. Cell 04 auto-probes candidate mount
   paths for each source in `sources.yaml` (both `/kaggle/input/<slug>` and
   `/kaggle/input/datasets/<user>/<slug>` forms), verifies the parser's expected
   sub-structure, and writes `/kaggle/working/sources_resolved.yaml`.
2. **SD3.5 model** — mount the model dataset, **or** add a Kaggle secret
   `HF_TOKEN` so it downloads from Hugging Face.
3. Run cells: install → clone → GPU preflight → resolve mounts → (HF login) →
   build release → validate → ImageFolder contract → dry run → smoke train (50
   steps) → verify adapter → full train (1000 steps) → metrics → zip artifacts.

**Dataset & split policy**

- Sources (`LoRA/data/sources.yaml`): CityPersons (YOLO, `benchmark_lock`),
  MOT17-02 (MOT), Human Detection (classification). Trigger token `<vin_ped>`.
- A `benchmark_lock` source contributes **both** frozen benchmark and LoRA data,
  kept scene-disjoint by the group-aware splitter:
  - source `valid` → `detector_val_real_frozen`
  - the group-disjoint ~15% `test` slice of source `train` →
    `detector_test_real_frozen`
  - the remaining ~85% of source `train` → `lora_positive` (LoRA training)
- 14 hard validation gates run in `validate.py` (split-group leakage, duplicate
  clusters, captions/trigger token, crop bounds, source share, etc.).

**Artifacts** (per training run): `pytorch_lora_weights.safetensors` + `.pt`,
`training_config.json`, `training_provenance.json`, `dataset_provenance.json`,
`gpu_info.json`, `pip_freeze.txt`, `validation_prompts.json`,
`adapter_verification.json`, and `reports/training/<run_id>/` (metrics.jsonl,
summary.json, loss_curve.png).

See `LoRA/README_LORA.md` for inference/adapter-loading config details.

---

## 3. AI Replace flow (`inpaint/`)

Standalone Pokecut-style inpainting. Builds an insertion mask from a bbox, runs
SD inpainting, then **hard-restores the original pixels outside the mask** so the
background is provably preserved; validates with detector/ghost checks and emits
manifest metrics.

- `inpaint/sd35_ai_replace.py` — main flow
- `inpaint/sd35_mask_refinement.py`, `sd35_harmonization.py`,
  `sd35_ghost_detection.py` — mask/blend/validation
- `inpaint/smoke_runner.py`, `inpaint/smoke_test_ai_replace.ipynb` — smoke run
- `inpaint/test_background_preservation.py` — outside-mask restoration test

Current smoke uses a fixed heuristic bbox (≈42% image height, centered ≈52% x,
bottom ≈90% y) for every image — so a passing smoke run proves background
preservation, **not** placement quality. See `inpaint/README_ai_replace.md`.

---

## 4. ADDIT (experimental, `addit(experimental)/`)

Concept-injection experiments (`addit_core.py`, `addit_pipeline.py`,
`addit_config.py`, `addit_run.ipynb`). `ADDIT_CONCEPT_ENABLED=True` but the
advanced components (weighted extended attention, structure transfer,
subject-guided blend proxy) are still disabled. Treat as an experimental branch,
not a baseline.

---

## Import gotchas

All `LoRA/*.py` modules use bare absolute imports (`from sd35_config import *`).
Because the repo root **and** `LoRA/` both contain `sd35_config.py` (and the root
copy lacks the `LORA_TRAINING_*` names), the LoRA flow requires `…/VIN/LoRA` to
be **ahead of** `…/VIN` on `sys.path` so bare imports resolve to
`LoRA/sd35_config.py`. `LoRA/sd35_run.ipynb` Cell 02 sets this up.

To patch training config at runtime, patch the **bare** module
(`import sd35_config as _cfg`) — the same object the trainer reads — not
`import LoRA.sd35_config` (a separate module loaded from the same file). After
patching, `importlib.reload(LoRA.sd35_lora_training)` re-runs its
`from sd35_config import *` and picks up the change.

---

## Hugging Face access

For any flow that downloads SD3.5 from Hugging Face, add a Kaggle secret:

```text
Kaggle Notebook -> Add-ons -> Secrets -> Add secret
Name: HF_TOKEN
Value: <your Hugging Face access token>
```

Notebooks read it via `UserSecretsClient().get_secret("HF_TOKEN")`; locally you
can set an `HF_TOKEN` environment variable instead.

---

## GPU memory

Defaults target an NVIDIA T4 16 GB: SD3.5 Medium, 512 px, fp16, batch size 1,
gradient accumulation 4, gradient checkpointing, 8-bit Adam, frozen text
encoders, rank-8 LoRA. The augmentation runner processes devices sequentially
(`load pipeline → run shard → del pipe → clear_cuda() → next device`) to keep
VRAM stable.

---

## Status

A research toolkit, not a production service. The V5 augmentation flow is a
smoke-runnable baseline; the LoRA flow (data release + training) is the current
focus; AI Replace is a background-preservation smoke flow; ADDIT is experimental.
Datasets are **not** stored in this repo.
