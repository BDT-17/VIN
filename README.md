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
| `LoRA/` | **LoRA training + inpaint test** | Two sub-flows: (A) data ETL → release → train adapter; (B) SD3.5 inpaint baseline vs LoRA | Active focus; clean package (no V5 copy) |
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

## 2. LoRA flow (`LoRA/`)

A clean package with **no copy of the V5 augmentation flow** (no scale
correction, semantic placement, object-only composite, harmonization, or
autotune). Two independent sub-flows:

```text
A. Data ETL + Train      raw datasets -> LoRA release -> train adapter -> artifacts
B. SD3.5 Inpaint Test    frozen cases -> SD3.5 inpaint (B0) vs +LoRA (B1) -> paired metrics
```

**Layout**

- `LoRA/configs/` — `sources.yaml`, `prompt_templates.yaml`, `lora_train.yaml`,
  `inpaint_eval.yaml`
- `LoRA/data/` — ETL: `ingest → normalize → dedupe → build_eval_cases → curate →
  captions → splits → export → validate` (+ `parsers/`)
- `LoRA/train/` — `train_sd35_lora`, `export_artifacts`, `provenance`
- `LoRA/inference/` — `sd35_inpaint_runner`, `inpaint_metrics`, `report`
- `LoRA/notebooks/` — `01_build_lora_release`, `02_train_sd35_lora`,
  `03_test_sd35_inpaint_lora` (thin orchestration only)
- `LoRA/vendor/diffusers/<commit>/` — pinned training script

**Run on Kaggle** (add the repo root to `sys.path`, then `from LoRA.data… import`):

1. `01_build_lora_release.ipynb` — resolves dataset mounts, runs the ETL, and
   hard-fails unless the release validates. Produces a `pedestrian_lora_v1`
   release + frozen `inpaint_eval_v1` / `final_inpaint_test_v1` eval sets.
2. `02_train_sd35_lora.ipynb` — vendors the pinned trainer, verifies the release
   is `validated`, dry-run → smoke (100) → 1000 steps, exports adapter +
   provenance. Caption mode always passes `--instance_prompt`.
3. `03_test_sd35_inpaint_lora.ipynb` — raw SD3.5 inpaint, B0 vs B1 with identical
   inputs except the trigger token; writes paired component metrics.

**Dataset & eval policy** (`configs/sources.yaml`): CityPersons `train`, MOT17-02,
and Human Detection `1` feed the LoRA release (group-aware train/val split);
CityPersons `valid` is frozen for the inpaint eval and never trains LoRA. Trigger
token `<vin_ped>` in 100% of captions, validation prompts, and provenance.

**Validation hard-fails** on: empty trigger token, any caption missing the
trigger, empty train/val, unreadable crop, duplicate-cluster/group overlap
between train↔val, or any eval image/group leaking into the release.

**Metrics** (inpaint test, component-only — no fused score): `outside_mask_mae`,
`outside_mask_ssim`, `person_detected/confidence`, `person_inside_mask_ratio`,
`scale_ratio`, `edge_seam_score`, `runtime_seconds`, `cuda_peak_mb`. The report
emits per-metric `delta_* = LoRA − baseline`.

See `LoRA/README_LORA.md` for the full per-stage contract.

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

## Imports

- **`LoRA/`** is a clean Python package: add the repo root to `sys.path` and use
  `from LoRA.data.pipeline import run_full_etl` etc. No bare `sd35_*` imports, no
  dual-config ambiguity. Behavior is driven by YAML under `LoRA/configs/`.
- **Root augmentation flow** still uses bare `from sd35_config import *`, so its
  notebooks import the root `sd35_*.py` modules directly (the LoRA copies are gone).

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
