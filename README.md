# VIN — Pedestrian Data Augmentation & LoRA Toolkit

Research toolkit built around **Stable Diffusion 3.5 Medium** for generating and
training on synthetic pedestrian data, with **CityPersons** as the reference
benchmark. The repository hosts two independent flows that share the same SD3.5
foundation but serve different goals.

> The central design rule across every flow: **the source image is the trusted
> background; generated background pixels are not trusted.**

---

## Repository layout

| Path | Flow | Purpose | Status |
|------|------|---------|--------|
| `sd35_*.py`, `sd35_run.ipynb`, `sd3.5-…clean.ipynb` | **V5 augmentation** | Insert objects into existing images while preserving the background | Smoke-runnable research baseline |
| `LoRA/` | **LoRA inpaint-edit** | Add an object into a scene (preserve 100% bg, match the vibe). Data ETL → train edit-LoRA on PIPE pairs → infer + eval | **Active focus** |
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

## 2. LoRA inpaint-edit flow (`LoRA/`) — active focus

A clean package (no copy of the V5 augmentation flow). Goal: **add an object
(person) into many kinds of scenes, preserve 100% of the background, and match
the photo's vibe.** Background preservation is achieved by hard-restore at
inference, not learned; the LoRA learns the *edit / vibe-match* behavior from
before/after pairs.

```text
Data: paint-by-inpaint/PIPE pairs  (source = object erased, target = real photo)
Train:  SD3.5 + LoRA(attn) + a trainable InputAdapter Conv(33->16) on
        [noisy | source_lat | mask_lat]  -> learns to fill the mask in-context
Infer:  full denoise loop (edit-conditioned each step) + hard-restore outside mask
```

SD3.5 has no official inpaint/edit trainer, so it is hand-rolled. A single-pair
overfit spike confirmed the conditioning (collapse_ratio 0.18); smoke train and
the full end-to-end run are verified (background preserved, edit only in the mask).

**Two required artifacts** (both needed at inference):
`adapter/pytorch_lora_weights.safetensors` + `adapter/input_adapter.pt`.

**Layout**

- `LoRA/configs/` — `inpaint_edit_train.yaml`, `prompt_templates.yaml`,
  `sources.yaml`, `inpaint_eval.yaml`
- `LoRA/train/` — `train_inpaint_edit` (full trainer), `inpaint_edit_dataset`
  (PIPE stream + diff-mask), `spike_inpaint_edit` (feasibility), `provenance`
- `LoRA/inference/` — `sd35_edit_runner` (denoise loop + hard-restore),
  `inpaint_metrics`, `report`
- `LoRA/data/` — PIPE eval builder + the older ETL/release helpers
- `LoRA/notebooks/` — flow-prefixed: **`maskfree_01_all_in_one.ipynb`** /
  **`maskfree_02_augment.ipynb`** (current focus); `maskbased_01..05` (the older
  mask + hard-restore edit flow); `concept_lora_01..02`, `data_01`, `spike_*`
- `LoRA/vendor/diffusers/<commit>/` — pinned training script

> **Pivot 2026-06-26:** the active flow is now **mask-FREE** (IP2P/PIPE-style:
> condition on source image + instruction, model decides where/scale, no mask,
> no hard-restore). The mask-based flow (`maskbased_*`) is kept for comparison —
> it produced people that did not fit the mask, traced to a near-mute mask channel
> and effective-batch-1 training vs the PIPE paper's 4096.

**Run on Kaggle:** open `maskfree_01_all_in_one.ipynb`, set GPU + SD3.5 access
(HF_TOKEN secret or a mounted model dataset), pick `SMOKE` true/false, then
**Save Version → Save & Run All**. Then `maskfree_02_augment.ipynb` loads the
trained adapter and adds people to real images from any sources.yaml dataset.
Training data is general PIPE person pairs (diverse scenes), not street-only.

See `LoRA/README_LORA.md` for the full per-stage contract.

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
smoke-runnable baseline; the **LoRA inpaint-edit flow is the current focus**
(conditioning + end-to-end run verified; full-train for quality in progress).
Datasets and trained model artifacts are **not** stored in this repo.
