# LoRA flow

> **Active direction — inpaint-EDIT LoRA.** The goal is a model that adds an
> object while preserving 100% of the background and matching the photo's vibe.
> Base SD3.5 already makes good pedestrians, so a plain concept LoRA adds little;
> the value is in *edit* behavior learned from before/after pairs (PIPE). SD3.5
> has no official inpaint/edit trainer, so we hand-rolled one. A single-pair
> overfit spike (notebook `00`, collapse_ratio 0.18) confirmed the edit
> conditioning is wired correctly. Training: notebook `05` /
> `train/train_inpaint_edit.py`. Inference + eval (D2) is being built next.
>
> The concept-LoRA + reconstruction-eval pieces below remain for comparison /
> the detector-utility track, but the edit flow is the headline.

Two independent sub-flows, no V5 copy:

```text
A. Data ETL + Train
   raw datasets -> LoRA release -> train adapter -> model artifacts

B. SD3.5 Inpaint Test
   frozen test images + masks
   -> SD3.5 inpaint baseline (B0)
   -> SD3.5 inpaint + LoRA   (B1)
   -> paired metrics + report
```

There is **no** scale correction, semantic placement, object-only composite,
harmonization, or autotune here. The inpaint test calls raw SD3.5 inpaint; the
only difference between B0 and B1 is the trigger token in the prompt.

## Layout

```text
LoRA/
  configs/
    sources.yaml            # sources, quality thresholds, split + eval ratios
    prompt_templates.yaml   # trigger token + caption / validation / inpaint prompts
    lora_train.yaml         # training hyperparameters + pinned trainer path
    inpaint_eval.yaml       # baseline-vs-LoRA eval config
  data/                     # ETL: ingest -> ... -> export -> validate + build_eval_cases
    parsers/{yolo,mot,classification}.py
  train/                    # train_sd35_lora, export_artifacts, provenance
  inference/                # sd35_inpaint_runner, inpaint_metrics, report
  notebooks/                # 01_build_lora_release, 02_train_sd35_lora, 03_test_sd35_inpaint_lora
  tests/
  vendor/diffusers/<commit>/train_dreambooth_lora_sd3.py   # pinned trainer
```

Notebooks hold **no** ETL/train/eval logic — they only call `LoRA.data`,
`LoRA.train`, `LoRA.inference`. Import style is clean package imports
(`from LoRA.data.pipeline import run_full_etl`); add the repo root to `sys.path`.

## A. Data ETL (`notebooks/01_build_lora_release.ipynb`)

```text
00 ingest -> 01 normalize -> 02 dedupe/group -> 03 build eval cases
-> 04 filter/crop -> 05 caption -> 06 split -> 07 export -> 08 validate
```

- **Sources** (`configs/sources.yaml`): CityPersons (YOLO), MOT17-02 (MOT),
  Human Detection (classification). Per source, `lora_splits` feed the LoRA
  release and `eval_splits` are frozen for the inpaint test (never train LoRA).
  CityPersons `train` -> LoRA; CityPersons `valid` -> inpaint eval.
- **Crops** keep pedestrian + 25% context (no transparent cutouts).
- **Captions** are per-image and always contain the trigger token `<vin_ped>`;
  attributes not present in annotations (gender, age, emotion) are never invented.
- **Split** is group-aware (scene / sequence-window / dedupe cluster) so train
  and val never share a scene.
- **Eval cases** (`build_eval_cases`) come from the eval-locked splits and are
  partitioned group-disjoint into `inpaint_eval_v1` (dev) and
  `final_inpaint_test_v1` (touch once).

`validate_release` hard-fails on: empty trigger token, any caption missing the
trigger, empty train/val, unreadable crop, duplicate-cluster or group overlap
between train/val, or any eval image/group leaking into the release. On success
`release.json: dataset_status` flips to `validated`.

## B. Train (`notebooks/02_train_sd35_lora.ipynb`)

```text
00 git SHA -> 01 GPU/deps -> 02 vendor pinned trainer -> 03 verify validated release
-> 04 dry run -> 05 smoke 100 steps -> 06 1000 steps -> 07 zip artifacts
```

The trainer is pinned under `vendor/diffusers/<commit>/` (see its `VENDOR.md`) —
never downloaded from `main` per run. The command runs in caption mode but always
passes `--instance_prompt` (the Diffusers SD3 LoRA script requires it even with
`--caption_column`). Loss is monitored live and training hard-fails on NaN/Inf.

Run output (`models/<model_name>/run_NNN/`): `adapter/pytorch_lora_weights.safetensors`
(canonical) + `.pt` (handoff), `checkpoints/`, `training_config.json`,
`train_command.json`, `training_provenance.json`, `dataset_provenance.json`,
`adapter_verification.json`, `adapter_sha256.txt`, `pip_freeze.txt`,
`gpu_info.json`, `validation_prompts.json`. Adapter selection uses validation
prompts + `lora_val`; the frozen eval is run once after config is locked.

## C. Inpaint test (`notebooks/03_test_sd35_inpaint_lora.ipynb`)

**Golden eval set — PIPE.** The eval cases come from
[`paint-by-inpaint/PIPE`](https://huggingface.co/datasets/paint-by-inpaint/PIPE),
which provides **real** before/after pairs: `source_img` is the object-erased
background (inpaint input) and `target_img` is the real photo (ground truth).
PIPE ships no mask, so `build_eval_cases_pipe.py` derives the object mask from the
thresholded, dilated `|target − source|` difference. The builder filters to
person instructions and writes `eval/pipe_eval_v1/{cases.jsonl,images,masks,
reference}` (`images`=source, `reference`=target). Because the reference is a
real photo (not another model's output), `outside_mask_*` metrics are meaningful.
Settings live under `configs/inpaint_eval.yaml: pipe_eval`.

> The earlier `build_eval_cases.py` (CityPersons valid → mask over the existing
> person) is a reconstruction proxy and is kept for that flow; PIPE is the real
> golden set for "add a person to a background".

**Background preservation = hard-restore, not learned.** The final task ("add a
person, keep the background 100%") is achieved at inference, not by training: the
runner composites `result = input*(1-mask) + generated*mask` (config
`hard_restore: true`), so the LoRA only fills inside the mask and the background
outside is preserved byte-for-byte. The LoRA's job is the person quality inside
the mask; the restore guarantees the background. (PIPE's source/target differ
only inside the object region, so `outside_mask_mae` vs the target reference
stays ~0 and effectively verifies the restore + the derived mask.)

Runs B0 then B1 with **identical** image/mask/prompt-fields/seed/resolution/
strength/guidance/steps/negative-prompt — only the trigger token differs (B1 also
attaches the LoRA adapter). Per-case metrics (component metrics only, no fused
score):

`outside_mask_mae`, `outside_mask_ssim`, `person_detected`, `person_confidence`,
`person_inside_mask_ratio`, `expected/detected_height`, `scale_ratio`,
`edge_seam_score`, `runtime_seconds`, `cuda_peak_mb`.

`report.build_paired_comparison` writes `metrics_per_case.csv`,
`metrics_summary.json`, and `paired_comparison.csv` (per-metric `delta_*` =
LoRA − baseline). The inpaint stack is pinned to `diffusers==0.35.2` /
`transformers==4.46.3` / `accelerate==1.11.0` (see `configs/inpaint_eval.yaml`
and notebook 03).

## Trigger token

`<vin_ped>` appears in 100% of train/val captions, the validation prompts, the
training instance prompt, `training_provenance.json`, and the B1 inpaint prompt.
It is a LoRA concept trigger, not a textual-inversion embedding (no tokenizer
special token is added).

## Tests

`pytest LoRA/tests/` — release validation gates, caption contract, inpaint metric
contract, provenance. (Requires `pandas`, `pyarrow`, `pillow`, `numpy`, `pyyaml`.)
