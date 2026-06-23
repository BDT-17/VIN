# LoRA Experiment Flow

This folder is an isolated copy of the current SD3.5 CityPersons augmentation flow. The root pipeline is intentionally untouched.

## What changed here

- Outputs are written to `/kaggle/working/sd35_citypersons_lora`.
- Metrics are written to `/kaggle/working/lora_metrics`.
- Autotune snapshots are written to `/kaggle/working/lora_autotune_snapshots`.
- Optional LoRA adapter loading is available in `sd35_model.py` for both img2img and inpaint pipelines.
- Manifest rows, including rejected rows, include LoRA metadata for traceability.

## Scope

This folder is an inference/evaluation harness for an already-trained LoRA adapter. It does not include LoRA training, data preparation, or captioning. Keep training/data-prep scripts separate until they are ready to be integrated deliberately.

Because this folder copies the root SD3.5 pipeline, bug fixes in the root flow can drift from the LoRA flow. When changing shared placement, scale, compositing, or validation behavior, port and test the same change in both places or extract a shared module first.

## How to enable LoRA

Edit `sd35_config.py` in this folder:

```python
LORA_CONFIG = {
    "LORA_ENABLED": True,
    "LORA_PATH": "/kaggle/input/path-to-lora-or-hf-repo",
    "LORA_WEIGHT_NAME": "pytorch_lora_weights.safetensors",
    "LORA_ADAPTER_NAME": "citypersons_lora",
    "LORA_SCALE": 0.7,
    "LORA_FUSE": False,
    "LORA_TRIGGER_TOKEN": "cityperson_lora",
    "LORA_PROMPT_PREFIX": "cityperson_lora pedestrian",
}
```

Keep `LORA_ENABLED=False` to run the copied baseline behavior. Do not set `LORA_ENABLED=True` until `LORA_PATH` points to a valid local folder or Hugging Face repo. If your adapter was trained with a trigger token, set `LORA_TRIGGER_TOKEN` and/or `LORA_PROMPT_PREFIX`; the generation prompt will prepend those terms automatically.

`LORA_FUSE=True` permanently fuses the adapter into the loaded pipeline. In that mode the configured `LORA_SCALE` is applied during `fuse_lora()` only. When `LORA_FUSE=False`, the scale is applied through `set_adapters()`.

## Suggested first run

Use the smoke preset first and inspect accept rate, reject reasons, and debug strips before changing validation thresholds. Treat LoRA as a generator adapter; keep placement, scale correction, compositing, and YOLO validation unchanged until the smoke run shows a clear pattern.

## Training and `.pt` model artifact

`sd35_lora_training.py` adds an explicit training/export entrypoint while keeping the augmentation runner focused on inference and evaluation.

Default training settings are conservative for an NVIDIA T4 16 GB: SD3.5 Medium, 512 px, fp16, batch size 1, gradient accumulation 4, gradient checkpointing, 8-bit Adam, frozen text encoders, and rank-8 attention LoRA.

A dry run writes the command and config without starting a long training job:

```python
from sd35_lora_training import run_lora_training
run_lora_training(dry_run=True)
```

A real run expects a prepared captioned/cropped training folder at `LORA_TRAINING_DATA_DIR` and a local Diffusers SD3 LoRA training script, for example `train_dreambooth_lora_sd3.py`. After training finishes, the helper exports the native adapter plus:

```text
/kaggle/working/sd35m-pedestrian-v1/pytorch_lora_weights.pt
/kaggle/working/sd35m-pedestrian-v1/training_config.json
/kaggle/working/sd35m-pedestrian-v1/training_provenance.json
```

`export_outputs()` now includes `LORA_TRAINING_OUTPUT_DIR`, so the final zip contains the `.pt` model artifact whenever the training output directory exists.
