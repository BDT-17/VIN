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
