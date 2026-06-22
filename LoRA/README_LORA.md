# LoRA Experiment Flow

This folder is an isolated copy of the current SD3.5 CityPersons augmentation flow. The root pipeline is intentionally untouched.

## What changed here

- Outputs are written to `/kaggle/working/sd35_citypersons_lora`.
- Metrics are written to `/kaggle/working/lora_metrics`.
- Autotune snapshots are written to `/kaggle/working/lora_autotune_snapshots`.
- Optional LoRA adapter loading is available in `sd35_model.py` for both img2img and inpaint pipelines.
- Manifest rows include LoRA metadata for traceability.

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
}
```

Keep `LORA_ENABLED=False` to run the copied baseline behavior.

## Suggested first run

Use the smoke preset first and inspect accept rate, reject reasons, and debug strips before changing validation thresholds. Treat LoRA as a generator adapter; keep placement, scale correction, compositing, and YOLO validation unchanged until the smoke run shows a clear pattern.
