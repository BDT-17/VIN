# Vendored Diffusers training script

The SD3.5 LoRA trainer is **pinned**, not downloaded from `main` each run.

## Pinned file

```
LoRA/vendor/diffusers/v0.31.0/train_dreambooth_lora_sd3.py
```

Source (pinned tag `v0.31.0`):
<https://raw.githubusercontent.com/huggingface/diffusers/v0.31.0/examples/dreambooth/train_dreambooth_lora_sd3.py>

## How to vendor it

The raw script is large and is not committed by the assistant. Fetch it once and
commit it under the pinned path:

```bash
mkdir -p LoRA/vendor/diffusers/v0.31.0
curl -sSL \
  https://raw.githubusercontent.com/huggingface/diffusers/v0.31.0/examples/dreambooth/train_dreambooth_lora_sd3.py \
  -o LoRA/vendor/diffusers/v0.31.0/train_dreambooth_lora_sd3.py
git add LoRA/vendor/diffusers/v0.31.0/train_dreambooth_lora_sd3.py
```

`LoRA/configs/lora_train.yaml: train_script` points at this path. To bump the
pin, add a new `vN.N.N/` folder and update the config — never edit in place.
