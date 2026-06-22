# AI Replace Flow

Standalone Pokecut-style inpainting flow for VIN experiments. This lives under
`inpaint/` and does not modify the rollback-era V5 pipeline.

## What It Does

1. Builds an insertion mask from a bbox.
2. Refines hard and soft masks.
3. Runs SD inpainting when a diffusers pipeline is available.
4. Restores original pixels outside the hard mask after generation.
5. Extracts a person mask through YOLO when provided, or uses a dry-run fallback.
6. Applies layer-aware harmonization.
7. Runs ghost/background validation.
8. Emits manifest-ready metrics.

The invariant is simple:

```text
Generated background is untrusted.
Original background is trusted.
Pixels outside the hard insertion mask are restored from the original image.
```

## Files

- `config.py`: local AI Replace config.
- `sd35_mask_refinement.py`: bbox to mask, dilation, blur, hard restore.
- `sd35_harmonization.py`: mask layers, color transfer, contact shadow.
- `sd35_ghost_detection.py`: opacity, contrast and seam scoring.
- `sd35_ai_replace.py`: orchestration class.
- `test_background_preservation.py`: standalone hard-restore test.

## Minimal Dry Run

```python
from PIL import Image
from inpaint.sd35_ai_replace import AIReplacePipeline

image = Image.open("input.png").convert("RGB")
pipe = AIReplacePipeline(pipe=None, device="cpu")
result = pipe.run(image, bbox=(180, 180, 300, 430), seed=42)
print(result.manifest_row)
```

## Real Inpainting

```python
from inpaint.sd35_ai_replace import AIReplacePipeline

pipe = AIReplacePipeline.from_pretrained(device="cuda")
result = pipe.run(image, bbox=(180, 180, 300, 430), seed=42, yolo_segmenter=yolo)
AIReplacePipeline.save_result(result, "ai_replace_outputs", stem="sample_0001")
```

## Dependencies

Required at runtime: numpy, Pillow. Real SD3.5 inpainting also needs torch, a recent diffusers build with StableDiffusion3InpaintPipeline, transformers, accelerate, safetensors, and optionally ultralytics for YOLO segmentation.

Default inpainting model: `stabilityai/stable-diffusion-3.5-medium` via `StableDiffusion3InpaintPipeline`. The old `stabilityai/stable-diffusion-2-inpainting` repo may return 404 on Hugging Face and is automatically remapped to the community mirror; set `AI_REPLACE_MODEL_ID=/path/to/local/model` or another accessible repo if needed.

## Verification

```bash
python inpaint/test_background_preservation.py
```

Expected:

```text
background preservation ok
```

## Full Smoke Test

Dry-run wiring test:

```bash
python inpaint/smoke_runner.py --input-dir /path/to/images --output-dir ./ai_replace_smoke --num-images 20 --no-yolo
```

Kaggle/model smoke test with repo update first:

```bash
python inpaint/smoke_runner.py \
  --git-pull \
  --input-dir /kaggle/input/your-dataset \
  --output-dir /kaggle/working/ai_replace_smoke \
  --num-images 20 \
  --load-model
```

Outputs:

- `manifest.jsonl`
- `manifest.csv`
- `metrics/metrics_summary.json`
- `metrics/metrics_summary.csv`
- `previews/*_original.png`
- `previews/*_mask.png`
- `previews/*_generated_raw.png`
- `previews/*_object_mask.png`
- `previews/*_harmonized.png`
- `previews/*_diff_outside_mask.png`