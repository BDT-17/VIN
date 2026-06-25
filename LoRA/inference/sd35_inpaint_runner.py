"""Raw SD3.5 inpaint runner for the paired baseline-vs-LoRA evaluation.

This is intentionally minimal: it calls StableDiffusion3InpaintPipeline directly.
It does NOT do hard background restoration, mask refinement, harmonization, or
ghost validation — those belong to the inpaint/ AI-Replace flow, not here. The
only difference between conditions B0 and B1 is the trigger token in the prompt.

  B0: SD3.5 Inpaint            (no LoRA)
  B1: SD3.5 Inpaint + vinped LoRA

Every other input (image, mask, prompt fields, seed, resolution, strength,
guidance, steps, negative prompt) is held identical.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from ..data.config import load_inpaint_eval_config, load_prompt_config


def load_cases(eval_release: Path) -> List[Dict]:
    cases_path = Path(eval_release) / "cases.jsonl"
    if not cases_path.exists():
        raise FileNotFoundError(f"cases.jsonl not found: {cases_path}")
    return [json.loads(l) for l in cases_path.read_text(encoding="utf-8").splitlines() if l.strip()]


def render_inpaint_prompt(template: str, fields: Dict, trigger: str) -> str:
    text = template.replace("{trigger}", trigger)
    for k, v in (fields or {}).items():
        text = text.replace("{" + k + "}", v or "")
    parts = [p.strip() for p in text.split(",")]
    return ", ".join(p for p in parts if p)


class SD35InpaintRunner:
    """Thin wrapper around StableDiffusion3InpaintPipeline."""

    def __init__(self, base_model_id: str, device: str = "cuda"):
        self.base_model_id = base_model_id
        self.device = device
        self.pipe = None
        self._lora_loaded = False

    def load(self):
        import torch
        from diffusers import StableDiffusion3InpaintPipeline
        self.pipe = StableDiffusion3InpaintPipeline.from_pretrained(
            self.base_model_id, torch_dtype=torch.float16)
        self.pipe = self.pipe.to(self.device)
        return self

    def attach_lora(self, adapter_path: Path, adapter_name: str, weight: float):
        adapter_path = Path(adapter_path)
        self.pipe.load_lora_weights(str(adapter_path.parent),
                                    weight_name=adapter_path.name,
                                    adapter_name=adapter_name)
        self.pipe.set_adapters([adapter_name], [weight])
        self._lora_loaded = True

    def detach_lora(self):
        if self._lora_loaded:
            try:
                self.pipe.disable_lora()
            except Exception:
                pass
            self._lora_loaded = False

    def run_case(self, case: Dict, eval_release: Path, prompt: str, negative: str,
                 seed: int, cfg: dict) -> Dict:
        import torch
        from PIL import Image
        eval_release = Path(eval_release)
        image = Image.open(eval_release / case["image_path"]).convert("RGB")
        mask = Image.open(eval_release / case["mask_path"]).convert("L")
        res = int(cfg.get("resolution", 512))
        image_r = image.resize((res, res))
        mask_r = mask.resize((res, res))

        peak_before = _cuda_reset_peak()
        import time
        t0 = time.time()
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=image_r,
            mask_image=mask_r,
            strength=float(cfg.get("strength", 1.0)),
            guidance_scale=float(cfg.get("guidance_scale", 7.5)),
            num_inference_steps=int(cfg.get("num_inference_steps", 30)),
            generator=torch.Generator(device=self.device).manual_seed(int(seed)),
        ).images[0]
        runtime = time.time() - t0
        # restore to original resolution for metric comparison vs reference
        result = result.resize(image.size)
        return {"image": result, "runtime_seconds": round(runtime, 3),
                "cuda_peak_mb": _cuda_peak_mb()}


def _cuda_reset_peak():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _cuda_peak_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
    except Exception:
        pass
    return None
