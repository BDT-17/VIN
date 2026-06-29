"""Concept (text->image) SD3.5 LoRA inference: base + LoRA, plain generation.

Inference counterpart of train_concept_lora.py. There is NO source image, NO
input_proj, NO segment/paste — just the base SD3.5 pipeline with the trained LoRA
merged in, generating from a text prompt:

    image = pipe(prompt, ...)

Only ONE artifact is required (training_provenance.requires_input_proj == false):
    adapter/pytorch_lora_weights.safetensors

Kaggle-GPU only.
"""

import json
from pathlib import Path

import torch


class SD35ConceptRunner:
    def __init__(self, base_model_id, adapter_dir, device="cuda", hf_token=None):
        self.base_model_id = base_model_id
        self.adapter_dir = Path(adapter_dir)
        self.device = device
        self.hf_token = hf_token
        self.compute_dtype = torch.bfloat16
        self.pipe = None

    def load(self):
        from diffusers import StableDiffusion3Pipeline
        kw = {"torch_dtype": self.compute_dtype}
        if self.hf_token and not Path(self.base_model_id).exists():
            kw["token"] = self.hf_token
        self.pipe = StableDiffusion3Pipeline.from_pretrained(self.base_model_id, **kw)

        lora_file = self.adapter_dir / "pytorch_lora_weights.safetensors"
        if not lora_file.exists():
            raise FileNotFoundError(f"LoRA not found: {lora_file}")
        self.pipe.load_lora_weights(str(self.adapter_dir),
                                    weight_name="pytorch_lora_weights.safetensors")
        self.pipe.to(self.device)
        self.pipe.vae.enable_slicing()
        return self

    @torch.no_grad()
    def generate(self, prompt, negative_prompt="", seed=42, num_inference_steps=28,
                 guidance_scale=7.0, resolution=512):
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        out = self.pipe(
            prompt=prompt, negative_prompt=negative_prompt or None,
            num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
            width=resolution, height=resolution, generator=gen,
        )
        return out.images[0]


def load_concept_runner_from_run(run_dir, base_model_id=None, device="cuda", hf_token=None):
    """Build a concept runner from a training run_dir, validating provenance."""
    run_dir = Path(run_dir)
    prov = json.loads((run_dir / "training_provenance.json").read_text(encoding="utf-8"))
    base = base_model_id or prov["base_model_id"]
    runner = SD35ConceptRunner(base, run_dir / "adapter", device=device, hf_token=hf_token).load()
    return runner, prov
