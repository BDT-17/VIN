"""Mask-FREE SD3.5 edit inference (IP2P/PIPE-style): base + LoRA + input_proj.

Inference counterpart of train_maskfree_edit.py. The model is conditioned on the
SOURCE IMAGE + instruction only (no mask, no hard-restore). It runs the full
denoise loop, reconstructing the edit conditioning every step:

    cond_t = input_proj(concat[latent_t, source_lat])     # 32 -> 16
    v_t    = transformer(hidden_states=cond_t, timestep=t, ...)
    latent = scheduler.step(v_t, t, latent)

Classifier-free guidance (IP2P, two scales): each step runs the transformer 3×
(uncond, image-only, full) and combines

    pred = e_uncond
         + s_image * (e_image - e_uncond)        # source-image adherence
         + s_text  * (e_full  - e_image)         # instruction adherence

so s_image controls how much the output stays the source, s_text how strongly the
instruction is applied. Defaults s_image=1.5, s_text=7.5 (IP2P-style).

BOTH artifacts are required (training_provenance.requires_input_proj == true):
    adapter/pytorch_lora_weights.safetensors
    adapter/input_proj.pt

There is NO hard-restore: the model owns the whole image. Background preservation
is whatever the model learned — not guaranteed byte-exact (mask-free tradeoff).

Kaggle-GPU only.
"""

import json
from pathlib import Path

import torch

from ..train.spike_inpaint_edit import InputProj, _vae_encode


class SD35MaskFreeRunner:
    def __init__(self, base_model_id, adapter_dir, device="cuda", hf_token=None):
        self.base_model_id = base_model_id
        self.adapter_dir = Path(adapter_dir)
        self.device = device
        self.hf_token = hf_token
        self.compute_dtype = torch.bfloat16
        self.pipe = None
        self.input_proj = None

    def load(self):
        from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
        kw = {"torch_dtype": self.compute_dtype}
        if self.hf_token and not Path(self.base_model_id).exists():
            kw["token"] = self.hf_token
        self.pipe = StableDiffusion3Pipeline.from_pretrained(self.base_model_id, **kw)
        self.pipe.vae.enable_slicing()
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config)

        lora_file = self.adapter_dir / "pytorch_lora_weights.safetensors"
        if not lora_file.exists():
            raise FileNotFoundError(f"LoRA not found: {lora_file}")
        self.pipe.load_lora_weights(str(self.adapter_dir),
                                    weight_name="pytorch_lora_weights.safetensors")
        # load_lora_weights pulls the text encoders (incl. T5-XXL) onto the GPU,
        # filling a 16GB T4 before precompute. Force everything to CPU so
        # precompute_embeds starts from an empty GPU (same fix as the mask-based runner).
        self.pipe.to("cpu")
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ip_file = self.adapter_dir / "input_proj.pt"
        if not ip_file.exists():
            raise FileNotFoundError(
                f"input_proj.pt not found: {ip_file}. The mask-free edit flow is "
                "useless without it (training_provenance.requires_input_proj == true).")
        self.input_proj = InputProj().to(self.device, dtype=self.compute_dtype)
        self.input_proj.load_state_dict(torch.load(ip_file, weights_only=True))
        self.input_proj.eval()
        self._embeds = {}          # prompt -> (pe_cpu, pooled_cpu); "" = uncond
        self._encoders_dropped = False
        return self

    @torch.no_grad()
    def precompute_embeds(self, prompts):
        """Encode all prompts + the empty prompt (for CFG) on GPU, cache to CPU,
        then DROP the text encoders and move transformer + VAE to GPU. Runs ONCE."""
        if self._encoders_dropped:
            return
        device = self.device
        for e in (self.pipe.text_encoder, self.pipe.text_encoder_2, self.pipe.text_encoder_3):
            if e is not None:
                e.to(device)
        e = None
        for p in list(dict.fromkeys(list(prompts) + [""])):   # ensure "" (uncond)
            pe, _, pooled, _ = self.pipe.encode_prompt(
                p, prompt_2=p, prompt_3=p, device=device,
                num_images_per_prompt=1, do_classifier_free_guidance=False)
            self._embeds[p] = (pe.to("cpu"), pooled.to("cpu"))
            del pe, pooled
        for attr in ("text_encoder", "text_encoder_2", "text_encoder_3"):
            enc = getattr(self.pipe, attr, None)
            if enc is not None:
                enc.to("cpu")
            setattr(self.pipe, attr, None)
            del enc
        import gc; gc.collect(); torch.cuda.empty_cache()
        self.pipe.transformer.to(device); self.pipe.vae.to(device)
        self._encoders_dropped = True

    @torch.no_grad()
    def edit(self, source_img, prompt, seed=42, num_inference_steps=30,
             resolution=512, s_image=1.5, s_text=7.5):
        from PIL import Image
        import numpy as np

        device, dtype = self.device, self.compute_dtype
        src = source_img.convert("RGB").resize((resolution, resolution))

        if prompt not in self._embeds:
            self.precompute_embeds([prompt])
        pe_full, pooled_full = (t.to(device) for t in self._embeds[prompt])
        pe_empty, pooled_empty = (t.to(device) for t in self._embeds[""])

        a = np.asarray(src, dtype=np.float32) / 255.0
        src_t = (torch.from_numpy(a).permute(2, 0, 1) * 2 - 1).unsqueeze(0).to(device, dtype=dtype)
        source_lat = _vae_encode(self.pipe.vae, src_t)
        zero_lat = torch.zeros_like(source_lat)              # dropped-image condition

        gen = torch.Generator(device=device).manual_seed(int(seed))
        latents = torch.randn(source_lat.shape, generator=gen, device=device, dtype=dtype)

        def predict(lat, src_lat, pe, pooled):
            cond = self.input_proj(lat, src_lat)
            return self.pipe.transformer(
                hidden_states=cond, timestep=timestep,
                encoder_hidden_states=pe, pooled_projections=pooled,
                return_dict=False)[0]

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        for t in self.scheduler.timesteps:
            timestep = t.expand(latents.shape[0])
            # Three CFG forwards run SEQUENTIALLY and we free each activation set
            # before the next so the T4 only ever holds one transformer pass worth
            # of activations (three live at once can OOM a 16GB T4).
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                e_uncond = predict(latents, zero_lat, pe_empty, pooled_empty)
                e_image = predict(latents, source_lat, pe_empty, pooled_empty)
                e_full = predict(latents, source_lat, pe_full, pooled_full)
                v = (e_uncond
                     + s_image * (e_image - e_uncond)
                     + s_text * (e_full - e_image))
            del e_uncond, e_image, e_full
            latents = self.scheduler.step(v, t, latents, return_dict=False)[0]

        lat = (latents / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        img = self.pipe.vae.decode(lat.to(dtype), return_dict=False)[0]
        img = (img / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
        return Image.fromarray((img * 255).round().astype("uint8"))


def load_maskfree_runner_from_run(run_dir, base_model_id=None, device="cuda", hf_token=None):
    """Build a mask-free runner from a training run_dir, validating provenance."""
    run_dir = Path(run_dir)
    prov = json.loads((run_dir / "training_provenance.json").read_text(encoding="utf-8"))
    base = base_model_id or prov["base_model_id"]
    runner = SD35MaskFreeRunner(base, run_dir / "adapter", device=device, hf_token=hf_token).load()
    return runner, prov
