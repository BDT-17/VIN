"""SD3.5 inpaint-EDIT inference: base + LoRA + the trained input adapter.

This is the inference counterpart of train_inpaint_edit.py. It runs the FULL
denoise loop (not a single step like the spike), reconstructing the edit
conditioning at EVERY step because the latent changes each step:

    cond_t = input_adapter(concat[latent_t, source_lat, mask_lat])   # 33 -> 16
    v_t    = transformer(hidden_states=cond_t, timestep=t, ...)
    latent = scheduler.step(v_t, t, latent)

BOTH artifacts are required (training_provenance.requires_input_adapter == true):
    adapter/pytorch_lora_weights.safetensors
    adapter/input_adapter.pt

After decoding, hard-restore keeps the original background outside the mask:
    result = source*(1-mask) + generated*mask
so the background is preserved 100% by construction; the LoRA+adapter only fill
the masked region to match the photo.

Kaggle-GPU only (needs SD3.5).
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .sd35_inpaint_runner import hard_restore
from ..train.spike_inpaint_edit import InputAdapter, _vae_encode


class SD35EditRunner:
    def __init__(self, base_model_id, adapter_dir, device="cuda", hf_token=None):
        self.base_model_id = base_model_id
        self.adapter_dir = Path(adapter_dir)
        self.device = device
        self.hf_token = hf_token
        self.compute_dtype = torch.bfloat16
        self.pipe = None
        self.input_adapter = None

    def load(self):
        from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
        kw = {"torch_dtype": self.compute_dtype}
        if self.hf_token and not Path(self.base_model_id).exists():
            kw["token"] = self.hf_token
        self.pipe = StableDiffusion3Pipeline.from_pretrained(self.base_model_id, **kw)
        # SD3.5 + T5-XXL does NOT fit resident on a 16GB T4. Keep everything on CPU
        # at load; edit() moves the text encoders to GPU only for encode_prompt, then
        # frees them, keeping just transformer + VAE resident for the denoise loop
        # (the same memory plan the trainer uses).
        self.pipe.vae.enable_slicing()
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config)

        # LoRA
        lora_file = self.adapter_dir / "pytorch_lora_weights.safetensors"
        if not lora_file.exists():
            raise FileNotFoundError(f"LoRA not found: {lora_file}")
        self.pipe.load_lora_weights(str(self.adapter_dir),
                                    weight_name="pytorch_lora_weights.safetensors")

        # load_lora_weights pulls the 3 text encoders (incl. T5-XXL ~9GB) onto the
        # GPU, which fills a 16GB T4 before precompute_embeds even runs. Force the
        # whole pipeline back to CPU so precompute_embeds starts from an empty GPU
        # and can run its plan: encoders -> GPU -> encode -> drop -> transformer+VAE.
        self.pipe.to("cpu")
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # input adapter (the 33->16 edit conv) — REQUIRED
        ia_file = self.adapter_dir / "input_adapter.pt"
        if not ia_file.exists():
            raise FileNotFoundError(
                f"input_adapter.pt not found: {ia_file}. The edit flow is useless "
                "without it (training_provenance.requires_input_adapter == true).")
        self.input_adapter = InputAdapter().to(self.device, dtype=self.compute_dtype)
        self.input_adapter.load_state_dict(torch.load(ia_file, weights_only=True))
        self.input_adapter.eval()
        self._embeds = {}        # prompt -> (prompt_embeds_cpu, pooled_cpu)
        self._encoders_dropped = False
        return self

    @torch.no_grad()
    def precompute_embeds(self, prompts):
        """Encode ALL prompts with the text encoders on GPU, cache to CPU, then
        DROP the text encoders and move transformer + VAE to GPU. SD3.5 + T5-XXL
        cannot be resident alongside the transformer on a 16GB T4, so this must
        run ONCE before any edit() call (same plan the trainer uses)."""
        if self._encoders_dropped:
            return
        device = self.device
        for e in (self.pipe.text_encoder, self.pipe.text_encoder_2, self.pipe.text_encoder_3):
            if e is not None:
                e.to(device)
        e = None                                   # drop the loop ref to the last encoder
        for p in dict.fromkeys(prompts):           # de-dup, preserve order
            pe, _, pooled, _ = self.pipe.encode_prompt(
                p, prompt_2=p, prompt_3=p, device=device,
                num_images_per_prompt=1, do_classifier_free_guidance=False)
            self._embeds[p] = (pe.to("cpu"), pooled.to("cpu"))
            del pe, pooled
        # Free the encoders for real. Just setting the attrs to None and calling
        # empty_cache() leaves ~9GB (T5-XXL) resident: the loop var `e` and the
        # pipeline still ref them, so the CUDA storage is not released and the VAE
        # encode below OOMs. Move each encoder back to CPU, del it, THEN clear.
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
    def edit(self, source_img, mask_img, prompt, negative_prompt="", seed=42,
             num_inference_steps=30, resolution=512, hard_restore_bg=True):
        from PIL import Image
        import numpy as np

        device, dtype = self.device, self.compute_dtype
        src = source_img.convert("RGB").resize((resolution, resolution))
        msk = mask_img.convert("L").resize((resolution, resolution))

        # embeddings come from the one-time precompute (text encoders are gone)
        if prompt not in self._embeds:
            self.precompute_embeds([prompt])
        pe_cpu, pooled_cpu = self._embeds[prompt]
        prompt_embeds, pooled = pe_cpu.to(device), pooled_cpu.to(device)

        def to_t(img, mode):
            a = np.asarray(img, dtype=np.float32) / 255.0
            t = (torch.from_numpy(a).permute(2, 0, 1) * 2 - 1) if mode == "RGB" else torch.from_numpy(a)[None]
            return t.unsqueeze(0).to(device, dtype=dtype)

        src_t = to_t(src, "RGB"); mask_t = to_t(msk, "L")
        source_lat = _vae_encode(self.pipe.vae, src_t * (1 - mask_t), mode=True)  # PIPE: cond image uses .mode()
        mask_lat = F.interpolate(mask_t, size=source_lat.shape[-2:])

        gen = torch.Generator(device=device).manual_seed(int(seed))
        latents = torch.randn(source_lat.shape, generator=gen, device=device, dtype=dtype)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        for t in self.scheduler.timesteps:
            timestep = t.expand(latents.shape[0])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                cond = self.input_adapter(latents, source_lat, mask_lat)
                v = self.pipe.transformer(
                    hidden_states=cond, timestep=timestep,
                    encoder_hidden_states=prompt_embeds, pooled_projections=pooled,
                    return_dict=False)[0]
            latents = self.scheduler.step(v, t, latents, return_dict=False)[0]

        lat = (latents / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        img = self.pipe.vae.decode(lat.to(dtype), return_dict=False)[0]
        img = (img / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
        out = Image.fromarray((img * 255).round().astype("uint8"))

        if hard_restore_bg:
            out = hard_restore(out, src, msk)
        return out


def load_edit_runner_from_run(run_dir, base_model_id=None, device="cuda", hf_token=None):
    """Build a runner from a training run_dir, validating provenance."""
    run_dir = Path(run_dir)
    prov = json.loads((run_dir / "training_provenance.json").read_text(encoding="utf-8"))
    base = base_model_id or prov["base_model_id"]
    return SD35EditRunner(base, run_dir / "adapter", device=device, hf_token=hf_token).load(), prov
