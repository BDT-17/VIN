"""Concept (text->image) SD3.5 LoRA trainer.

This is the SIMPLE flow the user actually wants: train ONE LoRA so the base model
learns to GENERATE images like the PIPE "add a person" subset. It is a plain
DreamBooth-style text->image LoRA — there is NO mask, NO source image, NO
input_proj, and NO segment/paste step. Contrast with train_maskfree_edit.py,
which trains an EDIT model on before/after pairs; that whole machinery is dropped
here.

Conditioning (standard SD3 flow matching):
    target_lat = VAE(image)
    noisy = (1-sigma)*target_lat + sigma*noise
    pred  = transformer(hidden_states=noisy, timestep, encoder_hidden_states=...)
    loss  = weighting(sigma) * (pred - (noise - target_lat))^2

ONE artifact is exported and required at inference:
    adapter/pytorch_lora_weights.safetensors

Memory plan mirrors the edit trainer (T4-friendly): precompute text embeds to
disk, drop the 3 text encoders, keep VAE + transformer resident, bf16 compute /
fp32 trainable. Kaggle-GPU only.
"""

import json
import math
import re
from pathlib import Path

import torch

from ..data.config import load_train_config
from .spike_inpaint_edit import _vae_encode
from .provenance import write_gpu_info, write_pip_freeze, sha256_file


def _next_run_dir(model_root: Path, model_name: str) -> Path:
    base = Path(model_root) / model_name
    base.mkdir(parents=True, exist_ok=True)
    existing = [int(m.group(1)) for p in base.glob("run_*")
                if (m := re.match(r"run_(\d+)$", p.name))]
    run_dir = base / f"run_{(max(existing)+1) if existing else 1:03d}"
    (run_dir / "adapter").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_artifacts(run_dir, transformer, tag="adapter"):
    """Save the LoRA safetensors only (no input_proj — this is plain t2i)."""
    from diffusers import StableDiffusion3Pipeline
    from peft.utils import get_peft_model_state_dict
    out = run_dir / tag
    out.mkdir(parents=True, exist_ok=True)
    lora_sd = get_peft_model_state_dict(transformer)
    StableDiffusion3Pipeline.save_lora_weights(save_directory=str(out),
                                               transformer_lora_layers=lora_sd)
    return out


def run_training(work_dir, base_model_id=None, hf_token=None, train_config_path=None,
                 max_train_steps=None, num_train_samples=None, device="cuda"):
    cfg = load_train_config(train_config_path or
                            Path(__file__).resolve().parent.parent / "configs" / "concept_lora_train.yaml")
    if max_train_steps is not None:
        cfg["max_train_steps"] = int(max_train_steps)
    if num_train_samples is not None:
        cfg["num_train_samples"] = int(num_train_samples)
    base_model_id = base_model_id or cfg["base_model_id"]
    resolution = int(cfg.get("resolution", 512))
    grad_accum = int(cfg.get("grad_accum_steps", 4))

    from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
    from diffusers.training_utils import (compute_density_for_timestep_sampling,
                                          compute_loss_weighting_for_sd3)
    from peft import LoraConfig
    from .concept_lora_dataset import load_images

    run_dir = _next_run_dir(Path(work_dir) / "models", cfg["model_name"])
    (run_dir / "training_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    write_gpu_info(run_dir); write_pip_freeze(run_dir)

    print(f"[data] streaming {cfg['num_train_samples']} PIPE-{cfg['pipe_split']} person images (concept t2i)...")
    items = load_images(num_samples=cfg["num_train_samples"], split=cfg["pipe_split"],
                        person_only=cfg.get("person_only", True),
                        caption_prefix=cfg.get("caption_prefix", "a photo of "),
                        trigger_token=cfg.get("trigger_token", ""),
                        diff_thresh=cfg.get("diff_thresh", 25),
                        min_change_pixels=cfg.get("min_change_pixels", 64))
    if not items:
        raise RuntimeError("No PIPE person images loaded.")
    print(f"[data] {len(items)} images ready")

    compute_dtype = torch.bfloat16
    _kw = {"torch_dtype": compute_dtype}
    if hf_token and not Path(base_model_id).exists():
        _kw["token"] = hf_token
    pipe = StableDiffusion3Pipeline.from_pretrained(base_model_id, **_kw)

    def to_t(img):
        import numpy as np
        a = np.asarray(img.convert("RGB").resize((resolution, resolution)), dtype=np.float32) / 255.0
        t = torch.from_numpy(a).permute(2, 0, 1) * 2 - 1
        return t.unsqueeze(0).to(device, dtype=compute_dtype)

    # ---- precompute embeddings to DISK, then drop text encoders ----
    emb_dir = run_dir / "_emb_cache"; emb_dir.mkdir(exist_ok=True)
    uniq_prompts = list(dict.fromkeys(it["caption"] for it in items))
    prompt_to_id = {p: i for i, p in enumerate(uniq_prompts)}
    pipe.text_encoder.to(device); pipe.text_encoder_2.to(device)
    if pipe.text_encoder_3 is not None:
        pipe.text_encoder_3.to(device)

    def _encode_to(pid, text):
        pe, _, pooled, _ = pipe.encode_prompt(
            text, prompt_2=text, prompt_3=text, device=device,
            num_images_per_prompt=1, do_classifier_free_guidance=False)
        torch.save({"pe": pe.to("cpu"), "pooled": pooled.to("cpu")}, emb_dir / f"{pid}.pt")
        del pe, pooled

    with torch.no_grad():
        for p, pid in prompt_to_id.items():
            _encode_to(pid, p)
    pipe.text_encoder = pipe.text_encoder_2 = pipe.text_encoder_3 = None
    import gc; gc.collect(); torch.cuda.empty_cache()
    print(f"  cached {len(uniq_prompts)} unique caption embeds to disk")

    vae, transformer = pipe.vae, pipe.transformer
    vae.to(device); transformer.to(device)
    noise_sched = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    vae.requires_grad_(False); transformer.requires_grad_(False)
    transformer.enable_gradient_checkpointing()

    transformer.add_adapter(LoraConfig(
        r=int(cfg["rank"]), lora_alpha=int(cfg["rank"]), init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    for p in lora_params:
        p.data = p.data.float()
    opt = torch.optim.AdamW(lora_params, lr=float(cfg["learning_rate"]))

    num_train_ts = noise_sched.config.num_train_timesteps
    sched_sigmas = noise_sched.sigmas.to(device=device, dtype=torch.float32)
    sched_ts = noise_sched.timesteps.to(device)

    def get_sigma(timesteps):
        idx = [(sched_ts == t).nonzero().item() for t in timesteps]
        s = sched_sigmas[idx].flatten()
        while s.ndim < 4:
            s = s.unsqueeze(-1)
        return s

    steps = int(cfg["max_train_steps"])
    ckpt_every = int(cfg["checkpointing_steps"])
    clip = float(cfg.get("grad_clip_norm", 1.0))
    metrics_path = run_dir / "metrics.jsonl"
    mf = open(metrics_path, "w", encoding="utf-8")

    losses = []
    transformer.train()
    opt.zero_grad()
    for step in range(steps):
        i = step % len(items)
        it = items[i]
        image = to_t(it["image"])
        with torch.no_grad():
            target_lat = _vae_encode(vae, image)
            _emb = torch.load(emb_dir / f"{prompt_to_id[it['caption']]}.pt", weights_only=True)
            prompt_embeds, pooled = _emb["pe"].to(device), _emb["pooled"].to(device)

        noise = torch.randn_like(target_lat)
        u = compute_density_for_timestep_sampling(
            weighting_scheme=cfg.get("weighting_scheme", "logit_normal"),
            batch_size=1, logit_mean=float(cfg.get("logit_mean", 0.0)),
            logit_std=float(cfg.get("logit_std", 1.0)), mode_scale=1.29)
        idx = (u * num_train_ts).long().to(device)
        timesteps = sched_ts[idx]
        sigma = get_sigma(timesteps).to(target_lat.dtype)
        noisy = (1.0 - sigma) * target_lat + sigma * noise

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = transformer(hidden_states=noisy, timestep=timesteps,
                               encoder_hidden_states=prompt_embeds,
                               pooled_projections=pooled, return_dict=False)[0]
        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=cfg.get("weighting_scheme", "logit_normal"), sigmas=sigma)
        target_v = noise - target_lat
        loss = (weighting.float() * (pred.float() - target_v.float()) ** 2).mean()

        (loss / grad_accum).backward()
        lv = loss.item()
        if not math.isfinite(lv):
            mf.close()
            raise RuntimeError(f"Training diverged: non-finite loss at step {step}")
        losses.append(lv)
        mf.write(json.dumps({"step": step, "loss": round(lv, 6)}) + "\n"); mf.flush()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(lora_params, clip)
            opt.step(); opt.zero_grad()

        if step % 25 == 0:
            print(f"  step {step:4d}/{steps}  loss {lv:.4f}")
        if ckpt_every and (step + 1) % ckpt_every == 0 and (step + 1) < steps:
            _save_artifacts(run_dir, transformer, tag=f"checkpoints/checkpoint-{step+1}")
            print(f"  [ckpt] checkpoint-{step+1}")

    if steps % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(lora_params, clip)
        opt.step(); opt.zero_grad()
    mf.close()
    import shutil as _sh
    _sh.rmtree(emb_dir, ignore_errors=True)

    out = _save_artifacts(run_dir, transformer, tag="adapter")
    adapter_file = out / "pytorch_lora_weights.safetensors"
    provenance = {
        "base_model_id": base_model_id,
        "model_name": cfg["model_name"],
        "flow": "sd35_concept_lora",
        "conditioning": "text_to_image",
        "rank": cfg["rank"], "learning_rate": cfg["learning_rate"],
        "max_train_steps": steps, "num_train_samples": len(items),
        "grad_accum_steps": grad_accum, "effective_batch": grad_accum,
        "caption_prefix": cfg.get("caption_prefix", "a photo of "),
        "trigger_token": cfg.get("trigger_token", ""),
        "weighting_scheme": cfg.get("weighting_scheme"),
        "lora_sha256": sha256_file(adapter_file) if adapter_file.exists() else "",
        "loss_first25_mean": round(sum(losses[:25]) / min(25, len(losses)), 4),
        "loss_last25_mean": round(sum(losses[-25:]) / min(25, len(losses)), 4),
        "requires_input_proj": False,
        "mask_free": True,
    }
    (run_dir / "training_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")

    print(f"\nTRAINING COMPLETE -> {run_dir}")
    print(f"  LoRA: {adapter_file}")
    print(f"  loss {provenance['loss_first25_mean']} -> {provenance['loss_last25_mean']}")

    del transformer, vae, opt, lora_params, pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"run_dir": run_dir, "adapter_dir": out, "provenance": provenance}
