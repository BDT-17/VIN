"""Full SD3.5 inpaint-EDIT LoRA trainer.

Learns to fill a masked region so the inserted object matches the photo, from
PIPE (source, target, mask) pairs. Extends the validated spike with the standard
SD3 sampling/weighting and proper artifact export.

Conditioning (validated by the overfit spike, collapse_ratio 0.18):
    target_lat = VAE(target);  source_lat = VAE(source * (1 - mask))
    noisy = (1-sigma)*target_lat + sigma*noise
    cond  = input_adapter(concat[noisy, source_lat, mask_lat])   # 33 -> 16
    pred  = transformer(hidden_states=cond, timestep, encoder_hidden_states=...)
    loss  = weighting(sigma) * (pred - (noise - target_lat))^2

TWO artifacts are exported and BOTH are required at inference:
    adapter/pytorch_lora_weights.safetensors   (LoRA on attention)
    adapter/input_adapter.pt                    (the Conv 33->16)

Memory: precompute text embeddings, drop the 3 text encoders, keep only
VAE + transformer resident (fits a 16GB T4). bf16 compute, fp32 trainable params.
Kaggle-GPU only.
"""

import json
import math
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from ..data.config import load_train_config
from .spike_inpaint_edit import InputAdapter, _vae_encode
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


def _save_artifacts(run_dir, transformer, input_adapter, tag="adapter"):
    """Save LoRA safetensors + the input-adapter Conv. tag='adapter' for final,
    or 'checkpoints/checkpoint-N' for intermediate."""
    from diffusers import StableDiffusion3Pipeline
    from peft.utils import get_peft_model_state_dict
    out = run_dir / tag
    out.mkdir(parents=True, exist_ok=True)
    lora_sd = get_peft_model_state_dict(transformer)
    StableDiffusion3Pipeline.save_lora_weights(save_directory=str(out),
                                               transformer_lora_layers=lora_sd)
    torch.save(input_adapter.state_dict(), out / "input_adapter.pt")
    return out


def run_training(work_dir, base_model_id=None, hf_token=None, train_config_path=None,
                 max_train_steps=None, num_train_samples=None, device="cuda"):
    cfg = load_train_config(train_config_path or
                            Path(__file__).resolve().parent.parent / "configs" / "inpaint_edit_train.yaml")
    if max_train_steps is not None:
        cfg["max_train_steps"] = int(max_train_steps)
    if num_train_samples is not None:
        cfg["num_train_samples"] = int(num_train_samples)
    base_model_id = base_model_id or cfg["base_model_id"]
    resolution = int(cfg.get("resolution", 512))
    grad_accum = int(cfg.get("grad_accum_steps", 16))
    mask_loss_weight = float(cfg.get("mask_loss_weight", 5.0))

    from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
    from diffusers.training_utils import (compute_density_for_timestep_sampling,
                                          compute_loss_weighting_for_sd3)
    from peft import LoraConfig
    from .inpaint_edit_dataset import load_pairs

    run_dir = _next_run_dir(Path(work_dir) / "models", cfg["model_name"])
    (run_dir / "training_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    write_gpu_info(run_dir); write_pip_freeze(run_dir)

    print(f"[data] streaming {cfg['num_train_samples']} PIPE-{cfg['pipe_split']} person pairs...")
    pairs = load_pairs(num_samples=cfg["num_train_samples"], split=cfg["pipe_split"],
                       person_only=cfg.get("person_only", True),
                       thresh=cfg.get("mask_diff_threshold", 25),
                       dilate_px=cfg.get("mask_dilate_px", 6),
                       min_mask_pixels=cfg.get("min_mask_pixels", 64))
    if not pairs:
        raise RuntimeError("No PIPE person pairs loaded.")
    print(f"[data] {len(pairs)} pairs ready")

    compute_dtype = torch.bfloat16
    _kw = {"torch_dtype": compute_dtype}
    if hf_token and not Path(base_model_id).exists():
        _kw["token"] = hf_token
    pipe = StableDiffusion3Pipeline.from_pretrained(base_model_id, **_kw)

    def to_t(img, mode="RGB"):
        import numpy as np
        a = np.asarray(img.convert(mode).resize((resolution, resolution)), dtype=np.float32) / 255.0
        if mode == "RGB":
            t = torch.from_numpy(a).permute(2, 0, 1) * 2 - 1
        else:
            t = torch.from_numpy(a)[None]
        return t.unsqueeze(0).to(device, dtype=compute_dtype)

    # ---- precompute embeddings to DISK, then drop text encoders ----
    # Caching all embeds in CPU RAM OOMs at scale (each SD3 embed ~2-3MB; 4000 ->
    # ~11GB). Write each UNIQUE prompt's embed to a .pt on disk and load the small
    # per-step tensor lazily in the loop. RAM stays flat regardless of dataset size.
    emb_dir = run_dir / "_emb_cache"; emb_dir.mkdir(exist_ok=True)
    uniq_prompts = list(dict.fromkeys(it["prompt"] for it in pairs))
    prompt_to_id = {p: i for i, p in enumerate(uniq_prompts)}
    pipe.text_encoder.to(device); pipe.text_encoder_2.to(device)
    if pipe.text_encoder_3 is not None:
        pipe.text_encoder_3.to(device)
    with torch.no_grad():
        for p, pid in prompt_to_id.items():
            pe, _, pooled, _ = pipe.encode_prompt(
                p, prompt_2=p, prompt_3=p, device=device,
                num_images_per_prompt=1, do_classifier_free_guidance=False)
            torch.save({"pe": pe.to("cpu"), "pooled": pooled.to("cpu")},
                       emb_dir / f"{pid}.pt")
            del pe, pooled
    pipe.text_encoder = pipe.text_encoder_2 = pipe.text_encoder_3 = None
    import gc; gc.collect(); torch.cuda.empty_cache()
    print(f"  cached {len(uniq_prompts)} unique prompt embeds to disk")

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
    input_adapter = InputAdapter().to(device, dtype=torch.float32)
    opt = torch.optim.AdamW(lora_params + list(input_adapter.parameters()),
                            lr=float(cfg["learning_rate"]))

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
        i = step % len(pairs)
        it = pairs[i]
        target = to_t(it["target"]); source = to_t(it["source"]); mask = to_t(it["mask"], "L")
        with torch.no_grad():
            target_lat = _vae_encode(vae, target)
            source_lat = _vae_encode(vae, source * (1 - mask))
            mask_lat = F.interpolate(mask, size=target_lat.shape[-2:])
            _emb = torch.load(emb_dir / f"{prompt_to_id[it['prompt']]}.pt", weights_only=True)
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
            cond = input_adapter(noisy, source_lat, mask_lat)
            pred = transformer(hidden_states=cond, timestep=timesteps,
                               encoder_hidden_states=prompt_embeds,
                               pooled_projections=pooled, return_dict=False)[0]
        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=cfg.get("weighting_scheme", "logit_normal"), sigmas=sigma)
        target_v = noise - target_lat
        # Mask-weighted loss: the person lives INSIDE the mask, and hard-restore
        # discards everything the model predicts outside it, so weight the in-mask
        # error higher (the model should spend its capacity making the person good,
        # not the about-to-be-discarded background). pix_w = 1 + (W-1)*mask_lat.
        pix_w = 1.0 + (mask_loss_weight - 1.0) * mask_lat.float()
        err = weighting.float() * (pred.float() - target_v.float()) ** 2
        loss = (pix_w * err).sum() / pix_w.sum() / err.shape[1]

        (loss / grad_accum).backward()       # accumulate over grad_accum steps
        lv = loss.item()
        if not math.isfinite(lv):
            mf.close()
            raise RuntimeError(f"Training diverged: non-finite loss at step {step}")
        losses.append(lv)
        mf.write(json.dumps({"step": step, "loss": round(lv, 6)}) + "\n"); mf.flush()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(lora_params + list(input_adapter.parameters()), clip)
            opt.step(); opt.zero_grad()

        if step % 25 == 0:
            print(f"  step {step:4d}/{steps}  loss {lv:.4f}")
        if ckpt_every and (step + 1) % ckpt_every == 0 and (step + 1) < steps:
            _save_artifacts(run_dir, transformer, input_adapter, tag=f"checkpoints/checkpoint-{step+1}")
            print(f"  [ckpt] checkpoint-{step+1}")
    # flush remaining accumulated gradient
    if steps % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(lora_params + list(input_adapter.parameters()), clip)
        opt.step(); opt.zero_grad()
    mf.close()
    # drop the on-disk embed cache (not part of the artifact)
    import shutil as _sh
    _sh.rmtree(emb_dir, ignore_errors=True)

    out = _save_artifacts(run_dir, transformer, input_adapter, tag="adapter")
    adapter_file = out / "pytorch_lora_weights.safetensors"
    provenance = {
        "base_model_id": base_model_id,
        "model_name": cfg["model_name"],
        "flow": "sd35_inpaint_edit_lora",
        "trigger_token": "<vin_ped>",
        "rank": cfg["rank"], "learning_rate": cfg["learning_rate"],
        "max_train_steps": steps, "num_train_samples": len(pairs),
        "grad_accum_steps": grad_accum, "effective_batch": grad_accum,
        "mask_loss_weight": mask_loss_weight,
        "weighting_scheme": cfg.get("weighting_scheme"),
        "lora_sha256": sha256_file(adapter_file) if adapter_file.exists() else "",
        "input_adapter_sha256": sha256_file(out / "input_adapter.pt"),
        "loss_first25_mean": round(sum(losses[:25]) / min(25, len(losses)), 4),
        "loss_last25_mean": round(sum(losses[-25:]) / min(25, len(losses)), 4),
        "requires_input_adapter": True,
    }
    (run_dir / "training_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")

    print(f"\nTRAINING COMPLETE -> {run_dir}")
    print(f"  LoRA:          {adapter_file}")
    print(f"  input_adapter: {out/'input_adapter.pt'}")
    print(f"  loss {provenance['loss_first25_mean']} -> {provenance['loss_last25_mean']}")

    # Release GPU before returning so a same-session eval subprocess isn't starved.
    del transformer, vae, input_adapter, opt, lora_params, pipe
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"run_dir": run_dir, "adapter_dir": out, "provenance": provenance}
