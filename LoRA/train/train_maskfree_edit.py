"""Mask-FREE SD3.5 edit trainer (IP2P/PIPE-style).

PIVOT 2026-06-26: the mask-based inpaint-edit adapter produced people that did
not fit the mask. Per the PIPE paper (Wasserman et al., 2404.18212), object
addition is learned mask-FREE: condition on the SOURCE IMAGE + instruction only,
and let the model decide where/scale/pose. There is no mask input and no
hard-restore — the model is responsible for the whole image.

Conditioning (flow matching, no mask):
    target_lat = VAE(target)         # the real photo WITH the person
    source_lat = VAE(source)         # the object-erased background  (NOT masked)
    noisy = (1-sigma)*target_lat + sigma*noise
    cond  = input_proj(concat[noisy, source_lat])   # 32 -> 16
    pred  = transformer(hidden_states=cond, timestep, encoder_hidden_states=...)
    loss  = weighting(sigma) * (pred - (noise - target_lat))^2

Classifier-free guidance is trained in (paper §4): with p=0.05 each, drop the
text (empty prompt embed), drop the image (zero source_lat), or drop both. At
inference the runner combines the three scores with separate text/image scales.

Paper hyperparams adapted to a single T4: effective batch via grad-accumulation
(paper used 4096 across 8×A100; we lift batch=1 to ~16 with accumulation),
lr 5e-5, CFG dropout 5%. TWO artifacts are exported and BOTH are required at
inference:
    adapter/pytorch_lora_weights.safetensors   (LoRA on attention)
    adapter/input_proj.pt                       (the Conv 32->16, mask-free)

Memory plan mirrors the mask-based trainer: precompute text embeds to disk, drop
the 3 text encoders, keep VAE + transformer resident (fits a 16GB T4). bf16
compute, fp32 trainable params. Kaggle-GPU only.
"""

import json
import math
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from ..data.config import load_train_config
from .spike_inpaint_edit import InputProj, _vae_encode
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


def _save_artifacts(run_dir, transformer, input_proj, tag="adapter"):
    """Save LoRA safetensors + the mask-free input projection. tag='adapter' for
    final, or 'checkpoints/checkpoint-N' for intermediate."""
    from diffusers import StableDiffusion3Pipeline
    from peft.utils import get_peft_model_state_dict
    out = run_dir / tag
    out.mkdir(parents=True, exist_ok=True)
    lora_sd = get_peft_model_state_dict(transformer)
    StableDiffusion3Pipeline.save_lora_weights(save_directory=str(out),
                                               transformer_lora_layers=lora_sd)
    torch.save(input_proj.state_dict(), out / "input_proj.pt")
    return out


def run_training(work_dir, base_model_id=None, hf_token=None, train_config_path=None,
                 max_train_steps=None, num_train_samples=None, device="cuda"):
    cfg = load_train_config(train_config_path or
                            Path(__file__).resolve().parent.parent / "configs" / "maskfree_edit_train.yaml")
    if max_train_steps is not None:
        cfg["max_train_steps"] = int(max_train_steps)
    if num_train_samples is not None:
        cfg["num_train_samples"] = int(num_train_samples)
    base_model_id = base_model_id or cfg["base_model_id"]
    resolution = int(cfg.get("resolution", 512))
    grad_accum = int(cfg.get("grad_accum_steps", 16))
    cfg_drop = float(cfg.get("cfg_dropout_prob", 0.05))

    from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
    from diffusers.training_utils import (compute_density_for_timestep_sampling,
                                          compute_loss_weighting_for_sd3)
    from peft import LoraConfig
    from .maskfree_edit_dataset import load_pairs

    run_dir = _next_run_dir(Path(work_dir) / "models", cfg["model_name"])
    (run_dir / "training_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    write_gpu_info(run_dir); write_pip_freeze(run_dir)

    print(f"[data] streaming {cfg['num_train_samples']} PIPE-{cfg['pipe_split']} person pairs (mask-free)...")
    pairs = load_pairs(num_samples=cfg["num_train_samples"], split=cfg["pipe_split"],
                       person_only=cfg.get("person_only", True),
                       diff_thresh=cfg.get("diff_thresh", 25),
                       min_change_pixels=cfg.get("min_change_pixels", 64))
    if not pairs:
        raise RuntimeError("No PIPE person pairs loaded.")
    print(f"[data] {len(pairs)} pairs ready")

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
    # Cache each UNIQUE prompt's embed to a .pt and load lazily in the loop.
    # Also cache the EMPTY-prompt embed (id -1) for CFG text-dropout.
    emb_dir = run_dir / "_emb_cache"; emb_dir.mkdir(exist_ok=True)
    uniq_prompts = list(dict.fromkeys(it["prompt"] for it in pairs))
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
        _encode_to("empty", "")          # CFG text-dropout embed
    pipe.text_encoder = pipe.text_encoder_2 = pipe.text_encoder_3 = None
    import gc; gc.collect(); torch.cuda.empty_cache()
    print(f"  cached {len(uniq_prompts)} unique prompt embeds (+empty) to disk")
    _empty = torch.load(emb_dir / "empty.pt", weights_only=True)

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
    input_proj = InputProj().to(device, dtype=torch.float32)
    opt = torch.optim.AdamW(lora_params + list(input_proj.parameters()),
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

    # deterministic per-step CFG-dropout decisions (no Math.random reliance)
    rng = torch.Generator(device="cpu").manual_seed(int(cfg.get("seed", 42)))

    losses = []
    transformer.train()
    opt.zero_grad()
    for step in range(steps):
        i = step % len(pairs)
        it = pairs[i]
        target = to_t(it["target"]); source = to_t(it["source"])
        with torch.no_grad():
            target_lat = _vae_encode(vae, target)
            source_lat = _vae_encode(vae, source)            # NO mask — full source
            # CFG dropout (paper §4): independently drop text / image at p each.
            drop_text = torch.rand(1, generator=rng).item() < cfg_drop
            drop_image = torch.rand(1, generator=rng).item() < cfg_drop
            if drop_text:
                prompt_embeds, pooled = _empty["pe"].to(device), _empty["pooled"].to(device)
            else:
                _emb = torch.load(emb_dir / f"{prompt_to_id[it['prompt']]}.pt", weights_only=True)
                prompt_embeds, pooled = _emb["pe"].to(device), _emb["pooled"].to(device)
            if drop_image:
                source_lat = torch.zeros_like(source_lat)

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
            cond = input_proj(noisy, source_lat)
            pred = transformer(hidden_states=cond, timestep=timesteps,
                               encoder_hidden_states=prompt_embeds,
                               pooled_projections=pooled, return_dict=False)[0]
        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=cfg.get("weighting_scheme", "logit_normal"), sigmas=sigma)
        target_v = noise - target_lat
        loss = (weighting.float() * (pred.float() - target_v.float()) ** 2).mean()

        (loss / grad_accum).backward()       # accumulate
        lv = loss.item()
        if not math.isfinite(lv):
            mf.close()
            raise RuntimeError(f"Training diverged: non-finite loss at step {step}")
        losses.append(lv)
        mf.write(json.dumps({"step": step, "loss": round(lv, 6)}) + "\n"); mf.flush()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(lora_params + list(input_proj.parameters()), clip)
            opt.step(); opt.zero_grad()

        if step % 25 == 0:
            print(f"  step {step:4d}/{steps}  loss {lv:.4f}")
        if ckpt_every and (step + 1) % ckpt_every == 0 and (step + 1) < steps:
            _save_artifacts(run_dir, transformer, input_proj, tag=f"checkpoints/checkpoint-{step+1}")
            print(f"  [ckpt] checkpoint-{step+1}")

    # flush any remaining accumulated grad
    if steps % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(lora_params + list(input_proj.parameters()), clip)
        opt.step(); opt.zero_grad()
    mf.close()
    import shutil as _sh
    _sh.rmtree(emb_dir, ignore_errors=True)

    out = _save_artifacts(run_dir, transformer, input_proj, tag="adapter")
    adapter_file = out / "pytorch_lora_weights.safetensors"
    provenance = {
        "base_model_id": base_model_id,
        "model_name": cfg["model_name"],
        "flow": "sd35_maskfree_edit_lora",
        "conditioning": "ip2p_maskfree",
        "rank": cfg["rank"], "learning_rate": cfg["learning_rate"],
        "max_train_steps": steps, "num_train_samples": len(pairs),
        "grad_accum_steps": grad_accum, "effective_batch": grad_accum,
        "cfg_dropout_prob": cfg_drop,
        "weighting_scheme": cfg.get("weighting_scheme"),
        "lora_sha256": sha256_file(adapter_file) if adapter_file.exists() else "",
        "input_proj_sha256": sha256_file(out / "input_proj.pt"),
        "loss_first25_mean": round(sum(losses[:25]) / min(25, len(losses)), 4),
        "loss_last25_mean": round(sum(losses[-25:]) / min(25, len(losses)), 4),
        "requires_input_proj": True,
        "mask_free": True,
    }
    (run_dir / "training_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")

    print(f"\nTRAINING COMPLETE -> {run_dir}")
    print(f"  LoRA:        {adapter_file}")
    print(f"  input_proj:  {out/'input_proj.pt'}")
    print(f"  loss {provenance['loss_first25_mean']} -> {provenance['loss_last25_mean']}")
    return {"run_dir": run_dir, "adapter_dir": out, "provenance": provenance}
