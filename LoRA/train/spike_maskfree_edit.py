"""SPIKE — decisive feasibility check for the mask-FREE edit conditioning.

Question this answers (cheaply, ~300 steps on ONE pair): can the mask-free
InputProj actually carry the source image into the SD3.5 transformer, or is the
32->16 channel squeeze too lossy to condition on the source?

Method (overfit one pair, fixed timestep):
    target_lat = VAE(target);  source_lat = VAE(source)
    noisy = (1-sigma)*target_lat + sigma*noise        (fixed sigma, fixed noise)
    cond  = input_proj([noisy | source_lat])          (32 -> 16)
    pred  = transformer(cond, t, text)
    loss  = (pred - (noise - target_lat))^2
Train LoRA + InputProj on this ONE pair. If conditioning works, loss collapses
(the model memorizes how to turn THIS source into THIS target). A flat loss means
the source signal is not reaching the prediction — the squeeze is too lossy and
we must change how source enters (e.g. extra tokens instead of channel concat).

PASS if collapse_ratio (last5/first5) < 0.3, like the mask-based spike.

Kaggle-GPU only.
"""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .spike_inpaint_edit import InputProj, _vae_encode


def run_maskfree_spike(base_model_id="stabilityai/stable-diffusion-3.5-medium",
                       split="train", steps=300, rank=16, lr=5e-4,
                       resolution=512, device="cuda",
                       out_dir="/kaggle/working/spike_maskfree_edit",
                       hf_token=None, fixed_timestep_frac=0.5):
    """Overfit ONE PIPE person pair to test the mask-free conditioning.

    lr is intentionally high (5e-4) for fast single-pair memorization.
    """
    from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
    from peft import LoraConfig
    from .maskfree_edit_dataset import load_pairs

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    compute_dtype = torch.bfloat16

    pairs = load_pairs(num_samples=1, split=split, person_only=True)
    if not pairs:
        raise RuntimeError("No PIPE person pair loaded for the spike.")
    item = pairs[0]
    print(f"[spike] pair prompt: {item['prompt'][:80]!r}")

    _kw = {"torch_dtype": compute_dtype}
    if hf_token and not Path(base_model_id).exists():
        _kw["token"] = hf_token
    pipe = StableDiffusion3Pipeline.from_pretrained(base_model_id, **_kw)

    def to_t(img):
        import numpy as np
        a = np.asarray(img.convert("RGB").resize((resolution, resolution)), dtype=np.float32) / 255.0
        t = torch.from_numpy(a).permute(2, 0, 1) * 2 - 1
        return t.unsqueeze(0).to(device, dtype=compute_dtype)

    # encode the one prompt, then drop text encoders
    pipe.text_encoder.to(device); pipe.text_encoder_2.to(device)
    if pipe.text_encoder_3 is not None:
        pipe.text_encoder_3.to(device)
    with torch.no_grad():
        pe, _, pooled, _ = pipe.encode_prompt(
            item["prompt"], prompt_2=item["prompt"], prompt_3=item["prompt"],
            device=device, num_images_per_prompt=1, do_classifier_free_guidance=False)
        prompt_embeds, pooled = pe.to(device), pooled.to(device)
    pipe.text_encoder = pipe.text_encoder_2 = pipe.text_encoder_3 = None
    import gc; gc.collect(); torch.cuda.empty_cache()

    vae, transformer = pipe.vae, pipe.transformer
    vae.to(device); transformer.to(device)
    noise_sched = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    vae.requires_grad_(False); transformer.requires_grad_(False)
    transformer.enable_gradient_checkpointing()
    transformer.add_adapter(LoraConfig(
        r=rank, lora_alpha=rank, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    for p in lora_params:
        p.data = p.data.float()
    input_proj = InputProj().to(device, dtype=torch.float32)
    opt = torch.optim.AdamW(lora_params + list(input_proj.parameters()), lr=lr)

    with torch.no_grad():
        target_lat = _vae_encode(vae, to_t(item["target"]))
        source_lat = _vae_encode(vae, to_t(item["source"]))
        # fixed noise + fixed timestep -> pure memorization signal
        torch.manual_seed(0)
        noise = torch.randn(target_lat.shape, device=device, dtype=target_lat.dtype)
    num_ts = noise_sched.config.num_train_timesteps
    timesteps = noise_sched.timesteps.to(device)[int(fixed_timestep_frac * num_ts)].view(1)
    sigma = (timesteps.float() / num_ts).view(-1, 1, 1, 1).to(target_lat.dtype)
    noisy = (1.0 - sigma) * target_lat + sigma * noise
    target_v = noise - target_lat

    losses = []
    transformer.train()
    for step in range(steps):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            cond = input_proj(noisy, source_lat)
            pred = transformer(hidden_states=cond, timestep=timesteps,
                               encoder_hidden_states=prompt_embeds,
                               pooled_projections=pooled, return_dict=False)[0]
        loss = F.mse_loss(pred.float(), target_v.float())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params + list(input_proj.parameters()), 1.0)
        opt.step()
        lv = loss.item()
        if not math.isfinite(lv):
            raise RuntimeError(f"SPIKE FAIL: non-finite loss at step {step}")
        losses.append(lv)
        if step % 25 == 0:
            print(f"  step {step:3d}  loss {lv:.4f}")

    n = len(losses)
    first = sum(losses[:5]) / min(5, n)
    last = sum(losses[-5:]) / min(5, n)
    ratio = last / first if first > 0 else 1.0
    verdict = {
        "flow": "maskfree", "steps": steps,
        "loss_first5_mean": round(first, 4),
        "loss_last5_mean": round(last, 4),
        "loss_min": round(min(losses), 4),
        "collapse_ratio": round(ratio, 4),
        "passed": bool(all(math.isfinite(x) for x in losses) and ratio < 0.3),
    }
    (out_dir / "spike_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print("\nSPIKE VERDICT:", verdict)
    print("PASS — single pair memorized: mask-free source conditioning is WIRED, full train is worth it"
          if verdict["passed"] else
          "FAIL — loss did not collapse: the source signal is NOT reaching the prediction; "
          "the 32->16 squeeze is too lossy -> change how source enters (extra tokens, not channel concat)")
    return verdict
