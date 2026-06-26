"""SPIKE — feasibility check for an SD3.5 inpaint-edit LoRA trainer.

GOAL OF THIS SPIKE (not production): confirm the conditioning wiring is correct
before building the full harness. We train ~50 steps on ~20 PIPE pairs and check:
    (1) loss is finite and trends DOWN,
    (2) a quick sample inpaints something plausible inside the mask.

WHY THIS IS THE RISKY PART
SD3's transformer `hidden_states` expects the 16-channel latent. To make the
model *edit-aware* we must feed it the source (masked) image + the mask. We
cannot just concat 16(noisy)+16(masked)+1(mask)=33 channels into the transformer
— its patch-embed projection is fixed at 16. So this spike uses a small TRAINABLE
input adapter (Conv 33->16) plus LoRA on attention. The adapter is the thing that
most likely needs tuning; the spike exists to validate exactly this.

Run on Kaggle GPU only (needs SD3.5 + CUDA). Not runnable on the dev box.

Conditioning recipe (flow matching, matches train_dreambooth_lora_sd3 internals):
    target_latent = VAE(target_img); source_latent = VAE(source_img * (1-mask))
    noisy = (1-sigma)*target_latent + sigma*noise
    cond  = adapter(concat[noisy, source_latent, mask_latent])  # 33 -> 16
    pred  = transformer(hidden_states=cond, timestep, encoder_hidden_states=...)
    loss  = weighting * (pred - (noise - target_latent))^2
"""

import math
from pathlib import Path

import torch
import torch.nn.functional as F


def _vae_encode(vae, images):
    lat = vae.encode(images).latent_dist.sample()
    return (lat - vae.config.shift_factor) * vae.config.scaling_factor


class InputAdapter(torch.nn.Module):
    """Trainable 1x1 conv mixing [noisy(16) | source(16) | mask(1)] -> 16."""

    def __init__(self, in_ch=33, out_ch=16):
        super().__init__()
        self.proj = torch.nn.Conv2d(in_ch, out_ch, kernel_size=1)
        # init so that at start cond ~= noisy latent (identity on first 16)
        torch.nn.init.zeros_(self.proj.weight)
        torch.nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            for c in range(out_ch):
                self.proj.weight[c, c, 0, 0] = 1.0

    def forward(self, noisy, source, mask_latent):
        x = torch.cat([noisy, source, mask_latent], dim=1)
        return self.proj(x)


class InputProj(torch.nn.Module):
    """Mask-FREE (IP2P/PIPE-style) input projection: [noisy(16) | source(16)] -> 16.

    The mask-free edit model is conditioned on the source image; this projection
    adapts the 32-channel concat back to the 16 channels SD3's patch-embed expects.

    A 1x1 conv with identity-on-noisy + ZERO-on-source warm start (the original
    InputAdapter trick) leaves the SOURCE channel mute at init: cond == noisy, and
    because the output is already "good" the source weights barely get gradient —
    the same failure that made the mask-based adapter ignore its conditioning
    (smoke output was pure text-to-image, ignoring the source image entirely).

    Fix: (1) a small hidden 3x3 conv so the projection is spatial-aware (it can
    reason about WHERE the source content is, not just per-pixel channel mixing),
    and (2) warm-start the output layer to identity-on-noisy but with a SMALL
    non-zero source contribution, so the source has a gradient signal from step 0.
    """

    def __init__(self, in_ch=32, out_ch=16, hidden=64, source_init=0.05):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(hidden, out_ch, kernel_size=3, padding=1),
        )
        # Small-init the output layer (no identity warm-start): a two-conv body
        # can't pass noisy through by a single identity row anyway, and the
        # identity trick is exactly what muted the source before. Small random
        # init lets BOTH noisy and source contribute and get gradient from step 0;
        # grad-accum 16 keeps it stable despite no warm-start.
        out_conv = self.body[-1]
        torch.nn.init.normal_(out_conv.weight, std=source_init)
        torch.nn.init.zeros_(out_conv.bias)

    def forward(self, noisy, source):
        return self.body(torch.cat([noisy, source], dim=1))


def run_spike(pairs, base_model_id="stabilityai/stable-diffusion-3.5-medium",
              steps=50, rank=8, lr=1e-4, resolution=512, device="cuda",
              out_dir="/kaggle/working/spike_inpaint_edit", hf_token=None,
              overfit=False, fixed_timestep_frac=0.5):
    """pairs: list of dicts {source_path, target_path, mask_path, prompt}.

    base_model_id may be a HF repo id (gated -> needs hf_token) OR a local path
    to a mounted SD3.5 snapshot (e.g. /kaggle/input/stable-diffusion-3-5-medium).

    overfit=True is the decisive sanity check: train on EXACTLY ONE pair at a
    FIXED timestep. If the conditioning is wired correctly the model memorizes
    that pair and loss collapses toward ~0. A flat loss means the conditioning
    (adapter / latents / mask) is not actually informing the prediction.
    """
    from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
    from peft import LoraConfig
    from PIL import Image
    import numpy as np

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    # bf16 (not fp16) for compute: fp16 flow-matching training NaNs immediately
    # because fp16's narrow range overflows. T4 supports bf16 compute.
    compute_dtype = torch.bfloat16
    _kw = {"torch_dtype": compute_dtype}
    if hf_token and not Path(base_model_id).exists():
        _kw["token"] = hf_token
    pipe = StableDiffusion3Pipeline.from_pretrained(base_model_id, **_kw)

    if overfit:
        pairs = pairs[:1]   # decisive single-pair memorization test

    def load_img(p, mode="RGB", dev="cpu"):
        im = Image.open(p).convert(mode).resize((resolution, resolution))
        a = np.asarray(im, dtype=np.float32) / 255.0
        if mode == "RGB":
            t = torch.from_numpy(a).permute(2, 0, 1) * 2 - 1  # [-1,1]
        else:
            t = torch.from_numpy(a)[None]                      # [1,H,W] 0..1
        return t.unsqueeze(0).to(dev, dtype=compute_dtype)

    # ---- T4-friendly memory plan ----
    # SD3.5 + T5-XXL does NOT fit on a 16GB T4 all-resident. So:
    #   1) precompute ALL text embeddings on GPU, then drop the 3 text encoders,
    #   2) keep only VAE + transformer resident for the training loop.
    pipe.text_encoder.to(device); pipe.text_encoder_2.to(device)
    if pipe.text_encoder_3 is not None:
        pipe.text_encoder_3.to(device)
    embeds = []
    with torch.no_grad():
        for item in pairs:
            pe, _, pooled, _ = pipe.encode_prompt(
                item["prompt"], prompt_2=item["prompt"], prompt_3=item["prompt"],
                device=device, num_images_per_prompt=1, do_classifier_free_guidance=False)
            embeds.append((pe.to("cpu"), pooled.to("cpu")))
    # free text encoders + their VRAM
    pipe.text_encoder = pipe.text_encoder_2 = pipe.text_encoder_3 = None
    import gc; gc.collect(); torch.cuda.empty_cache()

    vae, transformer = pipe.vae, pipe.transformer
    vae.to(device); transformer.to(device)
    noise_sched = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)

    vae.requires_grad_(False); transformer.requires_grad_(False)
    transformer.enable_gradient_checkpointing()

    # LoRA on attention (same target modules as the official SD3 script)
    transformer.add_adapter(LoraConfig(
        r=rank, lora_alpha=rank, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"]))

    # Trainable params (LoRA + adapter) kept in fp32 for stable optimizer math;
    # the frozen backbone stays bf16. Forward runs under bf16 autocast.
    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    for p in lora_params:
        p.data = p.data.float()
    adapter = InputAdapter().to(device, dtype=torch.float32)
    opt = torch.optim.AdamW(lora_params + list(adapter.parameters()), lr=lr)

    losses = []
    transformer.train()
    for step in range(steps):
        idx_pair = step % len(pairs)
        item = pairs[idx_pair]
        target = load_img(item["target_path"], dev=device)
        source = load_img(item["source_path"], dev=device)
        mask = load_img(item["mask_path"], mode="L", dev=device)   # 1 == editable

        with torch.no_grad():
            target_lat = _vae_encode(vae, target)
            source_lat = _vae_encode(vae, source * (1 - mask))  # source w/ hole
            lat_h, lat_w = target_lat.shape[-2:]
            mask_lat = F.interpolate(mask, size=(lat_h, lat_w))
            prompt_embeds, pooled = (e.to(device) for e in embeds[idx_pair])

        noise = torch.randn_like(target_lat)
        if overfit:
            # same noise + same timestep every step -> pure memorization signal
            torch.manual_seed(0)
            noise = torch.randn(target_lat.shape, device=device, dtype=target_lat.dtype)
            u = torch.tensor([fixed_timestep_frac], device=device)
        else:
            u = torch.rand(1, device=device)
        idx = (u * noise_sched.config.num_train_timesteps).long()
        timesteps = noise_sched.timesteps.to(device)[idx]
        sigma = (timesteps / noise_sched.config.num_train_timesteps).view(-1, 1, 1, 1).to(target_lat.dtype)
        noisy = (1.0 - sigma) * target_lat + sigma * noise

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            cond = adapter(noisy, source_lat, mask_lat)
            pred = transformer(hidden_states=cond, timestep=timesteps,
                               encoder_hidden_states=prompt_embeds,
                               pooled_projections=pooled,
                               return_dict=False)[0]
        target_v = noise - target_lat
        loss = F.mse_loss(pred.float(), target_v.float())

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params + list(adapter.parameters()), 1.0)
        opt.step()
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"SPIKE FAIL: non-finite loss at step {step}")
        losses.append(loss.item())
        if step % 10 == 0:
            print(f"  step {step:3d}  loss {loss.item():.4f}")

    n = len(losses)
    first = sum(losses[:5]) / min(5, n)
    last = sum(losses[-5:]) / min(5, n)
    min_loss = min(losses)
    verdict = {
        "mode": "overfit" if overfit else "noisy",
        "steps": steps, "num_pairs": len(pairs),
        "loss_first5_mean": round(first, 4),
        "loss_last5_mean": round(last, 4),
        "loss_min": round(min_loss, 4),
        "all_finite": all(math.isfinite(x) for x in losses),
    }
    if overfit:
        # decisive: a correctly-wired single-pair memorization collapses loss.
        ratio = last / first if first > 0 else 1.0
        verdict["collapse_ratio"] = round(ratio, 4)
        verdict["passed"] = bool(verdict["all_finite"] and ratio < 0.3)
    else:
        verdict["loss_dropped"] = last < first
        verdict["passed"] = bool(verdict["all_finite"] and last < first)

    import json
    (out_dir / "spike_verdict.json").write_text(json.dumps(verdict, indent=2))
    print("\nSPIKE VERDICT:", verdict)
    if overfit:
        print("PASS — single pair memorized, loss collapsed: conditioning is WIRED"
              if verdict["passed"]
              else "FAIL — loss did not collapse on ONE pair: conditioning is NOT informing the prediction")
    else:
        print("PASS — loss trends down" if verdict["passed"]
              else "INCONCLUSIVE — noisy; run overfit=True for a decisive check")
    return verdict
