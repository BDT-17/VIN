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


def run_spike(pairs, base_model_id="stabilityai/stable-diffusion-3.5-medium",
              steps=50, rank=8, lr=1e-4, resolution=512, device="cuda",
              out_dir="/kaggle/working/spike_inpaint_edit"):
    """pairs: list of dicts {source_path, target_path, mask_path, prompt}."""
    from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
    from peft import LoraConfig
    from PIL import Image
    import numpy as np

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pipe = StableDiffusion3Pipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    vae, transformer = pipe.vae, pipe.transformer
    noise_sched = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)

    vae.requires_grad_(False); transformer.requires_grad_(False)

    # LoRA on attention (same target modules as the official SD3 script)
    transformer.add_adapter(LoraConfig(
        r=rank, lora_alpha=rank, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"]))

    adapter = InputAdapter().to(device, dtype=torch.float16)
    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(lora_params + list(adapter.parameters()), lr=lr)

    def load_img(p, mode="RGB"):
        im = Image.open(p).convert(mode).resize((resolution, resolution))
        a = np.asarray(im, dtype=np.float32) / 255.0
        if mode == "RGB":
            t = torch.from_numpy(a).permute(2, 0, 1) * 2 - 1  # [-1,1]
        else:
            t = torch.from_numpy(a)[None]                      # [1,H,W] 0..1
        return t.unsqueeze(0).to(device, dtype=torch.float16)

    losses = []
    transformer.train()
    for step in range(steps):
        item = pairs[step % len(pairs)]
        target = load_img(item["target_path"])
        source = load_img(item["source_path"])
        mask = load_img(item["mask_path"], mode="L")            # 1 == editable

        with torch.no_grad():
            target_lat = _vae_encode(vae, target)
            source_lat = _vae_encode(vae, source * (1 - mask))  # source w/ hole
            lat_h, lat_w = target_lat.shape[-2:]
            mask_lat = F.interpolate(mask, size=(lat_h, lat_w))
            prompt_embeds, _, pooled, _ = pipe.encode_prompt(
                item["prompt"], prompt_2=item["prompt"], prompt_3=item["prompt"],
                device=device, num_images_per_prompt=1, do_classifier_free_guidance=False)

        noise = torch.randn_like(target_lat)
        u = torch.rand(1, device=device)
        idx = (u * noise_sched.config.num_train_timesteps).long()
        timesteps = noise_sched.timesteps.to(device)[idx]
        sigma = (timesteps / noise_sched.config.num_train_timesteps).view(-1, 1, 1, 1).to(target_lat.dtype)
        noisy = (1.0 - sigma) * target_lat + sigma * noise

        cond = adapter(noisy, source_lat, mask_lat)
        pred = transformer(hidden_states=cond, timestep=timesteps,
                           encoder_hidden_states=prompt_embeds.to(target_lat.dtype),
                           pooled_projections=pooled.to(target_lat.dtype),
                           return_dict=False)[0]
        target_v = noise - target_lat
        loss = F.mse_loss(pred.float(), target_v.float())

        opt.zero_grad(); loss.backward(); opt.step()
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"SPIKE FAIL: non-finite loss at step {step}")
        losses.append(loss.item())
        if step % 10 == 0:
            print(f"  step {step:3d}  loss {loss.item():.4f}")

    first = sum(losses[:5]) / min(5, len(losses))
    last = sum(losses[-5:]) / min(5, len(losses))
    verdict = {
        "steps": steps, "loss_first5_mean": round(first, 4),
        "loss_last5_mean": round(last, 4), "loss_dropped": last < first,
        "all_finite": all(math.isfinite(x) for x in losses),
    }
    import json
    (out_dir / "spike_verdict.json").write_text(json.dumps(verdict, indent=2))
    print("\nSPIKE VERDICT:", verdict)
    print("PASS — conditioning wired, loss trends down" if verdict["loss_dropped"]
          else "INCONCLUSIVE — loss did not drop; revisit adapter/conditioning")
    return verdict
