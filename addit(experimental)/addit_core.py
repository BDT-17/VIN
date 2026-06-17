"""Add-it core mechanisms adapted for SD3.5 MM-DiT.

Implements the three pillars of Add-it (arXiv 2411.07232):

1. **Weighted Extended-Attention** — ``AddItJointAttnProcessor2_0``
   Injects cached source-image K,V into every JointTransformerBlock
   attention layer so the generated image can attend to the original
   scene structure.

2. **Noise Structure Transfer** — ``transfer_noise_structure``
   Encodes the source image through the VAE and adds scheduler-
   appropriate noise to produce a starting latent that already
   carries the scene's spatial layout.

3. **Subject-Guided Latent Blending** — ``subject_guided_blend``
   At each denoising step, blends the generated latent with the
   (re-noised) source latent outside the insertion mask so the
   background is preserved pixel-exactly.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter


def _expand_bbox(bbox, image_size: Tuple[int, int], expansion: float):
    """Expand a bbox around its center and clamp it to image bounds."""
    img_w, img_h = image_size
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = x1 + bw / 2
    cy = y1 + bh / 2
    new_w = bw * max(1.0, float(expansion))
    new_h = bh * max(1.0, float(expansion))
    return (
        int(max(0, round(cx - new_w / 2))),
        int(max(0, round(cy - new_h / 2))),
        int(min(img_w, round(cx + new_w / 2))),
        int(min(img_h, round(cy + new_h / 2))),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Shared mutable state visible to every attention processor
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AddItState:
    """Mutable state shared across all ``AddItJointAttnProcessor2_0`` instances."""

    enabled: bool = False
    """Master switch — when *False* every processor falls back to vanilla attention."""

    # -- Attention-weight schedule values (updated each denoising step) ---
    w_source: float = 0.50
    w_self: float = 0.40

    # -- Source K,V cache (filled during the source forward pass) ---------
    cache_mode: bool = False
    """When *True*, processors store their K,V into ``kv_cache``."""

    inject_mode: bool = False
    """When *True*, processors extend target attention with cached K,V."""

    kv_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    """Per-layer cache: ``{layer_idx: (K, V)}`` in multi-head shape."""

    # -- Layer selection --------------------------------------------------
    layer_start: int = 0
    layer_end: int = 999  # inclusive
    """Only layers ``[layer_start, layer_end]`` participate in injection."""

    def clear_cache(self):
        self.kv_cache.clear()

    def set_weights(self, w_source: float, w_self: float):
        self.w_source = max(0.0, min(1.0, w_source))
        self.w_self = max(0.0, min(1.0, w_self))


# ═══════════════════════════════════════════════════════════════════════════
# 1. Weighted Extended-Attention Processor
# ═══════════════════════════════════════════════════════════════════════════

class AddItJointAttnProcessor2_0:
    """Drop-in replacement for ``JointAttnProcessor2_0`` that supports
    Add-it source-image K,V injection.

    Modes
    -----
    * **Disabled** (``state.enabled == False``): exact same behaviour as the
      stock ``JointAttnProcessor2_0``.
    * **Cache** (``state.cache_mode``): runs normal attention *and* stores
      the image-token K,V into ``state.kv_cache[layer_idx]``.
    * **Inject** (``state.inject_mode``): extends the target's K,V sequence
      with the cached source K,V, weighted by ``state.w_source``.
    """

    def __init__(self, layer_idx: int, state: AddItState, original_processor=None):
        self.layer_idx = layer_idx
        self.state = state
        self.original_processor = original_processor

    # -----------------------------------------------------------------
    # Main entry
    # -----------------------------------------------------------------
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        # Disabled — vanilla SD3.5 joint attention
        if not self.state.enabled:
            return self._standard_attention(
                attn, hidden_states, encoder_hidden_states, attention_mask, *args, **kwargs
            )

        # Cache mode — run normal attention, also save K,V
        if self.state.cache_mode:
            return self._cache_attention(
                attn, hidden_states, encoder_hidden_states, attention_mask, *args, **kwargs
            )

        # Inject mode — extend K,V with cached source features
        if self.state.inject_mode and self._should_inject():
            return self._extended_attention(
                attn, hidden_states, encoder_hidden_states, attention_mask, *args, **kwargs
            )

        # Layer outside injection range → standard attention
        return self._standard_attention(
            attn, hidden_states, encoder_hidden_states, attention_mask, *args, **kwargs
        )

    # -----------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------
    def _should_inject(self) -> bool:
        return (
            self.layer_idx in self.state.kv_cache
            and self.state.layer_start <= self.layer_idx <= self.state.layer_end
        )

    @staticmethod
    def _reshape_heads(tensor: torch.Tensor, heads: int) -> torch.Tensor:
        """[B, N, D] → [B, H, N, D/H]"""
        B, N, D = tensor.shape
        return tensor.view(B, N, heads, D // heads).transpose(1, 2)

    # ----- Standard JointAttnProcessor2_0 behaviour --------------------
    def _standard_attention(self, attn, hidden_states, encoder_hidden_states, attention_mask, *args, **kwargs):
        if self.original_processor is not None:
            return self.original_processor(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                *args,
                **kwargs,
            )

        B = hidden_states.shape[0]
        inner_dim = attn.to_k(hidden_states).shape[-1]
        head_dim = inner_dim // attn.heads

        # Image tokens
        query = self._reshape_heads(attn.to_q(hidden_states), attn.heads)
        key = self._reshape_heads(attn.to_k(hidden_states), attn.heads)
        value = self._reshape_heads(attn.to_v(hidden_states), attn.heads)

        if encoder_hidden_states is not None:
            # Text tokens
            enc_query = self._reshape_heads(attn.add_q_proj(encoder_hidden_states), attn.heads)
            enc_key = self._reshape_heads(attn.add_k_proj(encoder_hidden_states), attn.heads)
            enc_value = self._reshape_heads(attn.add_v_proj(encoder_hidden_states), attn.heads)

            query = torch.cat([enc_query, query], dim=2)
            key = torch.cat([enc_key, key], dim=2)
            value = torch.cat([enc_value, value], dim=2)

        out = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
        out = out.transpose(1, 2).reshape(B, -1, inner_dim)

        if encoder_hidden_states is not None:
            enc_seq = encoder_hidden_states.shape[1]
            enc_out, img_out = out[:, :enc_seq], out[:, enc_seq:]
            img_out = attn.to_out[0](img_out)
            img_out = attn.to_out[1](img_out)
            if getattr(attn, "to_add_out", None) is None:
                return img_out
            enc_out = attn.to_add_out(enc_out)
            return img_out, enc_out

        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out

    # ----- Cache mode: standard attention + store K,V -------------------
    def _cache_attention(self, attn, hidden_states, encoder_hidden_states, attention_mask, *args, **kwargs):
        # Compute image K,V and cache them
        if getattr(attn, "to_k", None) is not None and getattr(attn, "to_v", None) is not None:
            key_img = self._reshape_heads(attn.to_k(hidden_states), attn.heads)
            value_img = self._reshape_heads(attn.to_v(hidden_states), attn.heads)
            self.state.kv_cache[self.layer_idx] = (
                key_img.detach().clone(),
                value_img.detach().clone(),
            )
        # Then run normal attention for the source forward pass
        return self._standard_attention(
            attn, hidden_states, encoder_hidden_states, attention_mask, *args, **kwargs
        )

    # ----- Inject mode: extended attention with cached source K,V -------
    def _extended_attention(self, attn, hidden_states, encoder_hidden_states, attention_mask, *args, **kwargs):
        required = ("to_q", "to_k", "to_v", "to_out")
        if any(getattr(attn, name, None) is None for name in required):
            return self._standard_attention(
                attn, hidden_states, encoder_hidden_states, attention_mask, *args, **kwargs
            )
        if encoder_hidden_states is not None:
            context_required = ("add_q_proj", "add_k_proj", "add_v_proj", "to_add_out")
            if any(getattr(attn, name, None) is None for name in context_required):
                return self._standard_attention(
                    attn, hidden_states, encoder_hidden_states, attention_mask, *args, **kwargs
                )

        B = hidden_states.shape[0]
        inner_dim = attn.to_k(hidden_states).shape[-1]
        head_dim = inner_dim // attn.heads
        w_source = self.state.w_source
        w_self = self.state.w_self
        w_text = max(0.05, 1.0 - w_source - w_self)

        # Target image tokens Q,K,V
        query = self._reshape_heads(attn.to_q(hidden_states), attn.heads)
        key = self._reshape_heads(attn.to_k(hidden_states), attn.heads)
        value = self._reshape_heads(attn.to_v(hidden_states), attn.heads)

        # Cached source image K,V
        src_key, src_value = self.state.kv_cache[self.layer_idx]
        # Expand batch dim if needed (source batch=1 vs target batch=2 for CFG)
        if src_key.shape[0] < B:
            src_key = src_key.expand(B, -1, -1, -1)
            src_value = src_value.expand(B, -1, -1, -1)

        if encoder_hidden_states is not None:
            # Text tokens
            enc_query = self._reshape_heads(attn.add_q_proj(encoder_hidden_states), attn.heads)
            enc_key = self._reshape_heads(attn.add_k_proj(encoder_hidden_states), attn.heads)
            enc_value = self._reshape_heads(attn.add_v_proj(encoder_hidden_states), attn.heads)

            # Build extended Q, K, V. Add-it weights the keys from each
            # information source so attention balances prompt, target, source.
            # Q = [text_target, image_target]  (unchanged)
            # K = [text_target * w_text, image_target * w_self, image_source * w_source]
            # V = [text_target, image_target, image_source]
            full_query = torch.cat([enc_query, query], dim=2)
            full_key = torch.cat([enc_key * w_text, key * w_self, src_key * w_source], dim=2)
            full_value = torch.cat([enc_value, value, src_value], dim=2)
        else:
            full_query = query
            full_key = torch.cat([key * w_self, src_key * w_source], dim=2)
            full_value = torch.cat([value, src_value], dim=2)

        out = F.scaled_dot_product_attention(full_query, full_key, full_value, attn_mask=None)
        out = out.transpose(1, 2).reshape(B, -1, inner_dim)

        if encoder_hidden_states is not None:
            enc_seq = encoder_hidden_states.shape[1]
            enc_out, img_out = out[:, :enc_seq], out[:, enc_seq:]
            img_out = attn.to_out[0](img_out)
            img_out = attn.to_out[1](img_out)
            enc_out = attn.to_add_out(enc_out)
            return img_out, enc_out

        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


# ═══════════════════════════════════════════════════════════════════════════
# Processor injection / restoration helpers
# ═══════════════════════════════════════════════════════════════════════════

def inject_addit_processors(transformer, state: AddItState):
    """Replace every attention processor in *transformer* with an
    ``AddItJointAttnProcessor2_0`` sharing *state*.

    Returns
    -------
    original_processors : dict
        The mapping ``{name: processor}`` that was active before injection.
        Pass it to :func:`restore_processors` to undo.
    """
    original = {}
    new_processors = {}
    for idx, (name, module) in enumerate(transformer.attn_processors.items()):
        original[name] = module
        new_processors[name] = AddItJointAttnProcessor2_0(
            layer_idx=idx,
            state=state,
            original_processor=module,
        )
    transformer.set_attn_processor(new_processors)
    return original


def restore_processors(transformer, original_processors: dict):
    """Restore the attention processors saved by :func:`inject_addit_processors`."""
    transformer.set_attn_processor(original_processors)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Noise Structure Transfer
# ═══════════════════════════════════════════════════════════════════════════

def encode_image_to_latent(vae, image: Image.Image, resolution: int,
                           device, dtype) -> torch.Tensor:
    """Encode a PIL Image to VAE latent space.

    Returns
    -------
    latent : Tensor  [1, C, H/8, W/8]
    """
    image = image.convert("RGB").resize((resolution, resolution), Image.LANCZOS)
    pixel = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1)
    pixel = (pixel / 127.5 - 1.0).unsqueeze(0).to(device=device, dtype=dtype)
    with torch.no_grad():
        latent_dist = vae.encode(pixel).latent_dist
        latent = latent_dist.sample()
    # Apply SD3.5 scaling
    if hasattr(vae.config, "scaling_factor"):
        sf = vae.config.scaling_factor
        shift = getattr(vae.config, "shift_factor", 0.0)
        latent = (latent - shift) * sf
    return latent


def decode_latent_to_image(vae, latent: torch.Tensor) -> Image.Image:
    """Decode a VAE latent back to a PIL Image."""
    with torch.no_grad():
        if hasattr(vae.config, "scaling_factor"):
            sf = vae.config.scaling_factor
            shift = getattr(vae.config, "shift_factor", 0.0)
            latent = latent / sf + shift
        decoded = vae.decode(latent).sample
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    decoded = decoded[0].permute(1, 2, 0).cpu().float().numpy()
    decoded = (decoded * 255).astype(np.uint8)
    return Image.fromarray(decoded, mode="RGB")


def scheduler_add_noise(scheduler, sample: torch.Tensor, noise: torch.Tensor, timestep):
    """Add noise using the scheduler API available in the installed diffusers.

    SD3.5 commonly uses FlowMatchEulerDiscreteScheduler, where recent
    diffusers versions expose ``scale_noise(sample, timestep, noise)``.
    Other schedulers expose ``add_noise(sample, noise, timestep)``.
    """
    if hasattr(scheduler, "add_noise"):
        return scheduler.add_noise(sample, noise, timestep)
    if hasattr(scheduler, "scale_noise"):
        return scheduler.scale_noise(sample, timestep, noise)

    raise AttributeError(
        f"{type(scheduler).__name__} has neither add_noise nor scale_noise."
    )


def transfer_noise_structure(
    vae,
    scheduler,
    source_image: Image.Image,
    strength: float,
    noise_blend_ratio: float,
    num_inference_steps: int,
    generator: torch.Generator,
    resolution: int,
    device,
    dtype,
    source_latent: Optional[torch.Tensor] = None,
):
    """Implement Noise Structure Transfer.

    Encode *source_image* to latent, add flow-matching noise at the
    timestep corresponding to *strength*, and return the noised latent
    together with the clean source latent and the noise tensor.

    Parameters
    ----------
    strength : float
        0 = pure random noise (no structure), 1 = clean source (no denoising).
    noise_blend_ratio : float
        Fraction of source-correlated noise vs. pure random.

    Returns
    -------
    initial_latent, source_latent, noise, timesteps
    """
    if source_latent is None:
        source_latent = encode_image_to_latent(vae, source_image, resolution, device, dtype)
    else:
        source_latent = source_latent.to(device=device, dtype=dtype)

    # Set up scheduler timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    all_timesteps = scheduler.timesteps

    # Compute starting point based on strength
    init_step = min(int(num_inference_steps * strength), num_inference_steps)
    t_start_idx = max(0, len(all_timesteps) - init_step)
    timesteps = all_timesteps[t_start_idx:]

    if len(timesteps) == 0:
        # strength ≈ 0 → no denoising needed, return source as-is
        return source_latent.clone(), source_latent, torch.zeros_like(source_latent), all_timesteps[:0]

    # Generate noise — blend source-correlated noise with pure random
    noise_random = torch.randn(
        source_latent.shape, generator=generator,
        device=device, dtype=dtype,
    )
    if noise_blend_ratio < 1.0:
        noise_structural = torch.randn_like(source_latent)
        noise = noise_blend_ratio * noise_random + (1 - noise_blend_ratio) * noise_structural
        noise = noise / noise.std() * noise_random.std()  # normalise
    else:
        noise = noise_random

    # Add noise at the first active timestep
    initial_latent = scheduler_add_noise(scheduler, source_latent, noise, timesteps[:1])

    return initial_latent, source_latent, noise, timesteps


def get_source_noised_at_step(scheduler, source_latent, noise, timestep):
    """Compute the noised source latent at a given scheduler *timestep*.

    Used for latent blending — the background region should match what
    the source would look like at the current noise level.
    """
    return scheduler_add_noise(scheduler, source_latent, noise, timestep.unsqueeze(0))


# ═══════════════════════════════════════════════════════════════════════════
# 3. Subject-Guided Latent Blending
# ═══════════════════════════════════════════════════════════════════════════

def create_latent_insertion_mask(
    bbox,
    image_size: Tuple[int, int],
    latent_size: Tuple[int, int],
    feather: int = 3,
    dilation: int = 2,
    expansion: float = 1.15,
) -> torch.Tensor:
    """Create a soft mask in latent space representing a person silhouette.

    Parameters
    ----------
    bbox : tuple (x1, y1, x2, y2) in pixel coordinates
    image_size : (W, H) of the full image
    latent_size : (W, H) of the latent tensor (typically image/8)
    feather : Gaussian blur radius in latent pixels
    dilation : extra expansion in latent pixels
    expansion : multiplicative bbox expansion ratio

    Returns
    -------
    mask : Tensor [1, 1, latent_H, latent_W] in [0, 1]
    """
    img_w, img_h = image_size
    lat_w, lat_h = latent_size

    # Draw person silhouette at full resolution
    mask_full = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask_full)

    x1, y1, x2, y2 = _expand_bbox(bbox, image_size, expansion)
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw // 2

    # Define person shape proportions
    head_r = max(4, int(bw * 0.20))
    head_y = y1 + max(4, int(bh * 0.10))
    shoulder_y = y1 + int(bh * 0.26)
    hip_y = y1 + int(bh * 0.60)
    foot_y = y2

    # Draw Head
    draw.ellipse((cx - head_r, head_y, cx + head_r, head_y + 2 * head_r), fill=255)
    # Draw Torso
    draw.rounded_rectangle((cx - int(bw * 0.25), shoulder_y, cx + int(bw * 0.25), hip_y), radius=4, fill=255)
    # Draw Legs
    leg_w = max(4, int(bw * 0.12))
    draw.line((cx - int(bw * 0.12), hip_y, cx - int(bw * 0.18), foot_y), fill=255, width=leg_w)
    draw.line((cx + int(bw * 0.12), hip_y, cx + int(bw * 0.18), foot_y), fill=255, width=leg_w)
    # Draw Arms
    arm_w = max(3, int(bw * 0.08))
    draw.line((cx - int(bw * 0.22), shoulder_y + 6, cx - int(bw * 0.32), hip_y - 4), fill=255, width=arm_w)
    draw.line((cx + int(bw * 0.22), shoulder_y + 6, cx + int(bw * 0.32), hip_y - 4), fill=255, width=arm_w)

    # Resize to latent space
    mask_pil = mask_full.resize((lat_w, lat_h), Image.Resampling.BILINEAR)

    if dilation > 0:
        mask_pil = mask_pil.filter(ImageFilter.MaxFilter(size=1 + 2 * dilation))

    if feather > 0:
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=feather))

    mask_np = np.array(mask_pil, dtype=np.float32) / 255.0
    return torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]


def create_pixel_insertion_mask(
    bbox,
    image_size: Tuple[int, int],
    expansion: float = 1.15,
    feather: int = 0,
    mode: str = "bbox",
) -> Image.Image:
    """Create a pixel-space mask for final compositing.

    White keeps generated pixels; black restores source pixels. With
    ``mode="bbox"`` and ``feather=0``, every pixel outside the expanded
    insertion bbox is copied exactly from the source image.
    """
    img_w, img_h = image_size
    x1, y1, x2, y2 = _expand_bbox(bbox, image_size, expansion)
    mask = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)

    if mode == "silhouette":
        bw = x2 - x1
        bh = y2 - y1
        cx = x1 + bw // 2
        head_r = max(4, int(bw * 0.20))
        head_y = y1 + max(4, int(bh * 0.10))
        shoulder_y = y1 + int(bh * 0.26)
        hip_y = y1 + int(bh * 0.60)
        foot_y = y2

        draw.ellipse((cx - head_r, head_y, cx + head_r, head_y + 2 * head_r), fill=255)
        draw.rounded_rectangle((cx - int(bw * 0.25), shoulder_y, cx + int(bw * 0.25), hip_y), radius=4, fill=255)
        leg_w = max(4, int(bw * 0.12))
        draw.line((cx - int(bw * 0.12), hip_y, cx - int(bw * 0.18), foot_y), fill=255, width=leg_w)
        draw.line((cx + int(bw * 0.12), hip_y, cx + int(bw * 0.18), foot_y), fill=255, width=leg_w)
        arm_w = max(3, int(bw * 0.08))
        draw.line((cx - int(bw * 0.22), shoulder_y + 6, cx - int(bw * 0.32), hip_y - 4), fill=255, width=arm_w)
        draw.line((cx + int(bw * 0.22), shoulder_y + 6, cx + int(bw * 0.32), hip_y - 4), fill=255, width=arm_w)
    else:
        draw.rectangle((x1, y1, x2, y2), fill=255)

    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
    return mask


def composite_generated_region(
    source_image: Image.Image,
    generated_image: Image.Image,
    bbox,
    expansion: float = 1.15,
    feather: int = 0,
    mode: str = "bbox",
) -> Image.Image:
    """Restore source pixels outside the generated insertion region."""
    source = source_image.convert("RGB")
    generated = generated_image.convert("RGB").resize(source.size, Image.LANCZOS)
    mask = create_pixel_insertion_mask(
        bbox=bbox,
        image_size=source.size,
        expansion=expansion,
        feather=feather,
        mode=mode,
    )
    return Image.composite(generated, source, mask)


def subject_guided_blend(
    generated_latent: torch.Tensor,
    source_noised_at_t: torch.Tensor,
    latent_mask: torch.Tensor,
    step_ratio: float,
    blend_end_ratio: float = 0.80,
) -> torch.Tensor:
    """Blend generated latent with source-noised latent.

    Inside the mask (insertion region): keep the generated latent.
    Outside the mask (background): replace with source-noised latent.

    Blending is only active when ``step_ratio < blend_end_ratio``.
    Keep ``blend_end_ratio`` at 1.0 to preserve the background through
    the final denoising step.

    Parameters
    ----------
    generated_latent : [B, C, H, W]
    source_noised_at_t : [B, C, H, W]
    latent_mask : [1, 1, H, W] in [0, 1] — 1 = insertion region
    step_ratio : float in [0, 1] — current progress (0=start, 1=end)
    blend_end_ratio : float — stop blending after this ratio

    Returns
    -------
    blended : [B, C, H, W]
    """
    if step_ratio >= blend_end_ratio:
        return generated_latent

    mask = latent_mask.to(device=generated_latent.device, dtype=generated_latent.dtype)

    # mask=1 -> keep generated; mask=0 -> keep source-noised background.
    blended = mask * generated_latent + (1 - mask) * source_noised_at_t
    return blended


# ═══════════════════════════════════════════════════════════════════════════
# Attention Weight Scheduling
# ═══════════════════════════════════════════════════════════════════════════

def compute_attention_weights(
    step: int,
    total_steps: int,
    schedule: str = "cosine",
    w_source_range: Tuple[float, float] = (0.70, 0.05),
    w_self_range: Tuple[float, float] = (0.20, 0.85),
) -> Tuple[float, float]:
    """Compute timestep-dependent attention weights.

    Returns ``(w_source, w_self)``; the implicit text weight is
    ``max(0.05, 1 - w_source - w_self)``.

    Schedules
    ---------
    * ``cosine``: smooth cosine decay for w_source, cosine rise for w_self.
    * ``linear``: linear interpolation.
    * ``constant``: keep start values throughout.
    """
    if total_steps <= 1:
        return w_source_range[0], w_self_range[0]

    t = step / max(1, total_steps - 1)  # 0 → 1

    if schedule == "cosine":
        factor = 0.5 * (1 - math.cos(math.pi * t))
    elif schedule == "linear":
        factor = t
    elif schedule == "constant":
        factor = 0.0
    else:
        factor = t  # fallback to linear

    w_src = w_source_range[0] + (w_source_range[1] - w_source_range[0]) * factor
    w_self = w_self_range[0] + (w_self_range[1] - w_self_range[0]) * factor
    return float(w_src), float(w_self)
