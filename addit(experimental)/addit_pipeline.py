"""Add-it end-to-end pipeline — faithful re-implementation on SD3.5 Medium.

Implements the full Add-it loop of Tewel et al. (ICLR 2025, arXiv:2411.07232):

    1. Encode source → latent; structure transfer (§3.3): start latent =
       source noised to a fixed high t_struct with a single random ε.
    2. Denoise SOURCE and TARGET in parallel (§3.2/§3.5).  For a real source we
       do not invert; each step re-noises the source latent
       X^t_source = (1−σ_t)·X_source + σ_t·ε, which reconstructs it exactly.
    3. Each step the TARGET pulls cached source K,V into its joint attention,
       with eq.(3) key weights balanced by an auto-γ root-solver (§3.2).
    4. Capture the subject-token → image-patch attention over a mid window
       (§3.4), Otsu-threshold it, sample ≤4 SAM-2 point prompts, refine to a
       mask, and at t_blend composite  Z = M·Z_target + (1−M)·Z_source.
    5. Decode; optionally enforce pixel-exact source outside M.

There is **no input bounding box, no placement heuristic, and no detector
cutout/paste** — Add-it decides *where* to add the object (its affordance
thesis).  The only inputs are a source image, a target prompt, and the subject
token (the noun naming the added object).
"""

import gc
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter

try:
    from .addit_config import (
        ADDIT_ATTENTION_LAYER_FRAC, ADDIT_ATTENTION_SIGMA_MIN,
        ADDIT_DEBUG_DIR, ADDIT_DEBUG_MAX_ITEMS, ADDIT_DEFAULT_SUBJECT_TOKEN,
        ADDIT_DEFAULT_TARGET_PROMPT, ADDIT_EXTENDED_ATTENTION,
        ADDIT_FINAL_PIXEL_COMPOSITE, ADDIT_GAMMA_FIXED, ADDIT_GAMMA_MODE,
        ADDIT_GAMMA_SEARCH_HI, ADDIT_GAMMA_SEARCH_LO, ADDIT_GAMMA_SOLVE_EVERY,
        ADDIT_GAMMA_SOLVE_ITERS, ADDIT_GAMMA_SOURCE, ADDIT_GUIDANCE_SCALE,
        ADDIT_LATENT_BLENDING, ADDIT_MASK_DILATE_PX, ADDIT_MASK_FEATHER_PX,
        ADDIT_MASK_LAYER_FRACS, ADDIT_MASK_MAX_POINTS,
        ADDIT_MASK_POINT_EXCLUDE_FRAC, ADDIT_MASK_POINT_REL_THRESH,
        ADDIT_NEGATIVE_PROMPT, ADDIT_NUM_INFERENCE_STEPS, ADDIT_OUTPUT_DIR,
        ADDIT_PRIMARY_DEVICE, ADDIT_SAVE_DEBUG, ADDIT_SAVE_MASK_VIS,
        ADDIT_SEED, ADDIT_SOURCE_IS_REAL, ADDIT_SUBJECT_CAPTURE_END_FRAC,
        ADDIT_SUBJECT_CAPTURE_START_FRAC, ADDIT_T_BLEND_FRAC,
        ADDIT_T_STRUCT_FRAC_GEN, ADDIT_T_STRUCT_FRAC_REAL,
        ADDIT_TRANSFORMER_DEVICE, ADDIT_USE_SAM2, ADDIT_USE_TWO_GPUS,
        RESOLUTION, TRAIN_DEVICE,
    )
    from .addit_core import (
        AddItState, attn_vector_to_mask_image, coarse_mask_from_attention,
        composite_outside_mask, decode_latent_to_image, encode_image_to_latent,
        inject_addit_processors, latent_blend, nearest_timestep_index,
        pixel_mask_to_latent, renoise_source, resolve_layer_window,
        resolve_mask_layers, restore_processors, sample_attention_points,
        sigma_for_timestep, solve_gamma,
    )
    from .addit_sam import refine_mask_with_sam2
except ImportError:
    from addit_config import (
        ADDIT_ATTENTION_LAYER_FRAC, ADDIT_ATTENTION_SIGMA_MIN,
        ADDIT_DEBUG_DIR, ADDIT_DEBUG_MAX_ITEMS, ADDIT_DEFAULT_SUBJECT_TOKEN,
        ADDIT_DEFAULT_TARGET_PROMPT, ADDIT_EXTENDED_ATTENTION,
        ADDIT_FINAL_PIXEL_COMPOSITE, ADDIT_GAMMA_FIXED, ADDIT_GAMMA_MODE,
        ADDIT_GAMMA_SEARCH_HI, ADDIT_GAMMA_SEARCH_LO, ADDIT_GAMMA_SOLVE_EVERY,
        ADDIT_GAMMA_SOLVE_ITERS, ADDIT_GAMMA_SOURCE, ADDIT_GUIDANCE_SCALE,
        ADDIT_LATENT_BLENDING, ADDIT_MASK_DILATE_PX, ADDIT_MASK_FEATHER_PX,
        ADDIT_MASK_LAYER_FRACS, ADDIT_MASK_MAX_POINTS,
        ADDIT_MASK_POINT_EXCLUDE_FRAC, ADDIT_MASK_POINT_REL_THRESH,
        ADDIT_NEGATIVE_PROMPT, ADDIT_NUM_INFERENCE_STEPS, ADDIT_OUTPUT_DIR,
        ADDIT_PRIMARY_DEVICE, ADDIT_SAVE_DEBUG, ADDIT_SAVE_MASK_VIS,
        ADDIT_SEED, ADDIT_SOURCE_IS_REAL, ADDIT_SUBJECT_CAPTURE_END_FRAC,
        ADDIT_SUBJECT_CAPTURE_START_FRAC, ADDIT_T_BLEND_FRAC,
        ADDIT_T_STRUCT_FRAC_GEN, ADDIT_T_STRUCT_FRAC_REAL,
        ADDIT_TRANSFORMER_DEVICE, ADDIT_USE_SAM2, ADDIT_USE_TWO_GPUS,
        RESOLUTION, TRAIN_DEVICE,
    )
    from addit_core import (
        AddItState, attn_vector_to_mask_image, coarse_mask_from_attention,
        composite_outside_mask, decode_latent_to_image, encode_image_to_latent,
        inject_addit_processors, latent_blend, nearest_timestep_index,
        pixel_mask_to_latent, renoise_source, resolve_layer_window,
        resolve_mask_layers, restore_processors, sample_attention_points,
        sigma_for_timestep, solve_gamma,
    )
    from addit_sam import refine_mask_with_sam2


def _module_device(module, fallback="cpu") -> torch.device:
    try:
        return next(module.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device(fallback)


def _pipe_offload_enabled(pipe) -> bool:
    """True when accelerate model/sequential CPU offload is active on the pipe.

    Under offload, modules rest on CPU and accelerate hooks move them to the
    execution device just-in-time; their static parameter device therefore
    reports ``cpu`` even though forward runs on CUDA.
    """
    for name in ("_all_hooks", "_offload_gpu_id"):
        if getattr(pipe, name, None):
            return True
    for comp in (getattr(pipe, "vae", None), getattr(pipe, "transformer", None)):
        if comp is not None and hasattr(comp, "_hf_hook"):
            return True
    return False


def _execution_device(pipe, fallback) -> torch.device:
    """Device that hooked forwards actually run on (offload-aware).

    Diffusers exposes ``_execution_device`` precisely for this; fall back to the
    resting param device when it is unavailable.
    """
    dev = getattr(pipe, "_execution_device", None)
    if dev is not None:
        return torch.device(dev)
    return torch.device(fallback)


# ═══════════════════════════════════════════════════════════════════════════
# Result container
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AddItResult:
    success: bool
    result_image: Optional[Image.Image] = None
    source_image: Optional[Image.Image] = None
    subject_token: str = ""
    target_prompt: str = ""
    seed: int = 0
    mask_image: Optional[Image.Image] = None
    attn_image: Optional[Image.Image] = None
    gamma_trace: List[float] = field(default_factory=list)
    used_sam2: bool = False
    debug_path: Optional[Path] = None
    error: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════

class AddItPipeline:
    """Faithful Add-it object-insertion pipeline on a loaded SD3.5 pipeline.

    Parameters
    ----------
    sd35_pipe : StableDiffusion3Pipeline | StableDiffusion3Img2ImgPipeline
        A loaded diffusers SD3.5 pipeline.  We use its ``vae``, ``transformer``,
        ``scheduler``, text encoders and tokenizers directly.
    device : str, optional
        Device the transformer runs on for generation.
    """

    def __init__(self, sd35_pipe, device: Optional[str] = None):
        if sd35_pipe is None:
            raise ValueError("AddItPipeline requires a loaded SD3.5 pipeline.")
        self.pipe = sd35_pipe
        self.vae = sd35_pipe.vae
        self.transformer = sd35_pipe.transformer
        self.scheduler = sd35_pipe.scheduler

        first_param = next(self.transformer.parameters())
        self.dtype = first_param.dtype

        # Under accelerate CPU offload the modules rest on CPU and are moved to
        # the GPU by hooks at forward time, so their static param device reports
        # "cpu". Feeding VAE/transformer inputs to that resting device then
        # collides with the just-in-time-moved weights ("weight is on cuda:0,
        # ... other tensors on cpu"). Use the pipeline's execution device for I/O
        # tensors when offload is active; otherwise keep the resting devices.
        self._offload = _pipe_offload_enabled(self.pipe)
        if self._offload:
            exec_dev = _execution_device(self.pipe, fallback=device or first_param.device)
            self.transformer_device = torch.device(device) if device else exec_dev
            self.vae_device = exec_dev
        else:
            self.transformer_device = torch.device(device or first_param.device)
            self.vae_device = _module_device(self.vae, fallback=str(self.transformer_device))
        self.device = self.transformer_device

        self.state = AddItState()
        self._original_processors = None
        self.num_layers = len(self.transformer.transformer_blocks)

        # Resolve the SD3.5 layer windows from the paper's fractional config.
        self.state.inject_layers = resolve_layer_window(self.num_layers, ADDIT_ATTENTION_LAYER_FRAC)
        self.state.mask_layers = resolve_mask_layers(self.num_layers, ADDIT_MASK_LAYER_FRACS)

    # ------------------------------------------------------------------
    # Prompt encoding + subject-token localisation
    # ------------------------------------------------------------------
    def _encode_prompt(self, prompt: str, negative_prompt: str, device):
        result = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt if getattr(self.pipe, "tokenizer_3", None) is not None else None,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
            negative_prompt_3=negative_prompt if getattr(self.pipe, "tokenizer_3", None) is not None else None,
            device=device,
        )
        return tuple(x.to(device=device) if torch.is_tensor(x) else x for x in result)

    def _subject_token_indices(self, prompt: str, subject_token: str) -> List[int]:
        """Locate the subject word's token positions in the CLIP prompt sequence.

        SD3.5 concatenates CLIP-L + CLIP-G embeddings along the feature dim and
        keeps the 77-token CLIP sequence on the token axis; the T5 stream (when
        present) is concatenated along the token axis after CLIP.  The subject
        token lives in the CLIP segment, so locating it in tokenizer (CLIP-L)
        order gives valid positions for the attention rows.
        """
        tok = getattr(self.pipe, "tokenizer", None)
        if tok is None:
            return []
        try:
            full = tok(prompt, truncation=True, max_length=77,
                       return_tensors=None)["input_ids"]
            # Token ids for the subject word alone (drop BOS/EOS).
            subj_ids = tok(subject_token, add_special_tokens=False,
                           return_tensors=None)["input_ids"]
        except Exception:  # noqa: BLE001
            return []
        if not subj_ids:
            return []
        positions = []
        for i in range(len(full) - len(subj_ids) + 1):
            if full[i:i + len(subj_ids)] == subj_ids:
                positions.extend(range(i, i + len(subj_ids)))
        if not positions:
            # Fallback: match the first sub-token anywhere.
            first = subj_ids[0]
            positions = [i for i, t in enumerate(full) if t == first]
        return positions

    # ------------------------------------------------------------------
    # Source forward pass for K,V caching  (paper §3.2)
    # ------------------------------------------------------------------
    def _cache_source_kv(self, source_noised, t_batch, src_prompt_emb, src_pooled):
        self.state.cache_mode = True
        self.state.inject_mode = False
        self.state.probe_mode = False
        self.state.capture_subject = False
        self.state.clear_cache()
        with torch.no_grad():
            self.transformer(
                hidden_states=source_noised,
                timestep=t_batch,
                encoder_hidden_states=src_prompt_emb,
                pooled_projections=src_pooled,
                return_dict=False,
            )
        self.state.cache_mode = False

    # ------------------------------------------------------------------
    # One target forward pass (optionally probing for γ / subject attention)
    # ------------------------------------------------------------------
    def _target_forward(self, latent, t_batch, prompt_emb, pooled,
                        probe=False, capture=False):
        self.state.inject_mode = ADDIT_EXTENDED_ATTENTION
        self.state.probe_mode = probe
        self.state.capture_subject = capture
        with torch.no_grad():
            pred = self.transformer(
                hidden_states=latent,
                timestep=t_batch,
                encoder_hidden_states=prompt_emb,
                pooled_projections=pooled,
                return_dict=False,
            )[0]
        self.state.inject_mode = False
        self.state.probe_mode = False
        self.state.capture_subject = False
        return pred

    # ------------------------------------------------------------------
    # γ root-solve at the current step  (paper §3.2)
    # ------------------------------------------------------------------
    def _solve_gamma_now(self, latent, t_batch, cond_emb, cond_pooled):
        """Solve f(γ)=A_source−A_target=0 by probing single-cond forwards."""
        def probe(gamma):
            self.state.gamma_source = ADDIT_GAMMA_SOURCE
            self.state.gamma_target = gamma
            self.state.reset_probe()
            self._target_forward(latent, t_batch, cond_emb, cond_pooled,
                                 probe=True, capture=False)
            n = max(1.0, self.state.probe_accum.get("n", 1.0))
            return (self.state.probe_accum["a_source"] / n,
                    self.state.probe_accum["a_target"] / n)

        return solve_gamma(
            probe,
            gamma_lo=ADDIT_GAMMA_SEARCH_LO,
            gamma_hi=ADDIT_GAMMA_SEARCH_HI,
            iters=ADDIT_GAMMA_SOLVE_ITERS,
        )

    # ------------------------------------------------------------------
    # Main generation
    # ------------------------------------------------------------------
    def add_object(
        self,
        source_image: Image.Image,
        target_prompt: Optional[str] = None,
        subject_token: Optional[str] = None,
        seed: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        source_is_real: Optional[bool] = None,
    ) -> AddItResult:
        target_prompt = target_prompt or ADDIT_DEFAULT_TARGET_PROMPT
        subject_token = subject_token or ADDIT_DEFAULT_SUBJECT_TOKEN
        seed = ADDIT_SEED if seed is None else seed
        steps = num_inference_steps or ADDIT_NUM_INFERENCE_STEPS
        guidance = ADDIT_GUIDANCE_SCALE if guidance_scale is None else guidance_scale
        source_is_real = ADDIT_SOURCE_IS_REAL if source_is_real is None else source_is_real

        device = self.transformer_device
        vae_device = self.vae_device
        dtype = self.dtype
        gen_device = device if str(device).startswith("cuda") else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(seed)

        source_image = source_image.convert("RGB").resize((RESOLUTION, RESOLUTION), Image.LANCZOS)

        # ── Encode source latent ──
        source_latent = encode_image_to_latent(
            self.vae, source_image, RESOLUTION, vae_device, dtype
        ).to(device=device, dtype=dtype)
        lat_h, lat_w = source_latent.shape[2], source_latent.shape[3]

        # ── Scheduler timesteps ──
        self.scheduler.set_timesteps(steps, device=device)
        all_timesteps = self.scheduler.timesteps                         # high → low

        # ── Structure Transfer (§3.3): start at fixed t_struct ──
        num_train = float(getattr(self.scheduler.config, "num_train_timesteps", 1000))
        t_struct_frac = ADDIT_T_STRUCT_FRAC_REAL if source_is_real else ADDIT_T_STRUCT_FRAC_GEN
        t_struct_value = t_struct_frac * num_train
        start_idx = nearest_timestep_index(all_timesteps, t_struct_value)
        timesteps = all_timesteps[start_idx:]
        total_steps = len(timesteps)
        if total_steps == 0:
            return AddItResult(success=False, source_image=source_image,
                              error="empty_timestep_schedule")

        # Single random ε for the whole run (paper §3.5).
        noise = torch.randn(source_latent.shape, generator=generator,
                            device=device, dtype=dtype)
        # Start latent = source noised to t_struct.
        latent = renoise_source(source_latent, noise, self.scheduler, timesteps[0])

        # ── t_blend index (§3.4) ──
        t_blend_value = ADDIT_T_BLEND_FRAC * num_train
        blend_idx = nearest_timestep_index(timesteps, t_blend_value)

        # ── Encode prompts ──
        src_prompt = ADDIT_DEFAULT_TARGET_PROMPT  # source prompt = scene description
        source_embeds = self._encode_prompt(target_prompt, ADDIT_NEGATIVE_PROMPT, device)
        # SD3.5 encode_prompt → (prompt_emb, neg_emb, pooled, neg_pooled)
        tgt_prompt_emb, tgt_neg_emb, tgt_pooled, tgt_neg_pooled = source_embeds
        # Source uses the same target prompt's conditional embed (it denoises the
        # scene; its job is to provide structurally-aligned K,V, not to add the
        # object).  Using the cond embed keeps source/target in the same space.
        src_prompt_emb = tgt_prompt_emb
        src_pooled = tgt_pooled

        subject_ids = self._subject_token_indices(target_prompt, subject_token)
        self.state.subject_token_ids = subject_ids
        if not subject_ids:
            print(f"Add-it: subject token '{subject_token}' not located in prompt; "
                  "mask capture will be weak.")

        # ── Inject processors ──
        if ADDIT_EXTENDED_ATTENTION:
            self._original_processors = inject_addit_processors(self.transformer, self.state)
            self.state.enabled = True
        self.state.gamma_source = ADDIT_GAMMA_SOURCE
        self.state.gamma_target = ADDIT_GAMMA_FIXED

        gamma_trace: List[float] = []
        cap_start = ADDIT_SUBJECT_CAPTURE_START_FRAC
        cap_end = ADDIT_SUBJECT_CAPTURE_END_FRAC
        self.state.reset_subject_capture()

        try:
            for i, t in enumerate(timesteps):
                step_ratio = i / max(1, total_steps - 1)
                sigma_t = sigma_for_timestep(self.scheduler, t)
                t_batch = (t.unsqueeze(0) if t.dim() == 0 else t).to(device)

                inject_now = (
                    ADDIT_EXTENDED_ATTENTION and sigma_t >= ADDIT_ATTENTION_SIGMA_MIN
                )
                self.state.inject_layers = (
                    resolve_layer_window(self.num_layers, ADDIT_ATTENTION_LAYER_FRAC)
                    if inject_now else set()
                )

                # ── Re-noise + cache source K,V (§3.2/§3.5) ──
                source_noised_t = renoise_source(source_latent, noise, self.scheduler, t)
                if inject_now:
                    self._cache_source_kv(
                        source_noised_t, t_batch, src_prompt_emb, src_pooled
                    )

                # ── γ balancing (§3.2): re-solve periodically ──
                if (inject_now and ADDIT_GAMMA_MODE == "auto"
                        and (i % max(1, ADDIT_GAMMA_SOLVE_EVERY) == 0)):
                    g = self._solve_gamma_now(latent, t_batch, tgt_prompt_emb, tgt_pooled)
                    self.state.gamma_target = g
                gamma_trace.append(float(self.state.gamma_target))

                # ── Subject-attention capture window (§3.4) ──
                capture_now = (
                    bool(subject_ids) and inject_now
                    and cap_start <= step_ratio <= cap_end
                )

                # ── Target forward with CFG ──
                latent_in = torch.cat([latent, latent])
                emb_cfg = torch.cat([tgt_neg_emb, tgt_prompt_emb])
                pooled_cfg = torch.cat([tgt_neg_pooled, tgt_pooled])
                t_cfg = t_batch.expand(2)
                pred = self._target_forward(
                    latent_in, t_cfg, emb_cfg, pooled_cfg,
                    probe=False, capture=capture_now,
                )
                pred_uncond, pred_cond = pred.chunk(2)
                pred_cfg = pred_uncond + guidance * (pred_cond - pred_uncond)

                # ── Scheduler step ──
                latent = self.scheduler.step(pred_cfg, t, latent, return_dict=False)[0]

                # ── Subject-Guided Latent Blending at t_blend (§3.4) ──
                if ADDIT_LATENT_BLENDING and i == blend_idx:
                    mask_img, attn_img, used_sam2 = self._build_subject_mask(
                        latent, source_latent, noise, t, (lat_h, lat_w),
                        source_image, subject_token, vae_device,
                    )
                    if mask_img is not None:
                        mask_lat = pixel_mask_to_latent(
                            mask_img, (lat_h, lat_w), latent.device, latent.dtype
                        )
                        source_noised_blend = renoise_source(
                            source_latent, noise, self.scheduler, t
                        )
                        latent = latent_blend(latent, source_noised_blend, mask_lat)
                        self._last_mask = mask_img
                        self._last_attn = attn_img
                        self._last_used_sam2 = used_sam2
        finally:
            self.state.enabled = False
            self.state.inject_mode = False
            self.state.cache_mode = False
            self.state.probe_mode = False
            self.state.capture_subject = False
            self.state.clear_cache()
            if self._original_processors is not None:
                restore_processors(self.transformer, self._original_processors)
                self._original_processors = None

        # ── Decode + optional pixel composite outside the mask ──
        result_image = decode_latent_to_image(self.vae, latent.to(vae_device))
        mask_img = getattr(self, "_last_mask", None)
        attn_img = getattr(self, "_last_attn", None)
        used_sam2 = getattr(self, "_last_used_sam2", False)
        self._last_mask = self._last_attn = None

        if ADDIT_FINAL_PIXEL_COMPOSITE and mask_img is not None:
            result_image = composite_outside_mask(source_image, result_image, mask_img)

        return AddItResult(
            success=True,
            result_image=result_image,
            source_image=source_image,
            subject_token=subject_token,
            target_prompt=target_prompt,
            seed=seed,
            mask_image=mask_img,
            attn_image=attn_img,
            gamma_trace=gamma_trace,
            used_sam2=used_sam2,
        )

    # ------------------------------------------------------------------
    # Subject mask construction  (paper §3.4 + App A.1)
    # ------------------------------------------------------------------
    def _build_subject_mask(self, latent, source_latent, noise, t, latent_hw,
                            source_image, subject_token, vae_device):
        """Otsu coarse mask → point prompts → SAM-2 refine.

        Returns (mask_image_L, attn_image_L, used_sam2).
        """
        if self.state.subject_attn_accum is None or self.state.subject_attn_count == 0:
            return None, None, False

        h, w = latent_hw
        attn_vec = self.state.subject_attn_accum / max(1, self.state.subject_attn_count)
        attn_img = attn_vector_to_mask_image(
            attn_vec, (h, w), (RESOLUTION, RESOLUTION)
        )

        coarse = coarse_mask_from_attention(attn_img)             # Otsu (§3.4)
        points = sample_attention_points(
            attn_img,
            max_points=ADDIT_MASK_MAX_POINTS,
            rel_thresh=ADDIT_MASK_POINT_REL_THRESH,
            exclude_radius_frac=ADDIT_MASK_POINT_EXCLUDE_FRAC,
        )

        # Estimate the clean image X_0 at t_blend for SAM-2 (paper §3.4:
        # X_0 ≈ latent advanced to σ=0).  We decode the current denoised latent
        # as a practical X_0 estimate (it is already near-clean at t_blend≈0.5).
        x0_image = decode_latent_to_image(self.vae, latent.to(vae_device))

        used_sam2 = False
        if ADDIT_USE_SAM2 and points:
            mask_img = refine_mask_with_sam2(x0_image, points, coarse_mask=coarse)
            used_sam2 = True
        else:
            mask_img = Image.fromarray(coarse, mode="L").resize(
                (RESOLUTION, RESOLUTION), Image.NEAREST
            )

        # Post-process: dilate (keep shadows/contact inside M) + feather edge.
        if ADDIT_MASK_DILATE_PX > 0:
            mask_img = mask_img.filter(ImageFilter.MaxFilter(size=1 + 2 * ADDIT_MASK_DILATE_PX))
        if ADDIT_MASK_FEATHER_PX > 0:
            mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=ADDIT_MASK_FEATHER_PX))

        return mask_img, attn_img, used_sam2

    # ------------------------------------------------------------------
    # Step-by-step generation  (paper §3.5)
    # ------------------------------------------------------------------
    def step_by_step(
        self,
        source_image: Image.Image,
        edits: List[Tuple[str, str]],
        seed: Optional[int] = None,
    ) -> List[AddItResult]:
        """Iteratively add objects: each (target_prompt, subject_token) edit is
        applied to the previous output, building a complex scene (paper §3.5,
        fig. 1 & 10).  The first edit treats the input as a real source; later
        edits operate on Add-it outputs (still handled as real images via the
        no-inversion re-noising)."""
        results = []
        current = source_image
        for j, (prompt, subj) in enumerate(edits):
            res = self.add_object(
                current, target_prompt=prompt, subject_token=subj,
                seed=(ADDIT_SEED if seed is None else seed) + j,
            )
            results.append(res)
            if res.success and res.result_image is not None:
                current = res.result_image
        return results


# ═══════════════════════════════════════════════════════════════════════════
# Debug / save helpers
# ═══════════════════════════════════════════════════════════════════════════

def save_addit_debug(result: AddItResult, out_dir=ADDIT_DEBUG_DIR,
                     tag: str = "addit") -> Optional[Path]:
    if not ADDIT_SAVE_DEBUG or result.source_image is None or result.result_image is None:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(list(out_dir.glob("*.png"))) >= ADDIT_DEBUG_MAX_ITEMS:
        return None

    src = result.source_image
    res = result.result_image.resize(src.size)
    panels = [src, res]
    if ADDIT_SAVE_MASK_VIS and result.attn_image is not None:
        panels.append(result.attn_image.convert("RGB").resize(src.size))
    if ADDIT_SAVE_MASK_VIS and result.mask_image is not None:
        panels.append(result.mask_image.convert("RGB").resize(src.size))

    w, h = src.size
    strip = Image.new("RGB", (w * len(panels), h), "white")
    for k, p in enumerate(panels):
        strip.paste(p, (w * k, 0))
    path = out_dir / f"{tag}_{result.seed}_{result.subject_token}.png"
    strip.save(path)
    result.debug_path = path
    return path
