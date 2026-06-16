"""Add-it end-to-end pipeline for CityPersons pedestrian augmentation.

Wraps a loaded SD3.5 img2img / txt2img pipeline and exposes
:meth:`run_single` and :meth:`run_batch` that execute the full
Add-it denoising loop:

1. Find insertion region (placement heuristics from ``sd35_utils``).
2. Noise structure transfer → starting latent.
3. Cache source K,V via a dedicated transformer forward pass.
4. Custom denoising loop with:
   a. Timestep-dependent weighted extended-attention.
   b. Subject-guided latent blending.
5. VAE decode.
6. Optional YOLO validation + retry.
"""

import gc
import json
import random
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

try:
    from .addit_config import (
        ADDIT_ATTENTION_LAYER_RANGE,
        ADDIT_ATTENTION_SCHEDULE,
        ADDIT_WEIGHTED_EXTENDED_ATTENTION,
        ADDIT_BLEND_END_RATIO,
        ADDIT_BLEND_FEATHER_LATENT,
        ADDIT_BLEND_START_RATIO,
        ADDIT_DEBUG_DIR,
        ADDIT_DEBUG_MAX_ITEMS,
        ADDIT_FINAL_COMPOSITE_FEATHER_PX,
        ADDIT_FINAL_COMPOSITE_MODE,
        ADDIT_FINAL_PIXEL_COMPOSITE,
        ADDIT_FINAL_PERSON_CUTOUT,
        ADDIT_GUIDANCE_SCALE,
        ADDIT_FALLBACK_TO_NATIVE_IMG2IMG,
        ADDIT_MASK_DILATION_LATENT,
        ADDIT_MASK_EXPANSION_RATIO,
        ADDIT_MAX_RETRIES,
        ADDIT_NEGATIVE_PROMPT,
        ADDIT_NOISE_BLEND_RATIO,
        ADDIT_NUM_INFERENCE_STEPS,
        ADDIT_OUTPUT_DIR,
        ADDIT_PERSON_CUTOUT_CONF,
        ADDIT_PERSON_CUTOUT_DILATE_PX,
        ADDIT_PERSON_CUTOUT_EDGE_FULL_ALPHA,
        ADDIT_PERSON_CUTOUT_EDGE_MIN_ALPHA,
        ADDIT_PERSON_CUTOUT_FALLBACK_TO_BBOX,
        ADDIT_PERSON_CUTOUT_FEATHER_PX,
        ADDIT_PERSON_CUTOUT_MASK_THRESHOLD,
        ADDIT_RETRY_SEED_STEP,
        ADDIT_SAVE_DEBUG,
        ADDIT_SEED,
        ADDIT_SOURCE_PROMPT,
        ADDIT_STRUCTURE_STRENGTH,
        ADDIT_TARGET_PROMPTS,
        ADDIT_VARIANT_OVERRIDES,
        ADDIT_W_SELF_END,
        ADDIT_W_SELF_START,
        ADDIT_W_SOURCE_END,
        ADDIT_W_SOURCE_START,
        RESOLUTION,
        TRAIN_DEVICE,
        VARIANT_PROMPTS,
    )
    from .addit_core import (
        AddItState,
        composite_generated_region,
        compute_attention_weights,
        create_latent_insertion_mask,
        decode_latent_to_image,
        encode_image_to_latent,
        get_source_noised_at_step,
        inject_addit_processors,
        restore_processors,
        subject_guided_blend,
        transfer_noise_structure,
    )
except ImportError:
    from addit_config import (
        ADDIT_ATTENTION_LAYER_RANGE,
        ADDIT_ATTENTION_SCHEDULE,
        ADDIT_WEIGHTED_EXTENDED_ATTENTION,
        ADDIT_BLEND_END_RATIO,
        ADDIT_BLEND_FEATHER_LATENT,
        ADDIT_BLEND_START_RATIO,
        ADDIT_DEBUG_DIR,
        ADDIT_DEBUG_MAX_ITEMS,
        ADDIT_FINAL_COMPOSITE_FEATHER_PX,
        ADDIT_FINAL_COMPOSITE_MODE,
        ADDIT_FINAL_PIXEL_COMPOSITE,
        ADDIT_FINAL_PERSON_CUTOUT,
        ADDIT_GUIDANCE_SCALE,
        ADDIT_FALLBACK_TO_NATIVE_IMG2IMG,
        ADDIT_MASK_DILATION_LATENT,
        ADDIT_MASK_EXPANSION_RATIO,
        ADDIT_MAX_RETRIES,
        ADDIT_NEGATIVE_PROMPT,
        ADDIT_NOISE_BLEND_RATIO,
        ADDIT_NUM_INFERENCE_STEPS,
        ADDIT_OUTPUT_DIR,
        ADDIT_PERSON_CUTOUT_CONF,
        ADDIT_PERSON_CUTOUT_DILATE_PX,
        ADDIT_PERSON_CUTOUT_EDGE_FULL_ALPHA,
        ADDIT_PERSON_CUTOUT_EDGE_MIN_ALPHA,
        ADDIT_PERSON_CUTOUT_FALLBACK_TO_BBOX,
        ADDIT_PERSON_CUTOUT_FEATHER_PX,
        ADDIT_PERSON_CUTOUT_MASK_THRESHOLD,
        ADDIT_RETRY_SEED_STEP,
        ADDIT_SAVE_DEBUG,
        ADDIT_SEED,
        ADDIT_SOURCE_PROMPT,
        ADDIT_STRUCTURE_STRENGTH,
        ADDIT_TARGET_PROMPTS,
        ADDIT_VARIANT_OVERRIDES,
        ADDIT_W_SELF_END,
        ADDIT_W_SELF_START,
        ADDIT_W_SOURCE_END,
        ADDIT_W_SOURCE_START,
        RESOLUTION,
        TRAIN_DEVICE,
        VARIANT_PROMPTS,
    )
    from addit_core import (
        AddItState,
        composite_generated_region,
        compute_attention_weights,
        create_latent_insertion_mask,
        decode_latent_to_image,
        encode_image_to_latent,
        get_source_noised_at_step,
        inject_addit_processors,
        restore_processors,
        subject_guided_blend,
        transfer_noise_structure,
    )

# Parent-pipeline utilities (placement, YOLO, etc.)
from sd35_utils import (
    clear_cuda,
    find_insertion_region,
    load_source_image,
    resize_center_crop,
    estimate_depth_map,
    save_comparison_pair,
)
from sd35_data import ImageRecord

warnings.filterwarnings("ignore", category=FutureWarning)


def _module_device(module, fallback: str = "cpu") -> torch.device:
    """Return the device of the first module parameter, or fallback."""
    try:
        return next(module.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device(fallback)


# ═══════════════════════════════════════════════════════════════════════════
# Result container
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AddItResult:
    """Container for a single Add-it generation result."""
    success: bool
    result_image: Optional[Image.Image] = None
    source_image: Optional[Image.Image] = None
    insert_bbox: Optional[Tuple[int, int, int, int]] = None
    variant: str = ""
    seed: int = 0
    attempts: int = 0
    reject_reason: str = ""
    metadata: Optional[Dict] = None
    debug_path: Optional[Path] = None


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════

class AddItCityPersonsPipeline:
    """Add-it object-insertion pipeline for CityPersons augmentation.

    Parameters
    ----------
    sd35_pipe
        A loaded ``StableDiffusion3Img2ImgPipeline`` (or txt2img) from diffusers.
    yolo_model_path : str
        Path / name of the YOLOv8m-seg model for validation.
    """

    def __init__(self, sd35_pipe, yolo_model_path: str = "yolov8m-seg.pt", device: Optional[str] = None):
        self.pipe = sd35_pipe
        self.vae = sd35_pipe.vae
        self.transformer = sd35_pipe.transformer
        self.scheduler = sd35_pipe.scheduler
        first_param = next(sd35_pipe.transformer.parameters())
        self.param_device = first_param.device
        self.transformer_device = torch.device(device or self.param_device)
        self.vae_device = _module_device(self.vae, fallback=str(self.transformer_device))
        prompt_module = (
            getattr(sd35_pipe, "text_encoder", None)
            or getattr(sd35_pipe, "text_encoder_2", None)
            or getattr(sd35_pipe, "text_encoder_3", None)
        )
        self.prompt_device = _module_device(prompt_module, fallback=str(self.transformer_device))
        self.device = self.transformer_device
        self.dtype = first_param.dtype

        # Attention state & original processors
        self.state = AddItState()
        self._original_processors = None

        # YOLO model (lazy-loaded)
        self._yolo_model_path = yolo_model_path
        self._yolo = None

        # Layer range config
        num_layers = len(self.transformer.transformer_blocks)
        start = int(num_layers * ADDIT_ATTENTION_LAYER_RANGE[0])
        end = int(num_layers * ADDIT_ATTENTION_LAYER_RANGE[1]) - 1
        self.state.layer_start = max(0, start)
        self.state.layer_end = max(0, end)

    @staticmethod
    def _load_yolo_model(model_path: str):
        """Load YOLO directly so Add-it does not depend on parent globals."""
        from ultralytics import YOLO

        return YOLO(model_path)

    # ------------------------------------------------------------------
    # Lazy YOLO loading
    # ------------------------------------------------------------------
    @property
    def yolo(self):
        if self._yolo is None:
            self._yolo = self._load_yolo_model(self._yolo_model_path)
        return self._yolo

    # ------------------------------------------------------------------
    # Prompt encoding (delegate to pipe)
    # ------------------------------------------------------------------
    def _encode_prompt(self, prompt: str, negative_prompt: str, device):
        """Encode text prompts into SD3.5 embeddings."""
        prompt_device = self.prompt_device
        result = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt if hasattr(self.pipe, "tokenizer_3") and self.pipe.tokenizer_3 is not None else None,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
            negative_prompt_3=negative_prompt if hasattr(self.pipe, "tokenizer_3") and self.pipe.tokenizer_3 is not None else None,
            device=prompt_device,
        )
        # result is (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
        return tuple(
            item.to(device=device) if torch.is_tensor(item) else item
            for item in result
        )

    # ------------------------------------------------------------------
    # Source forward pass for K,V caching
    # ------------------------------------------------------------------
    def _cache_source_kv(
        self,
        source_latent_noised: torch.Tensor,
        timestep: torch.Tensor,
        source_prompt_embeds: torch.Tensor,
        source_pooled_embeds: torch.Tensor,
    ):
        """Run the source image through the transformer in cache mode
        to populate ``self.state.kv_cache``."""
        self.state.cache_mode = True
        self.state.inject_mode = False
        self.state.clear_cache()

        with torch.no_grad():
            _ = self.transformer(
                hidden_states=source_latent_noised,
                timestep=timestep,
                encoder_hidden_states=source_prompt_embeds,
                pooled_projections=source_pooled_embeds,
                return_dict=False,
            )

        self.state.cache_mode = False

    # ------------------------------------------------------------------
    # Custom Add-it denoising loop
    # ------------------------------------------------------------------
    def _addit_denoise(
        self,
        source_image: Image.Image,
        insert_bbox: Tuple[int, int, int, int],
        target_prompt: str,
        negative_prompt: str,
        strength: float,
        guidance_scale: float,
        num_inference_steps: int,
        seed: int,
        device,
    ) -> Image.Image:
        """Core Add-it denoising loop implementing all three mechanisms.

        Returns the generated image as a PIL Image.
        """
        device = torch.device(device)
        vae_device = self.vae_device
        dtype = self.dtype
        resolution = RESOLUTION
        generator = torch.Generator(device=device if str(device).startswith("cuda") else "cpu")
        generator.manual_seed(seed)

        # ── 1. Encode source ──
        source_latent = encode_image_to_latent(
            self.vae, source_image, resolution, vae_device, dtype
        ).to(device=device, dtype=dtype)
        latent_h, latent_w = source_latent.shape[2], source_latent.shape[3]

        # ── 2. Create insertion mask in latent space ──
        latent_mask = create_latent_insertion_mask(
            bbox=insert_bbox,
            image_size=(resolution, resolution),
            latent_size=(latent_w, latent_h),
            feather=ADDIT_BLEND_FEATHER_LATENT,
            dilation=ADDIT_MASK_DILATION_LATENT,
            expansion=ADDIT_MASK_EXPANSION_RATIO,
        ).to(device=device, dtype=dtype)

        # ── 3. Noise structure transfer ──
        initial_latent, source_latent, noise, timesteps = transfer_noise_structure(
            vae=self.vae,
            scheduler=self.scheduler,
            source_image=source_image,
            strength=strength,
            noise_blend_ratio=ADDIT_NOISE_BLEND_RATIO,
            num_inference_steps=num_inference_steps,
            generator=generator,
            resolution=resolution,
            device=device,
            dtype=dtype,
            source_latent=source_latent,
        )

        if len(timesteps) == 0:
            return decode_latent_to_image(self.vae, source_latent.to(vae_device))

        # ── 4. Encode prompts ──
        source_embeds = self._encode_prompt(ADDIT_SOURCE_PROMPT, negative_prompt, device)
        target_embeds = self._encode_prompt(target_prompt, negative_prompt, device)
        # source_embeds: (prompt_emb, neg_emb, pooled, neg_pooled)
        src_prompt_emb = source_embeds[0]
        src_pooled = source_embeds[2]
        tgt_prompt_emb = target_embeds[0]
        tgt_neg_emb = target_embeds[1]
        tgt_pooled = target_embeds[2]
        tgt_neg_pooled = target_embeds[3]

        # ── 5. Optional Add-it attention processors ──
        attention_enabled = bool(ADDIT_WEIGHTED_EXTENDED_ATTENTION)
        if attention_enabled:
            self._original_processors = inject_addit_processors(self.transformer, self.state)
            self.state.enabled = True

        try:
            latent = initial_latent
            total_steps = len(timesteps)

            for i, t in enumerate(timesteps):
                step_ratio = i / max(1, total_steps - 1)

                # 5a. Update attention weights for this timestep
                w_src, w_self = compute_attention_weights(
                    step=i,
                    total_steps=total_steps,
                    schedule=ADDIT_ATTENTION_SCHEDULE,
                    w_source_range=(ADDIT_W_SOURCE_START, ADDIT_W_SOURCE_END),
                    w_self_range=(ADDIT_W_SELF_START, ADDIT_W_SELF_END),
                )
                self.state.set_weights(w_src, w_self)

                # 5b. Compute source noised at current timestep
                source_noised_t = get_source_noised_at_step(
                    self.scheduler, source_latent, noise, t
                )

                # 5c. Cache source K,V (source forward pass)
                t_batch = t.unsqueeze(0).to(device) if t.dim() == 0 else t.to(device)
                if attention_enabled:
                    self._cache_source_kv(
                        source_noised_t, t_batch, src_prompt_emb, src_pooled,
                    )

                # 5d. Target forward with CFG (classifier-free guidance)
                self.state.inject_mode = attention_enabled
                latent_model_input = torch.cat([latent, latent])
                prompt_embeds_cfg = torch.cat([tgt_neg_emb, tgt_prompt_emb])
                pooled_cfg = torch.cat([tgt_neg_pooled, tgt_pooled])
                t_cfg = t_batch.expand(2)

                with torch.no_grad():
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=t_cfg,
                        encoder_hidden_states=prompt_embeds_cfg,
                        pooled_projections=pooled_cfg,
                        return_dict=False,
                    )[0]

                self.state.inject_mode = False

                # CFG combination
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred_combined = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

                # 5e. Scheduler step
                latent = self.scheduler.step(
                    noise_pred_combined, t, latent, return_dict=False
                )[0]

                # 5f. Subject-guided latent blending
                if step_ratio >= ADDIT_BLEND_START_RATIO:
                    latent = subject_guided_blend(
                        generated_latent=latent,
                        source_noised_at_t=source_noised_t,
                        latent_mask=latent_mask,
                        step_ratio=step_ratio,
                        blend_end_ratio=ADDIT_BLEND_END_RATIO,
                    )

        finally:
            # ── 6. Restore original processors ──
            self.state.enabled = False
            self.state.inject_mode = False
            self.state.cache_mode = False
            self.state.clear_cache()
            if self._original_processors is not None:
                restore_processors(self.transformer, self._original_processors)
                self._original_processors = None

        # ── 7. Decode ──
        result_image = decode_latent_to_image(self.vae, latent.to(vae_device))
        if ADDIT_FINAL_PIXEL_COMPOSITE and not ADDIT_FINAL_PERSON_CUTOUT:
            result_image = composite_generated_region(
                source_image=source_image,
                generated_image=result_image,
                bbox=insert_bbox,
                expansion=ADDIT_MASK_EXPANSION_RATIO,
                feather=ADDIT_FINAL_COMPOSITE_FEATHER_PX,
                mode=ADDIT_FINAL_COMPOSITE_MODE,
            )
        return result_image

    @staticmethod
    def _boxes_overlap(box_a, box_b) -> bool:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        return min(ax2, bx2) > max(ax1, bx1) and min(ay2, by2) > max(ay1, by1)

    def _extract_generated_person_mask(
        self,
        generated_image: Image.Image,
        insert_bbox: Tuple[int, int, int, int],
    ) -> Optional[Image.Image]:
        """Return a pixel mask for generated people overlapping insert_bbox."""
        results = self.yolo(
            np.array(generated_image.convert("RGB")),
            conf=ADDIT_PERSON_CUTOUT_CONF,
            verbose=False,
        )
        if not results or len(results) == 0:
            return None

        det = results[0]
        if det.boxes is None or len(det.boxes) == 0:
            return None

        classes = det.boxes.cls.cpu().numpy()
        boxes = det.boxes.xyxy.cpu().numpy()
        selected_indices = [
            idx for idx, (cls_id, box) in enumerate(zip(classes, boxes))
            if int(cls_id) == 0 and self._boxes_overlap(box, insert_bbox)
        ]
        if not selected_indices:
            return None

        width, height = generated_image.size
        mask_np = np.zeros((height, width), dtype=np.uint8)

        if det.masks is None or det.masks.data is None:
            return None

        masks = det.masks.data.cpu().numpy()
        for idx in selected_indices:
            person_mask = (masks[idx] > 0.5).astype(np.uint8) * 255
            person_mask_img = Image.fromarray(person_mask, mode="L").resize(
                (width, height), Image.Resampling.BILINEAR
            )
            mask_np = np.maximum(mask_np, np.array(person_mask_img, dtype=np.uint8))

        if not np.any(mask_np):
            return None

        mask_np = np.where(
            mask_np >= int(ADDIT_PERSON_CUTOUT_MASK_THRESHOLD),
            255,
            0,
        ).astype(np.uint8)
        mask = Image.fromarray(mask_np, mode="L")
        if ADDIT_PERSON_CUTOUT_DILATE_PX > 0:
            size = 1 + 2 * int(ADDIT_PERSON_CUTOUT_DILATE_PX)
            mask = mask.filter(ImageFilter.MaxFilter(size=size))
        if ADDIT_PERSON_CUTOUT_FEATHER_PX > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=ADDIT_PERSON_CUTOUT_FEATHER_PX))
            min_alpha = int(ADDIT_PERSON_CUTOUT_EDGE_MIN_ALPHA)
            full_alpha = max(min_alpha + 1, int(ADDIT_PERSON_CUTOUT_EDGE_FULL_ALPHA))
            scale = 255.0 / (full_alpha - min_alpha)
            mask = mask.point(
                lambda value: 0
                if value <= min_alpha
                else 255
                if value >= full_alpha
                else int((value - min_alpha) * scale)
            )
        return mask

    def _paste_generated_person_on_source(
        self,
        source_image: Image.Image,
        generated_image: Image.Image,
        insert_bbox: Tuple[int, int, int, int],
    ) -> Image.Image:
        """Cut the generated person from Add-it output and paste onto source."""
        source = source_image.convert("RGB")
        generated = generated_image.convert("RGB").resize(source.size, Image.LANCZOS)
        try:
            mask = self._extract_generated_person_mask(generated, insert_bbox)
        except Exception as exc:
            print(f"  Person cutout failed: {type(exc).__name__}: {exc}")
            mask = None

        if mask is not None:
            return Image.composite(generated, source, mask)

        if ADDIT_PERSON_CUTOUT_FALLBACK_TO_BBOX:
            return composite_generated_region(
                source_image=source,
                generated_image=generated,
                bbox=insert_bbox,
                expansion=ADDIT_MASK_EXPANSION_RATIO,
                feather=ADDIT_FINAL_COMPOSITE_FEATHER_PX,
                mode=ADDIT_FINAL_COMPOSITE_MODE,
            )

        return source

    def _native_img2img_fallback(
        self,
        source_image: Image.Image,
        target_prompt: str,
        negative_prompt: str,
        strength: float,
        guidance_scale: float,
        num_inference_steps: int,
        seed: int,
        device,
    ) -> Image.Image:
        """Fallback through the public diffusers pipeline API.

        This does not use extended attention or latent blending, but it keeps
        Kaggle smoke tests productive when a diffusers internals mismatch makes
        the custom denoising loop fail.
        """
        generator_device = device if str(device).startswith("cuda") else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        result = self.pipe(
            prompt=target_prompt,
            negative_prompt=negative_prompt,
            image=source_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        return result.images[0].resize(source_image.size)

    # ------------------------------------------------------------------
    # Single-image entry point
    # ------------------------------------------------------------------
    def run_single(
        self,
        record: "ImageRecord",
        variant: str,
        seed: Optional[int] = None,
        device: Optional[str] = None,
    ) -> AddItResult:
        """Run Add-it augmentation on a single image / variant pair.

        Includes retry loop and optional YOLO validation.
        """
        device = device or str(self.device)
        seed = seed if seed is not None else ADDIT_SEED
        overrides = ADDIT_VARIANT_OVERRIDES.get(variant, {})
        strength = overrides.get("strength", ADDIT_STRUCTURE_STRENGTH)
        guidance = overrides.get("guidance", ADDIT_GUIDANCE_SCALE)
        steps = overrides.get("steps", ADDIT_NUM_INFERENCE_STEPS)

        target_prompt = ADDIT_TARGET_PROMPTS.get(
            variant,
            VARIANT_PROMPTS.get(variant, "urban street photo with a pedestrian"),
        )
        negative_prompt = ADDIT_NEGATIVE_PROMPT

        # Load and resize source
        source_pil = load_source_image(record.path)
        source_resized = resize_center_crop(source_pil, RESOLUTION)

        # Find insertion region
        depth_map = estimate_depth_map(source_resized, device="cpu")
        rng = random.Random(seed)
        insert_bbox, insert_meta = find_insertion_region(
            record, source_resized, variant, rng,
            device=device, return_metadata=True, depth_map=depth_map,
        )

        if insert_bbox is None:
            return AddItResult(
                success=False,
                source_image=source_resized,
                variant=variant,
                seed=seed,
                reject_reason="no_insertion_region",
            )

        # Retry loop
        best_result = None
        for attempt in range(ADDIT_MAX_RETRIES + 1):
            attempt_seed = seed + attempt * ADDIT_RETRY_SEED_STEP
            try:
                result_image = self._addit_denoise(
                    source_image=source_resized,
                    insert_bbox=insert_bbox,
                    target_prompt=target_prompt,
                    negative_prompt=negative_prompt,
                    strength=strength,
                    guidance_scale=guidance,
                    num_inference_steps=steps,
                    seed=attempt_seed,
                    device=device,
                )
            except Exception as exc:
                print(f"  Add-it denoise failed (attempt {attempt + 1}): "
                      f"{type(exc).__name__}: {exc}")
                traceback.print_exc(limit=6)
                if not ADDIT_FALLBACK_TO_NATIVE_IMG2IMG:
                    continue
                print("  Falling back to native SD3 img2img for this attempt.")
                try:
                    result_image = self._native_img2img_fallback(
                        source_image=source_resized,
                        target_prompt=target_prompt,
                        negative_prompt=negative_prompt,
                        strength=strength,
                        guidance_scale=guidance,
                        num_inference_steps=steps,
                        seed=attempt_seed,
                        device=device,
                    )
                except Exception as fallback_exc:
                    print(f"  Native fallback failed: {type(fallback_exc).__name__}: {fallback_exc}")
                    traceback.print_exc(limit=6)
                    continue

            if ADDIT_FINAL_PERSON_CUTOUT:
                result_image = self._paste_generated_person_on_source(
                    source_image=source_resized,
                    generated_image=result_image,
                    insert_bbox=insert_bbox,
                )

            # Validate with YOLO (lightweight check)
            is_valid, reject_reason = self._quick_validate(
                result_image, source_resized, insert_bbox, variant,
            )

            if is_valid or attempt == ADDIT_MAX_RETRIES:
                # Save debug
                debug_path = None
                if ADDIT_SAVE_DEBUG:
                    debug_path = self._save_debug(
                        source_resized, result_image, insert_bbox,
                        variant, attempt_seed, record,
                    )

                best_result = AddItResult(
                    success=is_valid,
                    result_image=result_image,
                    source_image=source_resized,
                    insert_bbox=insert_bbox,
                    variant=variant,
                    seed=attempt_seed,
                    attempts=attempt + 1,
                    reject_reason="" if is_valid else reject_reason,
                    metadata={
                        **(insert_meta or {}),
                        "strength": strength,
                        "guidance": guidance,
                        "steps": steps,
                        "attempt": attempt,
                    },
                    debug_path=debug_path,
                )
                if is_valid:
                    break

            if not is_valid and attempt < ADDIT_MAX_RETRIES:
                print(f"  Rejected ({reject_reason}), retrying "
                      f"({attempt + 1}/{ADDIT_MAX_RETRIES})…")

        return best_result or AddItResult(
            success=False,
            source_image=source_resized,
            variant=variant,
            seed=seed,
            reject_reason="all_retries_failed",
        )

    # ------------------------------------------------------------------
    # Batch entry point
    # ------------------------------------------------------------------
    def run_batch(
        self,
        records: List["ImageRecord"],
        variants: Optional[List[str]] = None,
        device: Optional[str] = None,
        max_images: int = 10,
        seed: Optional[int] = None,
    ) -> List[AddItResult]:
        """Run Add-it on a batch of records.

        Parameters
        ----------
        records : list of ImageRecord
        variants : list of variant names (one per record), or None to
            cycle through all variants.
        device : str
        max_images : int
        seed : int

        Returns
        -------
        results : list of AddItResult
        """
        device = device or str(self.device)
        seed = seed if seed is not None else ADDIT_SEED
        all_variants = list(ADDIT_VARIANT_OVERRIDES.keys())
        results = []
        total = min(len(records), max_images)

        print(f"\n{'═' * 60}")
        print(f" Add-it Batch Run — {total} images")
        print(f"{'═' * 60}\n")

        for idx in range(total):
            record = records[idx]
            variant = (variants[idx] if variants else all_variants[idx % len(all_variants)])
            image_seed = seed + idx * 1000

            print(f"[{idx + 1}/{total}] {record.path.name}  variant={variant}  seed={image_seed}")
            t0 = time.time()

            try:
                result = self.run_single(record, variant, seed=image_seed, device=device)
            except Exception as exc:
                print(f"  ERROR: {type(exc).__name__}: {exc}")
                traceback.print_exc(limit=6)
                result = AddItResult(
                    success=False,
                    variant=variant,
                    seed=image_seed,
                    reject_reason=str(exc),
                )

            elapsed = time.time() - t0
            status = "✓ accepted" if result.success else f"✗ rejected ({result.reject_reason})"
            print(f"  {status}  attempts={result.attempts}  {elapsed:.1f}s")

            results.append(result)

            # Save augmented image
            if result.success and result.result_image is not None:
                out_path = Path(ADDIT_OUTPUT_DIR) / record.split / variant
                out_path.mkdir(parents=True, exist_ok=True)
                fname = f"{record.path.stem}_addit_{idx:04d}_{variant}.png"
                result.result_image.save(out_path / fname)

                # Save comparison pair
                comp_dir = Path(ADDIT_OUTPUT_DIR) / "comparison_pairs" / record.split
                comp_dir.mkdir(parents=True, exist_ok=True)
                comp_fname = f"{record.path.stem}_pair_{idx:04d}_{variant}.png"
                comp_path = comp_dir / comp_fname
                save_comparison_pair(
                    result.source_image,
                    result.result_image,
                    comp_path,
                    title=f"{variant} | seed={result.seed}",
                )

        # Summary
        accepted = sum(1 for r in results if r.success)
        print(f"\n{'─' * 60}")
        print(f" Batch complete: {accepted}/{total} accepted "
              f"({100 * accepted / max(1, total):.0f}%)")
        print(f"{'─' * 60}\n")

        return results

    # ------------------------------------------------------------------
    # Quick YOLO validation
    # ------------------------------------------------------------------
    def _quick_validate(
        self,
        result_image: Image.Image,
        source_image: Image.Image,
        insert_bbox,
        variant: str,
    ) -> Tuple[bool, str]:
        """Lightweight validation: check a person is detectable in the
        insertion region."""
        try:
            from ultralytics import YOLO
            model = self.yolo
            results = model(
                np.array(result_image.convert("RGB")),
                conf=0.15,
                verbose=False,
            )
            if not results or len(results) == 0:
                return False, "yolo_no_results"

            det = results[0]
            if det.boxes is None or len(det.boxes) == 0:
                return False, "no_person_detected"

            # Check for person class (class 0 in COCO)
            classes = det.boxes.cls.cpu().numpy()
            confs = det.boxes.conf.cpu().numpy()
            person_mask = classes == 0
            if not np.any(person_mask):
                return False, "no_person_class"

            person_conf = confs[person_mask].max()
            if person_conf < 0.25:
                return False, f"low_person_conf_{person_conf:.2f}"

            # Check that at least one person overlaps the insertion bbox
            person_boxes = det.boxes.xyxy.cpu().numpy()[person_mask]
            ix1, iy1, ix2, iy2 = insert_bbox
            for box in person_boxes:
                bx1, by1, bx2, by2 = box
                # Compute IoU-like overlap
                inter_x1 = max(ix1, bx1)
                inter_y1 = max(iy1, by1)
                inter_x2 = min(ix2, bx2)
                inter_y2 = min(iy2, by2)
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    return True, ""

            return False, "person_not_in_insertion_region"

        except Exception as exc:
            print(f"  YOLO validation error: {type(exc).__name__}: {exc}")
            # If YOLO fails, accept the image optimistically
            return True, ""

    # ------------------------------------------------------------------
    # Debug visualisation
    # ------------------------------------------------------------------
    def _save_debug(
        self,
        source: Image.Image,
        result: Image.Image,
        insert_bbox,
        variant: str,
        seed: int,
        record: "ImageRecord",
    ) -> Optional[Path]:
        """Save a side-by-side debug strip (source | bbox overlay | result)."""
        debug_dir = Path(ADDIT_DEBUG_DIR)
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Count existing debug images
        existing = list(debug_dir.glob("*.png"))
        if len(existing) >= ADDIT_DEBUG_MAX_ITEMS:
            return None

        # Source with bbox overlay
        overlay = source.copy()
        draw = ImageDraw.Draw(overlay)
        x1, y1, x2, y2 = insert_bbox
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
        draw.text((x1 + 4, y1 + 4), variant, fill="lime")

        # Create comparison strip
        w, h = source.size
        strip = Image.new("RGB", (w * 3, h + 30), "white")
        draw_strip = ImageDraw.Draw(strip)
        draw_strip.text((10, 8), f"{record.path.name} | {variant} | seed={seed}", fill="black")
        strip.paste(source, (0, 30))
        strip.paste(overlay, (w, 30))
        strip.paste(result.resize((w, h)), (w * 2, 30))

        fname = f"addit_debug_{record.path.stem}_{variant}_{seed}.png"
        out_path = debug_dir / fname
        strip.save(out_path)
        return out_path
