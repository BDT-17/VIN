"""Add-it end-to-end pipeline for CityPersons pedestrian augmentation.

Wraps a loaded FLUX or SD3.5 img2img / txt2img pipeline and exposes
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

try:
    from .addit_config import (
        ADDIT_ATTENTION_LAYER_RANGE,
        ADDIT_ATTENTION_SCHEDULE,
        ADDIT_ADAPTIVE_RETRY_ENABLED,
        ADDIT_ADAPTIVE_MAX_EXTRA_STEPS,
        ADDIT_ADAPTIVE_MAX_GUIDANCE_DELTA,
        ADDIT_ADAPTIVE_MAX_STRENGTH_DELTA,
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
        ADDIT_FLUX_CPU_OFFLOAD_MODE,
        ADDIT_FLUX_MASK_BLUR_PX,
        ADDIT_FLUX_MASK_PADDING_RATIO,
        ADDIT_FLUX_MAX_SEQUENCE_LENGTH,
        ADDIT_FLUX_MODEL_ID,
        ADDIT_FLUX_TRUE_CFG_SCALE,
        ADDIT_FLUX_USE_INPAINT,
        ADDIT_GENERATOR_BACKEND,
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
        ADDIT_PERSON_CUTOUT_CONTRAST_BOOST,
        ADDIT_PERSON_CUTOUT_FALLBACK_TO_BBOX,
        ADDIT_PERSON_CUTOUT_FEATHER_PX,
        ADDIT_PERSON_CUTOUT_MASK_THRESHOLD,
        ADDIT_PERSON_CUTOUT_RECOVER_WEAK_LOWER_BODY,
        ADDIT_PERSON_CUTOUT_BBOX_EXPANSION_RATIO,
        ADDIT_PERSON_CUTOUT_SHARPNESS_BOOST,
        ADDIT_REQUIRE_PERSON_CUTOUT,
        ADDIT_MIN_PERSON_CUTOUT_AREA_RATIO,
        ADDIT_MIN_PERSON_CUTOUT_MAE_255,
        ADDIT_MIN_PERSON_CUTOUT_SHARPNESS,
        ADDIT_RETRY_SEED_STEP,
        ADDIT_SAVE_DEBUG,
        ADDIT_SEED,
        ADDIT_SOURCE_PROMPT,
        ADDIT_STRUCTURE_STRENGTH,
        ADDIT_TARGET_PROMPTS,
        ADDIT_TRANSFORMER_DEVICE,
        ADDIT_USE_TWO_GPUS,
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
        ADDIT_ADAPTIVE_RETRY_ENABLED,
        ADDIT_ADAPTIVE_MAX_EXTRA_STEPS,
        ADDIT_ADAPTIVE_MAX_GUIDANCE_DELTA,
        ADDIT_ADAPTIVE_MAX_STRENGTH_DELTA,
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
        ADDIT_FLUX_CPU_OFFLOAD_MODE,
        ADDIT_FLUX_MASK_BLUR_PX,
        ADDIT_FLUX_MASK_PADDING_RATIO,
        ADDIT_FLUX_MAX_SEQUENCE_LENGTH,
        ADDIT_FLUX_MODEL_ID,
        ADDIT_FLUX_TRUE_CFG_SCALE,
        ADDIT_FLUX_USE_INPAINT,
        ADDIT_GENERATOR_BACKEND,
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
        ADDIT_PERSON_CUTOUT_CONTRAST_BOOST,
        ADDIT_PERSON_CUTOUT_FALLBACK_TO_BBOX,
        ADDIT_PERSON_CUTOUT_FEATHER_PX,
        ADDIT_PERSON_CUTOUT_MASK_THRESHOLD,
        ADDIT_PERSON_CUTOUT_RECOVER_WEAK_LOWER_BODY,
        ADDIT_PERSON_CUTOUT_BBOX_EXPANSION_RATIO,
        ADDIT_PERSON_CUTOUT_SHARPNESS_BOOST,
        ADDIT_REQUIRE_PERSON_CUTOUT,
        ADDIT_MIN_PERSON_CUTOUT_AREA_RATIO,
        ADDIT_MIN_PERSON_CUTOUT_MAE_255,
        ADDIT_MIN_PERSON_CUTOUT_SHARPNESS,
        ADDIT_RETRY_SEED_STEP,
        ADDIT_SAVE_DEBUG,
        ADDIT_SEED,
        ADDIT_SOURCE_PROMPT,
        ADDIT_STRUCTURE_STRENGTH,
        ADDIT_TARGET_PROMPTS,
        ADDIT_TRANSFORMER_DEVICE,
        ADDIT_USE_TWO_GPUS,
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


def _flux_dtype_for_device(device) -> torch.dtype:
    if str(device).startswith("cuda"):
        try:
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        except Exception:
            return torch.float16
    return torch.float32


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
        Optional loaded ``StableDiffusion3Img2ImgPipeline`` (or txt2img) from diffusers.
        The FLUX backend lazy-loads its own pipeline and does not require this.
    yolo_model_path : str
        Path / name of the YOLOv8m-seg model for validation.
    """

    def __init__(
        self,
        sd35_pipe=None,
        yolo_model_path: str = "yolov8m-seg.pt",
        device: Optional[str] = None,
        use_transformer_split: bool = True,
    ):
        self.pipe = sd35_pipe
        self.generator_backend = str(ADDIT_GENERATOR_BACKEND).lower()
        self._flux_pipe = None
        self._flux_device = None

        if sd35_pipe is not None:
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
        else:
            self.vae = None
            self.transformer = None
            self.scheduler = None
            self.param_device = torch.device(device or TRAIN_DEVICE)
            self.transformer_device = torch.device(device or TRAIN_DEVICE)
            self.vae_device = self.transformer_device
            self.prompt_device = self.transformer_device
            self.device = self.transformer_device
            self.dtype = _flux_dtype_for_device(self.device)
        self.use_transformer_split = use_transformer_split

        # Attention state & original processors
        self.state = AddItState()
        self._original_processors = None

        # YOLO model (lazy-loaded)
        self._yolo_model_path = yolo_model_path
        self._yolo = None

        # Layer range config
        if self.transformer is not None:
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

    def _load_flux_pipeline(self, device):
        """Lazy-load the FLUX generator used by the paper-style path."""
        requested_device = torch.device(device or self.device)
        if self._flux_pipe is not None and self._flux_device == requested_device:
            return self._flux_pipe

        if ADDIT_FLUX_USE_INPAINT:
            from diffusers import FluxInpaintPipeline as flux_cls
        else:
            from diffusers import FluxImg2ImgPipeline as flux_cls

        dtype = _flux_dtype_for_device(requested_device)
        kwargs = {
            "torch_dtype": dtype,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,
        }
        try:
            pipe = flux_cls.from_pretrained(ADDIT_FLUX_MODEL_ID, **kwargs)
        except TypeError:
            kwargs.pop("low_cpu_mem_usage", None)
            pipe = flux_cls.from_pretrained(ADDIT_FLUX_MODEL_ID, **kwargs)

        offload_mode = str(ADDIT_FLUX_CPU_OFFLOAD_MODE).lower()
        if (
            str(requested_device).startswith("cuda")
            and offload_mode == "sequential"
            and hasattr(pipe, "enable_sequential_cpu_offload")
        ):
            pipe.enable_sequential_cpu_offload(gpu_id=requested_device.index or 0)
            print(f"Enabled FLUX sequential CPU offload for {requested_device}")
        elif (
            str(requested_device).startswith("cuda")
            and offload_mode != "none"
            and hasattr(pipe, "enable_model_cpu_offload")
        ):
            pipe.enable_model_cpu_offload(gpu_id=requested_device.index or 0)
            print(f"Enabled FLUX model CPU offload for {requested_device}")
        else:
            pipe.to(requested_device)
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()

        self._flux_pipe = pipe
        self._flux_device = requested_device
        return pipe

    def _unload_flux_pipeline(self):
        if self._flux_pipe is not None:
            del self._flux_pipe
            self._flux_pipe = None
            self._flux_device = None
        gc.collect()
        clear_cuda()

    @staticmethod
    def _expand_bbox(bbox, image_size, expansion: float):
        width, height = image_size
        x1, y1, x2, y2 = [float(v) for v in bbox]
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        bw = max(1.0, (x2 - x1) * float(expansion))
        bh = max(1.0, (y2 - y1) * float(expansion))
        return (
            max(0, int(round(cx - bw * 0.5))),
            max(0, int(round(cy - bh * 0.5))),
            min(width, int(round(cx + bw * 0.5))),
            min(height, int(round(cy + bh * 0.5))),
        )

    @staticmethod
    def _make_flux_inpaint_mask(image_size, insert_bbox):
        width, height = image_size
        expansion = 1.0 + max(0.0, float(ADDIT_FLUX_MASK_PADDING_RATIO)) * 2.0
        x1, y1, x2, y2 = AddItCityPersonsPipeline._expand_bbox(
            insert_bbox, image_size, expansion
        )
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle((x1, y1, x2, y2), fill=255)
        if ADDIT_FLUX_MASK_BLUR_PX > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=ADDIT_FLUX_MASK_BLUR_PX))
        return mask

    def _flux_generate(
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
        """Generate a candidate with FLUX, then downstream segmentation keeps only the person."""
        flux_pipe = self._load_flux_pipeline(device)
        flux_device = self._flux_device or torch.device(device)
        generator_device = flux_device if str(flux_device).startswith("cuda") else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        prompt = (
            f"{target_prompt}, photorealistic CityPersons-style street camera image, "
            "new pedestrian fully inside the masked insertion area, natural scale, grounded feet"
        )
        kwargs = {
            "prompt": prompt,
            "image": source_image,
            "strength": strength,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
            "height": source_image.height,
            "width": source_image.width,
            "max_sequence_length": int(ADDIT_FLUX_MAX_SEQUENCE_LENGTH),
        }
        if ADDIT_FLUX_TRUE_CFG_SCALE and ADDIT_FLUX_TRUE_CFG_SCALE > 1.0:
            kwargs["negative_prompt"] = negative_prompt
            kwargs["true_cfg_scale"] = ADDIT_FLUX_TRUE_CFG_SCALE
        if ADDIT_FLUX_USE_INPAINT:
            kwargs["mask_image"] = self._make_flux_inpaint_mask(source_image.size, insert_bbox)

        result = flux_pipe(**kwargs)
        return result.images[0].resize(source_image.size)

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
        w_source_range: Optional[Tuple[float, float]] = None,
        w_self_range: Optional[Tuple[float, float]] = None,
    ) -> Image.Image:
        """Core Add-it denoising loop implementing all three mechanisms.

        Returns the generated image as a PIL Image.
        """
        base_device = torch.device(device)
        generation_device = base_device
        if (
            self.use_transformer_split
            and ADDIT_USE_TWO_GPUS
            and torch.cuda.is_available()
            and torch.cuda.device_count() >= 2
            and str(base_device).startswith("cuda")
        ):
            generation_device = torch.device(ADDIT_TRANSFORMER_DEVICE)

        device = generation_device
        vae_device = self.vae_device
        dtype = self.dtype
        resolution = RESOLUTION
        generator = torch.Generator(device=device if str(device).startswith("cuda") else "cpu")
        generator.manual_seed(seed)
        w_source_range = w_source_range or (ADDIT_W_SOURCE_START, ADDIT_W_SOURCE_END)
        w_self_range = w_self_range or (ADDIT_W_SELF_START, ADDIT_W_SELF_END)

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
        original_transformer_device = _module_device(self.transformer, fallback=str(base_device))
        moved_transformer_for_generation = original_transformer_device != generation_device
        if moved_transformer_for_generation:
            self.transformer.to(generation_device)
            self.transformer_device = generation_device
            self.device = generation_device
            if str(original_transformer_device).startswith("cuda"):
                torch.cuda.empty_cache()

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
                    w_source_range=w_source_range,
                    w_self_range=w_self_range,
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
            if moved_transformer_for_generation:
                self.transformer.to(original_transformer_device)
                self.transformer_device = original_transformer_device
                self.device = original_transformer_device
                if str(generation_device).startswith("cuda"):
                    torch.cuda.empty_cache()

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

    @staticmethod
    def _recover_weak_lower_body_mask(mask_np, soft_mask_np, box, threshold):
        """Keep weak YOLO mask pixels around legs/feet without pasting full bbox."""
        height, width = mask_np.shape
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1 = max(0, min(width - 1, x1))
        x2 = max(x1 + 1, min(width, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(y1 + 1, min(height, y2))
        lower_y = y1 + int((y2 - y1) * 0.55)
        weak_threshold = max(12, int(threshold * 0.55))
        lower_soft = soft_mask_np[lower_y:y2, x1:x2]
        lower_keep = lower_soft >= weak_threshold
        if np.any(lower_keep):
            mask_np[lower_y:y2, x1:x2] = np.maximum(
                mask_np[lower_y:y2, x1:x2],
                lower_keep.astype(np.uint8) * 255,
            )
        return mask_np

    @staticmethod
    def _mask_area_ratio(mask: Image.Image, threshold: int = 24) -> float:
        mask_arr = np.asarray(mask.convert("L"), dtype=np.uint8)
        return float(np.mean(mask_arr > threshold))

    @staticmethod
    def _masked_mae_255(image_a: Image.Image, image_b: Image.Image, mask: Image.Image, threshold: int = 24) -> float:
        arr_a = np.asarray(image_a.convert("RGB"), dtype=np.float32)
        arr_b = np.asarray(image_b.convert("RGB"), dtype=np.float32)
        mask_arr = np.asarray(mask.convert("L"), dtype=np.uint8) > threshold
        if not np.any(mask_arr):
            return 0.0
        return float(np.mean(np.abs(arr_a[mask_arr] - arr_b[mask_arr])))

    @staticmethod
    def _masked_gradient_sharpness(image: Image.Image, mask: Image.Image, threshold: int = 24) -> float:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        luma = arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722
        grad_y, grad_x = np.gradient(luma)
        grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
        mask_arr = np.asarray(mask.convert("L"), dtype=np.uint8) > threshold
        if not np.any(mask_arr):
            return 0.0
        return float(np.std(grad_mag[mask_arr]))

    @staticmethod
    def _enhance_generated_person_for_paste(generated: Image.Image, mask: Image.Image) -> Image.Image:
        enhanced = generated.convert("RGB")
        if ADDIT_PERSON_CUTOUT_CONTRAST_BOOST != 1.0:
            enhanced = ImageEnhance.Contrast(enhanced).enhance(ADDIT_PERSON_CUTOUT_CONTRAST_BOOST)
        if ADDIT_PERSON_CUTOUT_SHARPNESS_BOOST != 1.0:
            enhanced = ImageEnhance.Sharpness(enhanced).enhance(ADDIT_PERSON_CUTOUT_SHARPNESS_BOOST)
        return Image.composite(enhanced, generated.convert("RGB"), mask)

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
        threshold = int(ADDIT_PERSON_CUTOUT_MASK_THRESHOLD)
        selected_soft_masks = []
        for idx in selected_indices:
            person_mask = np.clip(masks[idx] * 255.0, 0, 255).astype(np.uint8)
            person_mask_img = Image.fromarray(person_mask, mode="L").resize(
                (width, height), Image.Resampling.BILINEAR
            )
            soft_mask_np = np.array(person_mask_img, dtype=np.uint8)
            selected_soft_masks.append((soft_mask_np, boxes[idx]))
            mask_np = np.maximum(
                mask_np,
                np.where(soft_mask_np >= threshold, 255, 0).astype(np.uint8),
            )

        if ADDIT_PERSON_CUTOUT_RECOVER_WEAK_LOWER_BODY:
            for soft_mask_np, box in selected_soft_masks:
                mask_np = self._recover_weak_lower_body_mask(
                    mask_np, soft_mask_np, box, threshold
                )

        if not np.any(mask_np):
            return None

        mask = Image.fromarray(mask_np, mode="L")
        clip = Image.new("L", (width, height), 0)
        clip_bbox = self._expand_bbox(
            insert_bbox,
            (width, height),
            ADDIT_PERSON_CUTOUT_BBOX_EXPANSION_RATIO,
        )
        ImageDraw.Draw(clip).rectangle(clip_bbox, fill=255)
        mask = Image.composite(mask, Image.new("L", (width, height), 0), clip)
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
    ) -> Tuple[Image.Image, Dict]:
        """Cut the generated person from Add-it output and paste onto source."""
        source = source_image.convert("RGB")
        generated = generated_image.convert("RGB").resize(source.size, Image.LANCZOS)
        meta = {
            "person_cutout_mask_found": False,
            "person_cutout_fallback_used": False,
            "person_cutout_area_ratio": 0.0,
            "person_cutout_mae_255": 0.0,
            "person_cutout_sharpness": 0.0,
        }
        try:
            mask = self._extract_generated_person_mask(generated, insert_bbox)
        except Exception as exc:
            print(f"  Person cutout failed: {type(exc).__name__}: {exc}")
            mask = None

        if mask is not None:
            generated_for_paste = self._enhance_generated_person_for_paste(generated, mask)
            meta.update({
                "person_cutout_mask_found": True,
                "person_cutout_area_ratio": self._mask_area_ratio(mask),
                "person_cutout_mae_255": self._masked_mae_255(generated_for_paste, source, mask),
                "person_cutout_sharpness": self._masked_gradient_sharpness(generated_for_paste, mask),
                "person_cutout_contrast_boost": ADDIT_PERSON_CUTOUT_CONTRAST_BOOST,
                "person_cutout_sharpness_boost": ADDIT_PERSON_CUTOUT_SHARPNESS_BOOST,
            })
            return Image.composite(generated_for_paste, source, mask), meta

        if ADDIT_PERSON_CUTOUT_FALLBACK_TO_BBOX:
            meta["person_cutout_fallback_used"] = True
            return composite_generated_region(
                source_image=source,
                generated_image=generated,
                bbox=insert_bbox,
                expansion=ADDIT_MASK_EXPANSION_RATIO,
                feather=ADDIT_FINAL_COMPOSITE_FEATHER_PX,
                mode=ADDIT_FINAL_COMPOSITE_MODE,
            ), meta

        return source, meta

    @staticmethod
    def _validate_person_cutout_meta(meta: Dict) -> Tuple[bool, str]:
        if ADDIT_REQUIRE_PERSON_CUTOUT and not meta.get("person_cutout_mask_found", False):
            return False, "no_generated_person_cutout"
        area_ratio = float(meta.get("person_cutout_area_ratio", 0.0))
        if area_ratio < ADDIT_MIN_PERSON_CUTOUT_AREA_RATIO:
            return False, f"generated_person_mask_too_small_{area_ratio:.5f}"
        mae_255 = float(meta.get("person_cutout_mae_255", 0.0))
        if mae_255 < ADDIT_MIN_PERSON_CUTOUT_MAE_255:
            return False, f"no_visible_new_person_change_{mae_255:.2f}"
        sharpness = float(meta.get("person_cutout_sharpness", 0.0))
        if sharpness < ADDIT_MIN_PERSON_CUTOUT_SHARPNESS:
            return False, f"blurry_generated_person_{sharpness:.2f}"
        return True, ""

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    @classmethod
    def _build_adaptive_retry_config(
        cls,
        base_prompt: str,
        base_negative_prompt: str,
        base_strength: float,
        base_guidance: float,
        base_steps: int,
        reject_reason: Optional[str],
        attempt: int,
    ) -> Dict:
        """Adjust Add-it generation inputs for the next attempt."""
        config = {
            "prompt": base_prompt,
            "negative_prompt": base_negative_prompt,
            "strength": base_strength,
            "guidance": base_guidance,
            "steps": base_steps,
            "w_source_range": (ADDIT_W_SOURCE_START, ADDIT_W_SOURCE_END),
            "w_self_range": (ADDIT_W_SELF_START, ADDIT_W_SELF_END),
            "adaptive_reason": reject_reason or "",
            "adaptive_note": "base",
        }
        if not ADDIT_ADAPTIVE_RETRY_ENABLED or not reject_reason or attempt <= 0:
            return config

        reason = str(reject_reason)
        strength_delta = min(ADDIT_ADAPTIVE_MAX_STRENGTH_DELTA, 0.045 * attempt)
        guidance_delta = min(ADDIT_ADAPTIVE_MAX_GUIDANCE_DELTA, 0.38 * attempt)
        extra_steps = min(ADDIT_ADAPTIVE_MAX_EXTRA_STEPS, 3 * attempt)
        prompt_boost = ""
        negative_boost = ""
        w_source_start = ADDIT_W_SOURCE_START
        w_source_end = ADDIT_W_SOURCE_END
        w_self_start = ADDIT_W_SELF_START
        w_self_end = ADDIT_W_SELF_END
        note = "seed_only"

        if (
            "no_generated_person_cutout" in reason
            or "no_person_detected" in reason
            or "no_person_class" in reason
            or "yolo_no_results" in reason
        ):
            prompt_boost = ", prominent newly added full-body pedestrian, high contrast human figure, unmistakably visible, not part of the original image"
            negative_boost = ", empty insertion area, no added person, invisible person, source image copy"
            config["strength"] = base_strength + strength_delta
            config["guidance"] = base_guidance + guidance_delta
            config["steps"] = base_steps + extra_steps
            w_source_start -= 0.20
            w_source_end -= 0.03
            w_self_start += 0.12
            w_self_end += 0.06
            note = "force_visible_person"
        elif "no_visible_new_person_change" in reason:
            prompt_boost = ", obvious new pedestrian added into the scene, strong foreground silhouette, visible clothing, legs and feet, high local contrast"
            negative_boost = ", unchanged image, no edit, invisible person, ghost, source-only reconstruction"
            config["strength"] = base_strength + min(ADDIT_ADAPTIVE_MAX_STRENGTH_DELTA, strength_delta + 0.04)
            config["guidance"] = base_guidance + guidance_delta
            config["steps"] = base_steps + extra_steps
            w_source_start -= 0.24
            w_source_end -= 0.04
            w_self_start += 0.14
            w_self_end += 0.08
            note = "increase_edit_visibility"
        elif "blurry_generated_person" in reason:
            prompt_boost = ", sharp detailed full-body pedestrian, crisp legs and feet, clear silhouette, high contrast edges"
            negative_boost = ", blurry person, soft legs, smeared feet, low detail, faded body, blended into background"
            config["strength"] = base_strength + min(ADDIT_ADAPTIVE_MAX_STRENGTH_DELTA, 0.02 * attempt)
            config["guidance"] = base_guidance + min(ADDIT_ADAPTIVE_MAX_GUIDANCE_DELTA, 0.28 * attempt)
            config["steps"] = base_steps + extra_steps
            w_source_start -= 0.12
            w_source_end -= 0.02
            w_self_start += 0.08
            note = "sharpen_person"
        elif "generated_person_mask_too_small" in reason or "low_person_conf" in reason:
            prompt_boost = ", larger clear full-body pedestrian, detectable human shape, visible head torso legs and feet, fills the planned insertion area"
            negative_boost = ", tiny person, partial person, cropped body, weak silhouette"
            config["strength"] = base_strength + strength_delta
            config["guidance"] = base_guidance + guidance_delta
            config["steps"] = base_steps + extra_steps
            w_source_start -= 0.18
            w_source_end -= 0.03
            w_self_start += 0.10
            note = "grow_detectable_person"
        elif "person_not_in_insertion_region" in reason:
            prompt_boost = ", pedestrian exactly inside the marked insertion area, grounded feet on road or sidewalk"
            negative_boost = ", person outside target area, misplaced pedestrian, floating person"
            config["guidance"] = base_guidance + min(ADDIT_ADAPTIVE_MAX_GUIDANCE_DELTA, 0.18 * attempt)
            config["steps"] = base_steps + min(ADDIT_ADAPTIVE_MAX_EXTRA_STEPS, attempt)
            w_source_start += 0.04
            w_self_start -= 0.02
            note = "improve_placement"
        elif "yolo_validation_error" in reason:
            config["steps"] = base_steps + min(ADDIT_ADAPTIVE_MAX_EXTRA_STEPS, attempt)
            note = "stabilize_validation"

        config["prompt"] = base_prompt + prompt_boost
        config["negative_prompt"] = base_negative_prompt + negative_boost
        config["strength"] = cls._clamp(
            float(config["strength"]),
            max(0.20, base_strength - ADDIT_ADAPTIVE_MAX_STRENGTH_DELTA),
            min(0.92, base_strength + ADDIT_ADAPTIVE_MAX_STRENGTH_DELTA),
        )
        config["guidance"] = cls._clamp(
            float(config["guidance"]),
            max(1.0, base_guidance - 0.20),
            base_guidance + ADDIT_ADAPTIVE_MAX_GUIDANCE_DELTA,
        )
        config["steps"] = int(max(1, min(base_steps + ADDIT_ADAPTIVE_MAX_EXTRA_STEPS, config["steps"])))
        config["w_source_range"] = (
            cls._clamp(w_source_start, 0.02, 0.90),
            cls._clamp(w_source_end, 0.005, 0.35),
        )
        config["w_self_range"] = (
            cls._clamp(w_self_start, 0.05, 0.90),
            cls._clamp(w_self_end, 0.20, 0.95),
        )
        config["adaptive_note"] = note
        return config

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
        fallback_device = self.vae_device
        original_transformer_device = _module_device(
            self.transformer, fallback=str(fallback_device)
        )
        moved_transformer = original_transformer_device != fallback_device

        if moved_transformer:
            self.transformer.to(fallback_device)
            if str(original_transformer_device).startswith("cuda"):
                torch.cuda.empty_cache()

        try:
            generator_device = fallback_device if str(fallback_device).startswith("cuda") else "cpu"
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
        finally:
            if moved_transformer:
                self.transformer.to(original_transformer_device)
                if str(fallback_device).startswith("cuda"):
                    torch.cuda.empty_cache()

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
        base_strength = overrides.get("strength", ADDIT_STRUCTURE_STRENGTH)
        base_guidance = overrides.get("guidance", ADDIT_GUIDANCE_SCALE)
        base_steps = overrides.get("steps", ADDIT_NUM_INFERENCE_STEPS)

        base_target_prompt = ADDIT_TARGET_PROMPTS.get(
            variant,
            VARIANT_PROMPTS.get(variant, "urban street photo with a pedestrian"),
        )
        base_negative_prompt = ADDIT_NEGATIVE_PROMPT

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
        last_reject_reason = None
        for attempt in range(ADDIT_MAX_RETRIES + 1):
            attempt_seed = seed + attempt * ADDIT_RETRY_SEED_STEP
            attempt_config = self._build_adaptive_retry_config(
                base_target_prompt,
                base_negative_prompt,
                base_strength,
                base_guidance,
                base_steps,
                last_reject_reason,
                attempt,
            )
            if attempt > 0:
                print(
                    "  Adaptive Add-it retry "
                    f"{attempt}: reason={last_reject_reason}, "
                    f"mode={attempt_config['adaptive_note']}, "
                    f"strength={attempt_config['strength']:.2f}, "
                    f"guidance={attempt_config['guidance']:.2f}, "
                    f"steps={attempt_config['steps']}, "
                    f"w_source={attempt_config['w_source_range']}, "
                    f"w_self={attempt_config['w_self_range']}"
                )
            try:
                if self.generator_backend == "flux":
                    result_image = self._flux_generate(
                        source_image=source_resized,
                        insert_bbox=insert_bbox,
                        target_prompt=attempt_config["prompt"],
                        negative_prompt=attempt_config["negative_prompt"],
                        strength=attempt_config["strength"],
                        guidance_scale=attempt_config["guidance"],
                        num_inference_steps=attempt_config["steps"],
                        seed=attempt_seed,
                        device=device,
                    )
                elif self.generator_backend == "sd35":
                    result_image = self._addit_denoise(
                        source_image=source_resized,
                        insert_bbox=insert_bbox,
                        target_prompt=attempt_config["prompt"],
                        negative_prompt=attempt_config["negative_prompt"],
                        strength=attempt_config["strength"],
                        guidance_scale=attempt_config["guidance"],
                        num_inference_steps=attempt_config["steps"],
                        seed=attempt_seed,
                        device=device,
                        w_source_range=attempt_config["w_source_range"],
                        w_self_range=attempt_config["w_self_range"],
                    )
                else:
                    raise ValueError(f"Unsupported Add-it generator backend: {self.generator_backend}")
            except Exception as exc:
                print(f"  Add-it generation failed (attempt {attempt + 1}): "
                      f"{type(exc).__name__}: {exc}")
                traceback.print_exc(limit=6)
                last_reject_reason = f"addit_generation_error_{type(exc).__name__}"
                if isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower():
                    print("  CUDA OOM during Add-it generation; unloading FLUX pipeline before retry.")
                    self._unload_flux_pipeline()
                if not ADDIT_FALLBACK_TO_NATIVE_IMG2IMG:
                    continue
                print("  Falling back to native SD3 img2img for this attempt.")
                try:
                    result_image = self._native_img2img_fallback(
                        source_image=source_resized,
                        target_prompt=attempt_config["prompt"],
                        negative_prompt=attempt_config["negative_prompt"],
                        strength=attempt_config["strength"],
                        guidance_scale=attempt_config["guidance"],
                        num_inference_steps=attempt_config["steps"],
                        seed=attempt_seed,
                        device=device,
                    )
                except Exception as fallback_exc:
                    print(f"  Native fallback failed: {type(fallback_exc).__name__}: {fallback_exc}")
                    traceback.print_exc(limit=6)
                    last_reject_reason = f"native_fallback_error_{type(fallback_exc).__name__}"
                    continue

            paste_meta = {
                "person_cutout_mask_found": not ADDIT_FINAL_PERSON_CUTOUT,
            }
            if ADDIT_FINAL_PERSON_CUTOUT:
                result_image, paste_meta = self._paste_generated_person_on_source(
                    source_image=source_resized,
                    generated_image=result_image,
                    insert_bbox=insert_bbox,
                )

            is_valid, reject_reason = True, ""
            if ADDIT_FINAL_PERSON_CUTOUT:
                is_valid, reject_reason = self._validate_person_cutout_meta(paste_meta)

            # Validate with YOLO only after proving a visible generated cutout exists.
            if is_valid:
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
                        **paste_meta,
                        "generator_backend": self.generator_backend,
                        "flux_model_id": ADDIT_FLUX_MODEL_ID if self.generator_backend == "flux" else "",
                        "flux_use_inpaint": bool(ADDIT_FLUX_USE_INPAINT) if self.generator_backend == "flux" else False,
                        "strength": attempt_config["strength"],
                        "guidance": attempt_config["guidance"],
                        "steps": attempt_config["steps"],
                        "attempt": attempt,
                        "adaptive_reason": attempt_config["adaptive_reason"],
                        "adaptive_note": attempt_config["adaptive_note"],
                        "adaptive_prompt": attempt_config["prompt"],
                        "adaptive_negative_prompt": attempt_config["negative_prompt"],
                        "adaptive_w_source_range": attempt_config["w_source_range"],
                        "adaptive_w_self_range": attempt_config["w_self_range"],
                    },
                    debug_path=debug_path,
                )
                if is_valid:
                    break

            if not is_valid and attempt < ADDIT_MAX_RETRIES:
                last_reject_reason = reject_reason
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
            return False, f"yolo_validation_error_{type(exc).__name__}"

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


def _save_successful_addit_result(result, record, variant, index):
    if not result.success or result.result_image is None:
        return
    out_path = Path(ADDIT_OUTPUT_DIR) / record.split / variant
    out_path.mkdir(parents=True, exist_ok=True)
    fname = f"{record.path.stem}_addit_{index:04d}_{variant}.png"
    result.result_image.save(out_path / fname)

    comp_dir = Path(ADDIT_OUTPUT_DIR) / "comparison_pairs" / record.split
    comp_dir.mkdir(parents=True, exist_ok=True)
    comp_fname = f"{record.path.stem}_pair_{index:04d}_{variant}.png"
    comp_path = comp_dir / comp_fname
    save_comparison_pair(
        result.source_image,
        result.result_image,
        comp_path,
        title=f"{variant} | seed={result.seed}",
    )


def _run_addit_shard_on_device(
    device,
    jobs,
    total_jobs,
    seed,
    pipeline=None,
):
    owns_pipeline = pipeline is None
    if str(device).startswith("cuda"):
        torch.cuda.set_device(torch.device(device).index or 0)
    if pipeline is None:
        print(f"[{device}] loading Add-it pipeline for {len(jobs)} jobs")
        pipe = None
        if str(ADDIT_GENERATOR_BACKEND).lower() == "sd35":
            from sd35_model import build_img2img_pipeline

            pipe = build_img2img_pipeline(device=device)
        pipeline = AddItCityPersonsPipeline(
            pipe,
            device=device,
            use_transformer_split=False,
        )
    else:
        print(f"[{device}] using existing Add-it pipeline for {len(jobs)} jobs")
        pipeline.use_transformer_split = False

    results = []
    try:
        for index, record, variant in jobs:
            image_seed = seed + index * 1000
            print(
                f"[{device}] [{index + 1}/{total_jobs}] "
                f"{record.path.name}  variant={variant}  seed={image_seed}"
            )
            t0 = time.time()
            try:
                result = pipeline.run_single(
                    record,
                    variant,
                    seed=image_seed,
                    device=device,
                )
            except Exception as exc:
                print(f"[{device}] ERROR: {type(exc).__name__}: {exc}")
                traceback.print_exc(limit=6)
                result = AddItResult(
                    success=False,
                    variant=variant,
                    seed=image_seed,
                    reject_reason=str(exc),
                )

            elapsed = time.time() - t0
            status = "accepted" if result.success else f"rejected ({result.reject_reason})"
            print(f"[{device}] {status}  attempts={result.attempts}  {elapsed:.1f}s")
            _save_successful_addit_result(result, record, variant, index)
            results.append((index, result))
    finally:
        if owns_pipeline:
            try:
                del pipeline
            except UnboundLocalError:
                pass
            gc.collect()
            clear_cuda()

    return results


def run_addit_batch_multi_gpu(
    records: List["ImageRecord"],
    variants: Optional[List[str]] = None,
    devices: Optional[List[str]] = None,
    max_images: int = 10,
    seed: Optional[int] = None,
    existing_pipeline: Optional[AddItCityPersonsPipeline] = None,
    existing_device: Optional[str] = None,
) -> List[AddItResult]:
    """Run Add-it data-parallel: each GPU gets its own image shard."""
    seed = seed if seed is not None else ADDIT_SEED
    total = min(len(records), max_images)
    all_variants = list(ADDIT_VARIANT_OVERRIDES.keys())
    jobs = [
        (
            index,
            records[index],
            variants[index] if variants else all_variants[index % len(all_variants)],
        )
        for index in range(total)
    ]

    if devices is None:
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            devices = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
        else:
            devices = [TRAIN_DEVICE if torch.cuda.is_available() else "cpu"]

    devices = [str(device) for device in devices]
    active_shards = [
        (device, jobs[index::len(devices)])
        for index, device in enumerate(devices)
        if jobs[index::len(devices)]
    ]

    print(f"\n{'=' * 60}")
    print(f" Add-it Multi-GPU Batch Run - {total} images")
    print(" Shards:", ", ".join(f"{device}:{len(shard)}" for device, shard in active_shards))
    print(f"{'=' * 60}\n")

    if len(active_shards) == 1:
        device, shard = active_shards[0]
        pipeline = existing_pipeline if existing_device in {None, device} else None
        shard_results = _run_addit_shard_on_device(device, shard, total, seed, pipeline=pipeline)
    else:
        shard_results = []
        with ThreadPoolExecutor(max_workers=len(active_shards)) as executor:
            futures = {}
            for device, shard in active_shards:
                pipeline = existing_pipeline if existing_device in {None, device} else None
                future = executor.submit(
                    _run_addit_shard_on_device,
                    device,
                    shard,
                    total,
                    seed,
                    pipeline,
                )
                futures[future] = device
                if pipeline is existing_pipeline:
                    existing_pipeline = None
            for future in as_completed(futures):
                device = futures[future]
                try:
                    shard_results.extend(future.result())
                except Exception as exc:
                    print(f"[{device}] shard failed: {type(exc).__name__}: {exc}")
                    traceback.print_exc(limit=6)

    results_by_index = {index: result for index, result in shard_results}
    results = [
        results_by_index.get(
            index,
            AddItResult(
                success=False,
                variant=jobs[index][2],
                seed=seed + index * 1000,
                reject_reason="missing_shard_result",
            ),
        )
        for index in range(total)
    ]
    accepted = sum(1 for result in results if result.success)
    print(f"\n{'-' * 60}")
    print(f" Multi-GPU batch complete: {accepted}/{total} accepted ({100 * accepted / max(1, total):.0f}%)")
    print(f"{'-' * 60}\n")
    return results
