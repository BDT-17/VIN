"""Standalone Pokecut-style AI Replace orchestration.

The implementation is deliberately independent from the V5 pipeline. It can
use a diffusers inpainting pipeline when available, but the hard background
restore and validation are local and testable without a model.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import json
import os
import numpy as np
from PIL import Image, ImageFilter

from .config import AIReplaceConfig, DEFAULT_CONFIG
from .sd35_mask_refinement import (
    AIReplaceMaskBundle,
    bbox_to_mask,
    hard_restore_outside_mask,
    outside_mask_diff,
    refine_mask,
)
from .sd35_harmonization import decompose_object_mask, harmonize_object_with_background
from .sd35_ghost_detection import GhostDetectionResult, detect_ghost


@dataclass(frozen=True)
class ObjectMaskResult:
    object_mask: Optional[np.ndarray] = None
    object_bbox: Optional[Tuple[int, int, int, int]] = None
    object_mask_area: int = 0
    object_mask_inside_ratio: float = 0.0
    object_bbox_inside_ratio: float = 0.0
    detector_confidence: float = 0.0
    reject_reason: Optional[str] = None


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    checks: dict
    reject_reason: Optional[str]
    outside_mask_diff: float


@dataclass(frozen=True)
class AIReplaceResult:
    raw_image: Image.Image
    harmonized_image: Image.Image
    mask_bundle: AIReplaceMaskBundle
    object_result: ObjectMaskResult
    ghost_result: GhostDetectionResult
    validation: ValidationResult
    manifest_row: dict


def _bbox_area(bbox):
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_intersection(a, b):
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    return max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))


def _mask_inside_ratio(object_mask: np.ndarray, hard_mask: np.ndarray) -> float:
    active = object_mask > 0.5
    denom = max(1, int(active.sum()))
    return float(np.logical_and(active, hard_mask > 0.5).sum() / denom)


def _bbox_inside_ratio(bbox, container_bbox) -> float:
    return float(_bbox_intersection(bbox, container_bbox) / max(1.0, _bbox_area(bbox)))


def _bbox_touches_image_margin(bbox, image_size, margin_ratio: float = 0.02) -> bool:
    if bbox is None:
        return True
    width, height = image_size
    margin = int(round(max(width, height) * margin_ratio))
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    return x1 <= margin or y1 <= margin or x2 >= width - margin or y2 >= height - margin


def _bbox_height_ratio(bbox, image_height: int) -> float:
    if bbox is None:
        return 0.0
    return max(0.0, float(bbox[3] - bbox[1]) / max(1.0, float(image_height)))


def _mask_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))

def _scale_bbox_to_size(bbox, source_size, target_size):
    src_w, src_h = source_size
    dst_w, dst_h = target_size
    if src_w <= 0 or src_h <= 0:
        return tuple(int(round(v)) for v in bbox)
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return (
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        int(round(x2 * sx)),
        int(round(y2 * sy)),
    )



class AIReplacePipeline:
    def __init__(self, config: AIReplaceConfig = DEFAULT_CONFIG, pipe=None, device: str = "cuda"):
        self.config = config
        self.pipe = pipe
        self.device = device
        self.last_conditioning_meta = {}

    @classmethod
    def from_pretrained(cls, config: AIReplaceConfig = DEFAULT_CONFIG, device: str = "cuda"):
        try:
            import torch
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("torch is required to load the AI Replace model") from exc

        model_type = str(config.AI_REPLACE_MODEL_TYPE).lower()
        if model_type == "sd35_inpainting":
            try:
                from diffusers import StableDiffusion3InpaintPipeline as PipelineClass
            except Exception as exc:  # pragma: no cover - depends on diffusers version
                raise RuntimeError(
                    "StableDiffusion3InpaintPipeline could not be imported for SD3.5 inpainting. "
                    "On Kaggle this is often a transformers/huggingface_hub version mismatch. "
                    "Run the notebook install cell with pinned versions, restart the kernel, then rerun from Imports: "
                    "pip install --force-reinstall diffusers==0.35.2 transformers==4.46.3 huggingface_hub==0.36.0 accelerate==1.11.0 safetensors"
                ) from exc
        else:
            try:
                from diffusers import StableDiffusionInpaintPipeline as PipelineClass
            except Exception as exc:  # pragma: no cover - environment dependent
                raise RuntimeError("diffusers is required to load the AI Replace model") from exc

        dtype = torch.float16 if config.TORCH_DTYPE == "float16" else torch.float32
        legacy_model_ids = {
            "stabilityai/stable-diffusion-2-inpainting": "sd2-community/stable-diffusion-2-inpainting",
        }

        def normalize_model_id(model_id) -> str:
            value = str(model_id or "").strip()
            return legacy_model_ids.get(value.lower(), value)

        requested_model = normalize_model_id(os.getenv("AI_REPLACE_MODEL_ID", config.MODEL_ID))
        fallback_models = getattr(config, "MODEL_ID_FALLBACKS", ())
        model_ids = []
        for model_id in (requested_model, *fallback_models):
            normalized_model_id = normalize_model_id(model_id)
            if normalized_model_id and normalized_model_id not in model_ids:
                model_ids.append(normalized_model_id)

        errors = []
        for model_id in model_ids:
            kwargs = {"torch_dtype": dtype, "use_safetensors": True}
            if model_type == "sd35_inpainting" and not getattr(config, "USE_T5", False):
                kwargs.update({"text_encoder_3": None, "tokenizer_3": None})
            try:
                print(f"Loading AI Replace {model_type} model: {model_id}")
                try:
                    pipe = PipelineClass.from_pretrained(model_id, **kwargs)
                except TypeError:
                    kwargs.pop("text_encoder_3", None)
                    kwargs.pop("tokenizer_3", None)
                    pipe = PipelineClass.from_pretrained(model_id, **kwargs)
                break
            except Exception as exc:  # pragma: no cover - depends on Hub/cache state
                errors.append(f"{model_id}: {type(exc).__name__}: {exc}")
        else:
            joined = "\n".join(f"- {error}" for error in errors)
            raise RuntimeError(
                "Unable to load an AI Replace inpainting model. "
                "For SD3.5, make sure the Hugging Face token has access to stabilityai/stable-diffusion-3.5-medium. "
                f"Tried:\n{joined}"
            )

        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        if str(device).startswith("cuda") and getattr(config, "USE_MODEL_CPU_OFFLOAD", False) and hasattr(pipe, "enable_model_cpu_offload"):
            gpu_id = torch.device(device).index or 0
            pipe.enable_model_cpu_offload(gpu_id=gpu_id)
        else:
            pipe = pipe.to(device)
        return cls(config=config, pipe=pipe, device=device)

    @staticmethod
    def _hard_clamp_background(z_t, z_origin, mask_latent):
        return z_origin * (1 - mask_latent) + z_t * mask_latent

    def build_mask_bundle(self, image: Image.Image, bbox: tuple) -> AIReplaceMaskBundle:
        raw = bbox_to_mask((image.height, image.width), bbox)
        return refine_mask(raw, bbox, self.config)

    def _classify_scene(self, image: Image.Image, bbox: tuple) -> str:
        _x1, y1, _x2, y2 = [int(round(v)) for v in bbox]
        y_center = ((y1 + y2) / 2.0) / max(1, image.height)
        return "urban street scene" if y_center > 0.58 else "outdoor scene"

    def _classify_lighting(self, image: Image.Image, bbox: tuple) -> str:
        crop = image.crop(tuple(int(round(v)) for v in bbox)).convert("L")
        arr = np.asarray(crop, dtype=np.float32)
        mean = float(arr.mean()) if arr.size else 128.0
        if mean < 75:
            return "low light"
        if mean > 180:
            return "bright daylight"
        return "natural lighting"

    def _build_dynamic_prompt(self, image: Image.Image, bbox: tuple, base_prompt: str) -> str:
        if not self.config.AI_REPLACE_DYNAMIC_PROMPT:
            return base_prompt
        subject = base_prompt
        scene = self._classify_scene(image, bbox)
        style = "realistic street photo"
        lighting = self._classify_lighting(image, bbox) if self.config.AI_REPLACE_LIGHTING_CONDITIONING else "natural lighting"
        camera = "matching camera perspective"
        if not self.config.AI_REPLACE_GOLDEN_FORMULA:
            return f"{subject}, {lighting}, {camera}"
        return self.config.AI_REPLACE_PROMPT_TEMPLATE.format(
            subject=subject,
            scene=scene,
            style=style,
            lighting=lighting,
            camera=camera,
        )

    def _compute_depth_conditioning(self, image: Image.Image) -> Image.Image:
        """Compute a depth-conditioning image; MiDaS if available, luminance fallback otherwise."""
        try:
            import torch
            midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            device = self.device if str(self.device).startswith("cuda") else "cpu"
            midas.to(device).eval()
            batch = transforms.small_transform(np.asarray(image.convert("RGB"))).to(device)
            with torch.no_grad():
                prediction = midas(batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image.size[::-1],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            depth = prediction.detach().cpu().numpy()
            source = "midas"
        except Exception:
            depth = np.asarray(image.convert("L"), dtype=np.float32)
            source = "luminance_fallback"
        depth = depth.astype(np.float32)
        depth = (depth - depth.min()) / max(1e-6, float(depth.max() - depth.min()))
        self.last_conditioning_meta["depth_conditioning_enabled"] = bool(self.config.AI_REPLACE_DEPTH_CONDITIONING)
        self.last_conditioning_meta["depth_conditioning_source"] = source
        return Image.fromarray(np.clip(depth * 255.0, 0, 255).astype(np.uint8), mode="L")
    def _run_inpainting(self, image: Image.Image, mask_bundle: AIReplaceMaskBundle, prompt: str, negative_prompt: str, seed: int) -> Image.Image:
        self.last_conditioning_meta = {}
        self.last_conditioning_meta["depth_conditioning_enabled"] = bool(self.config.AI_REPLACE_DEPTH_CONDITIONING)
        self.last_conditioning_meta["depth_conditioning_source"] = "not_applied"
        self.last_conditioning_meta["depth_controlnet_applied"] = False
        if self.config.AI_REPLACE_LIGHTING_INJECT_PROMPT:
            prompt = self._build_dynamic_prompt(image, mask_bundle.bbox, prompt)
        self.last_conditioning_meta["effective_prompt"] = prompt
        self.last_conditioning_meta["num_variants"] = int(self.config.AI_REPLACE_NUM_VARIANTS)
        self.last_conditioning_meta["model_type"] = self.config.AI_REPLACE_MODEL_TYPE
        if self.pipe is None:
            # Deterministic no-model fallback for unit tests and dry-run wiring.
            generated = image.copy()
        else:
            import torch
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
            mask_image = mask_bundle.to_pil().resize(image.size, Image.NEAREST)
            call_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": image,
                "mask_image": mask_image,
                "strength": float(self.config.AI_REPLACE_STRENGTH),
                "guidance_scale": float(self.config.AI_REPLACE_GUIDANCE_SCALE),
                "num_inference_steps": int(self.config.AI_REPLACE_STEPS),
                "generator": generator,
            }
            try:
                generated = self.pipe(**call_kwargs).images[0].resize(image.size)
            except TypeError:
                call_kwargs.pop("negative_prompt", None)
                generated = self.pipe(**call_kwargs).images[0].resize(image.size)
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        if self.config.AI_REPLACE_HARD_RESTORE_OUTSIDE_MASK:
            generated = hard_restore_outside_mask(image, generated, mask_bundle)
        return generated

    def _extract_object_mask(self, output_image: Image.Image, mask_bundle: AIReplaceMaskBundle, yolo_segmenter=None) -> ObjectMaskResult:
        if yolo_segmenter is None:
            # Conservative fallback: treat changed hard-mask area as the candidate.
            object_mask = mask_bundle.hard_mask.copy()
            bbox = _mask_bbox(object_mask)
            if bbox is None:
                return ObjectMaskResult(reject_reason="no_person_detected")
            inside = _mask_inside_ratio(object_mask, mask_bundle.hard_mask)
            return ObjectMaskResult(
                object_mask=object_mask,
                object_bbox=bbox,
                object_mask_area=int((object_mask > 0.5).sum()),
                object_mask_inside_ratio=round(float(inside), 4),
                object_bbox_inside_ratio=round(float(_bbox_inside_ratio(bbox, mask_bundle.expanded_bbox)), 4),
                detector_confidence=1.0,
            )
        return self._extract_with_yolo(output_image, mask_bundle, yolo_segmenter)

    def _extract_with_yolo(self, output_image, mask_bundle, yolo_segmenter) -> ObjectMaskResult:
        results = yolo_segmenter.predict(output_image, imgsz=self.config.AI_REPLACE_RESOLUTION, conf=0.12, verbose=False)
        if not results:
            return ObjectMaskResult(reject_reason="no_person_detected")
        result = results[0]
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        if boxes is None or masks is None or boxes.xyxy is None or masks.data is None:
            return ObjectMaskResult(reject_reason="no_person_detected")
        xyxy = boxes.xyxy.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxy))
        conf = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
        best = None
        best_score = -1.0
        for index, box in enumerate(xyxy):
            if int(cls[index]) != 0:
                continue
            mask = masks.data[index].detach().cpu().numpy()
            if mask.shape != mask_bundle.hard_mask.shape:
                mask_img = Image.fromarray((mask > 0.5).astype(np.uint8) * 255, mode="L").resize(output_image.size, Image.NEAREST)
                mask = np.asarray(mask_img, dtype=np.float32) / 255.0
            inside = _mask_inside_ratio(mask, mask_bundle.hard_mask)
            score = float(conf[index]) + inside
            if score > best_score:
                best = (mask, tuple(int(round(v)) for v in box), float(conf[index]), inside)
                best_score = score
        if best is None:
            return ObjectMaskResult(reject_reason="no_person_detected")
        mask, bbox, conf_value, inside = best
        if inside < self.config.MIN_OBJECT_INSIDE_RATIO:
            return ObjectMaskResult(reject_reason="object_outside_mask")
        return ObjectMaskResult(
            object_mask=(mask > 0.5).astype(np.float32),
            object_bbox=bbox,
            object_mask_area=int((mask > 0.5).sum()),
            object_mask_inside_ratio=round(float(inside), 4),
            object_bbox_inside_ratio=round(float(_bbox_inside_ratio(bbox, mask_bundle.expanded_bbox)), 4),
            detector_confidence=round(float(conf_value), 4),
        )

    def validate_ai_replace_output(self, original, composite, obj_result, mask_bundle, ghost_result) -> ValidationResult:
        diff = outside_mask_diff(original, composite, mask_bundle)
        image_area = max(1, original.width * original.height)
        object_area_ratio = obj_result.object_mask_area / image_area
        shadow_bleed_diff = self._shadow_bleed_diff(original, composite, mask_bundle)
        object_height_ratio = _bbox_height_ratio(obj_result.object_bbox, original.height)
        checks = {
            "person_exists": obj_result.reject_reason is None,
            "person_inside_mask": obj_result.object_mask_inside_ratio >= self.config.MIN_OBJECT_INSIDE_RATIO,
            "object_area_min": object_area_ratio >= self.config.AI_REPLACE_MIN_OBJECT_AREA_RATIO,
            "object_area_max": object_area_ratio <= self.config.AI_REPLACE_MAX_OBJECT_AREA_RATIO,
            "object_height_max": object_height_ratio <= 0.48,
            "object_not_cropped": not _bbox_touches_image_margin(obj_result.object_bbox, original.size, margin_ratio=0.018),
            "person_opaque": ghost_result.opacity_score >= self.config.MIN_OPACITY_SCORE,
            "person_contrast": ghost_result.contrast_score >= self.config.MIN_CONTRAST_SCORE,
            "detector_ok": ghost_result.conf_drop <= self.config.MAX_DETECTOR_CONF_DROP,
            "background_preserved": diff <= self.config.MAX_OUTSIDE_MASK_DIFF,
            "shadow_bleeding_ok": (
                not self.config.AI_REPLACE_CHECK_SHADOW_BLEEDING
                or shadow_bleed_diff <= self.config.AI_REPLACE_MAX_SHADOW_BLEED_DIFF
            ),
        }
        accepted = all(checks.values())
        reject_reason = next((key for key, value in checks.items() if not value), None)
        return ValidationResult(accepted=accepted, checks=checks, reject_reason=reject_reason, outside_mask_diff=round(float(diff), 6))

    def _shadow_bleed_diff(self, original: Image.Image, composite: Image.Image, mask_bundle: AIReplaceMaskBundle) -> float:
        mask = mask_bundle.to_pil().resize(original.size, Image.NEAREST)
        ring = mask.filter(ImageFilter.MaxFilter(15))
        ring_arr = np.asarray(ring, dtype=np.float32) / 255.0
        hard_arr = np.asarray(mask, dtype=np.float32) / 255.0
        shadow_ring = (ring_arr > 0.1) & (hard_arr <= 0.5)
        if not np.any(shadow_ring):
            return 0.0
        orig = np.asarray(original.convert("RGB"), dtype=np.float32)
        comp = np.asarray(composite.convert("RGB").resize(original.size), dtype=np.float32)
        return float(np.mean(np.abs(comp[shadow_ring] - orig[shadow_ring])))

    def run(self, image: Image.Image, bbox: tuple, seed: int = 42, yolo_segmenter=None) -> AIReplaceResult:
        original_size = image.size
        target_size = (self.config.AI_REPLACE_RESOLUTION, self.config.AI_REPLACE_RESOLUTION)
        scaled_bbox = _scale_bbox_to_size(bbox, original_size, target_size)
        image = image.convert("RGB").resize(target_size)
        mask_bundle = self.build_mask_bundle(image, scaled_bbox)
        raw = self._run_inpainting(image, mask_bundle, self.config.AI_REPLACE_PROMPT, self.config.AI_REPLACE_NEGATIVE_PROMPT, seed)
        self.last_conditioning_meta["source_image_size"] = original_size
        self.last_conditioning_meta["working_image_size"] = target_size
        self.last_conditioning_meta["source_bbox"] = tuple(int(round(v)) for v in bbox)
        obj = self._extract_object_mask(raw, mask_bundle, yolo_segmenter=yolo_segmenter)
        if obj.object_mask is None:
            object_mask = mask_bundle.hard_mask
            object_bbox = mask_bundle.bbox
        else:
            object_mask = obj.object_mask
            object_bbox = obj.object_bbox or mask_bundle.bbox
        layers = decompose_object_mask(object_mask, object_bbox)
        harm = harmonize_object_with_background(image, raw, object_mask, layers, self.config)
        ghost = detect_ghost(image, harm.image, object_mask, obj.detector_confidence, obj.detector_confidence)
        validation = self.validate_ai_replace_output(image, harm.image, obj, mask_bundle, ghost)
        manifest = self._manifest(mask_bundle, obj, ghost, validation, harm, seed)
        return AIReplaceResult(raw, harm.image, mask_bundle, obj, ghost, validation, manifest)

    def _manifest(self, mask_bundle, obj, ghost, validation, harm, seed):
        bg_pres_score = max(0.0, 1.0 - validation.outside_mask_diff / 10.0)
        quality = (
            0.25 * obj.detector_confidence
            + 0.25 * bg_pres_score
            + 0.20 * obj.object_mask_inside_ratio
            + 0.15 * harm.harmonization_score
            + 0.10 * ghost.opacity_score
            + 0.05 * 1.0
        )
        return {
            "flow_name": self.config.AI_REPLACE_FLOW,
            "mask_source": self.config.AI_REPLACE_MASK_SOURCE,
            "bbox": mask_bundle.bbox,
            "expanded_bbox": mask_bundle.expanded_bbox,
            "mask_area_ratio": round(float(mask_bundle.mask_area_ratio), 6),
            "object_bbox": obj.object_bbox,
            "object_mask_area": obj.object_mask_area,
            "object_area_ratio": round(float(obj.object_mask_area / max(1, self.config.AI_REPLACE_RESOLUTION * self.config.AI_REPLACE_RESOLUTION)), 6),
            "object_height_ratio": round(float(_bbox_height_ratio(obj.object_bbox, self.config.AI_REPLACE_RESOLUTION)), 6),
            "object_touches_margin": bool(_bbox_touches_image_margin(obj.object_bbox, (self.config.AI_REPLACE_RESOLUTION, self.config.AI_REPLACE_RESOLUTION), margin_ratio=0.018)),
            "object_mask_inside_ratio": obj.object_mask_inside_ratio,
            "object_bbox_inside_ratio": obj.object_bbox_inside_ratio,
            "outside_mask_diff": validation.outside_mask_diff,
            **self.last_conditioning_meta,
            "hard_restore_enabled": bool(self.config.AI_REPLACE_HARD_RESTORE_OUTSIDE_MASK),
            "color_transfer_strength": harm.color_transfer_strength,
            "max_core_blend": harm.max_core_blend,
            "max_boundary_blend": harm.max_boundary_blend,
            "max_edge_blend": harm.max_edge_blend,
            "shadow_alpha": harm.shadow_alpha,
            "shadow_blur": harm.shadow_blur,
            "sharpen_strength": harm.sharpen_strength,
            "opacity_score": ghost.opacity_score,
            "contrast_score": ghost.contrast_score,
            "edge_seam_score": ghost.edge_seam_score,
            "detector_conf_before": obj.detector_confidence,
            "detector_conf_after": obj.detector_confidence,
            "detector_conf_drop": ghost.conf_drop,
            "harmonization_score": harm.harmonization_score,
            "background_preservation_score": round(float(bg_pres_score), 4),
            "ai_replace_quality_score": round(float(quality), 4),
            "accepted": bool(validation.accepted),
            "validation_checks": validation.checks,
            "reject_reason": validation.reject_reason or obj.reject_reason or ghost.reject_reason or "",
            "reject_stage": "" if validation.accepted else "validation",
            "seed": int(seed),
        }

    @staticmethod
    def save_result(result: AIReplaceResult, output_dir: str | Path, stem: str = "sample") -> None:
        output_dir = Path(output_dir)
        preview_dir = output_dir / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        result.raw_image.save(preview_dir / f"{stem}_generated_raw.png")
        result.harmonized_image.save(preview_dir / f"{stem}_harmonized.png")
        result.mask_bundle.to_pil().save(preview_dir / f"{stem}_mask.png")
        with (output_dir / "manifest.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.manifest_row, ensure_ascii=False) + "\n")
