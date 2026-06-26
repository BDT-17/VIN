"""SAM-2 mask refinement for Add-it Subject-Guided Latent Blending (paper §3.4).

The paper refines the coarse Otsu attention mask into a precise object mask with
SAM-2 (Ravi et al., 2024), prompted by point coordinates sampled from the
attention map (App A.1).  SAM-2 is a *soft* dependency here: if neither the
``sam2`` package nor a usable HF auto-pipeline is importable, we fall back to the
raw coarse Otsu mask (the paper's ablation table 7 shows the coarse mask still
scores 0.809 affordance vs 0.828 for the full point-prompted SAM path, so the
fallback is degraded-but-functional, not broken).

Public API:
    refine_mask_with_sam2(image, points, coarse_mask) -> PIL.Image ("L")
    sam2_available() -> bool
"""

from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

_SAM2_PREDICTOR = None          # cached predictor instance
_SAM2_BACKEND = None            # "sam2" | "transformers" | "unavailable"


def _try_load_sam2():
    """Attempt to load a SAM-2 image predictor.  Sets module-level cache.

    Tries, in order:
      1. The official ``sam2`` package (``SAM2ImagePredictor`` + a HF checkpoint).
      2. A HF ``transformers`` Sam2 model if present.
    Any failure leaves the backend as "unavailable" and is non-fatal.
    """
    global _SAM2_PREDICTOR, _SAM2_BACKEND
    if _SAM2_BACKEND is not None:
        return

    # --- Option 1: official sam2 package ---
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

        # from_pretrained pulls the checkpoint from the HF hub on first use.
        predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-small")
        _SAM2_PREDICTOR = predictor
        _SAM2_BACKEND = "sam2"
        print("Add-it SAM-2: using official sam2 package (facebook/sam2-hiera-small).")
        return
    except Exception as exc:  # noqa: BLE001 — soft dependency
        print(f"Add-it SAM-2: sam2 package unavailable ({type(exc).__name__}: {exc}).")

    # --- Option 2: transformers Sam2 ---
    try:
        import torch  # noqa: F401
        from transformers import Sam2Model, Sam2Processor  # type: ignore

        model = Sam2Model.from_pretrained("facebook/sam2-hiera-small")
        processor = Sam2Processor.from_pretrained("facebook/sam2-hiera-small")
        if torch.cuda.is_available():
            model = model.to("cuda")
        _SAM2_PREDICTOR = (model, processor)
        _SAM2_BACKEND = "transformers"
        print("Add-it SAM-2: using transformers Sam2 (facebook/sam2-hiera-small).")
        return
    except Exception as exc:  # noqa: BLE001
        print(f"Add-it SAM-2: transformers Sam2 unavailable ({type(exc).__name__}: {exc}).")

    _SAM2_BACKEND = "unavailable"
    print("Add-it SAM-2: no SAM-2 backend; falling back to coarse Otsu masks.")


def sam2_available() -> bool:
    _try_load_sam2()
    return _SAM2_BACKEND in ("sam2", "transformers")


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component of a binary uint8 mask."""
    try:
        from scipy import ndimage  # type: ignore

        labeled, n = ndimage.label(mask > 0)
        if n <= 1:
            return mask
        sizes = ndimage.sum(mask > 0, labeled, range(1, n + 1))
        keep = int(np.argmax(sizes)) + 1
        return (labeled == keep).astype(np.uint8) * 255
    except Exception:  # noqa: BLE001 — scipy optional
        return mask


def refine_mask_with_sam2(
    image: Image.Image,
    points: List[Tuple[int, int]],
    coarse_mask: Optional[np.ndarray] = None,
) -> Image.Image:
    """Refine the subject mask using SAM-2 prompted by ``points`` (paper §3.4).

    Parameters
    ----------
    image : PIL.Image
        The estimated clean image X_0 at the blend timestep (paper estimates it
        from the velocity prediction before calling SAM-2).
    points : list of (x, y)
        Foreground point prompts sampled from the attention map.
    coarse_mask : np.ndarray, optional
        The Otsu coarse mask, used as the fallback / sanity floor.

    Returns
    -------
    PIL.Image ("L") — refined mask, 255 = subject.
    """
    _try_load_sam2()
    h, w = image.height, image.width

    if not points or _SAM2_BACKEND == "unavailable":
        return _fallback_mask(coarse_mask, (h, w))

    rgb = np.asarray(image.convert("RGB"))
    pt_coords = np.array(points, dtype=np.float32)
    pt_labels = np.ones(len(points), dtype=np.int32)  # all foreground

    try:
        if _SAM2_BACKEND == "sam2":
            predictor = _SAM2_PREDICTOR
            predictor.set_image(rgb)
            masks, scores, _ = predictor.predict(
                point_coords=pt_coords,
                point_labels=pt_labels,
                multimask_output=True,
            )
            best = int(np.argmax(scores))
            mask = (masks[best] > 0).astype(np.uint8) * 255
        else:  # transformers
            import torch

            model, processor = _SAM2_PREDICTOR
            inputs = processor(
                images=rgb,
                input_points=[[list(map(float, p)) for p in points]],
                input_labels=[[1] * len(points)],
                return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
            masks = processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"].cpu()
            )[0]
            scores = outputs.iou_scores.cpu().numpy().reshape(-1)
            best = int(np.argmax(scores))
            mask = (masks[best].numpy() > 0).astype(np.uint8) * 255
            if mask.ndim == 3:
                mask = mask[0]
    except Exception as exc:  # noqa: BLE001 — fall back rather than crash a run
        print(f"Add-it SAM-2 refine failed ({type(exc).__name__}: {exc}); using coarse mask.")
        return _fallback_mask(coarse_mask, (h, w))

    if mask.shape != (h, w):
        mask = np.asarray(Image.fromarray(mask, "L").resize((w, h), Image.NEAREST))
    mask = _largest_component(mask)
    return Image.fromarray(mask, mode="L")


def _fallback_mask(coarse_mask: Optional[np.ndarray], hw: Tuple[int, int]) -> Image.Image:
    h, w = hw
    if coarse_mask is None:
        return Image.new("L", (w, h), 0)
    arr = coarse_mask
    if arr.shape != (h, w):
        arr = np.asarray(Image.fromarray(arr.astype(np.uint8), "L").resize((w, h), Image.NEAREST))
    return Image.fromarray(_largest_component(arr.astype(np.uint8)), mode="L")
