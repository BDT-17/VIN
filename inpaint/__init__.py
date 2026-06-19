"""Pokecut-style AI Replace flow for VIN inpainting experiments.

Heavy modules are intentionally not imported here so config/docs can be read in
minimal environments without numpy/torch installed.
"""

__all__ = [
    "AIReplacePipeline",
    "AIReplaceResult",
    "AIReplaceMaskBundle",
    "ValidationResult",
    "bbox_to_mask",
    "refine_mask",
]
