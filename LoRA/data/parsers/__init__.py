"""Dataset parsers for VIN LoRA pipeline.

Each parser converts a specific dataset format into the canonical
ImageRecord and InstanceRecord schema.
"""

from .mot import MOTParser
from .yolo import YOLOParser
from .classification import ClassificationParser

__all__ = [
    "MOTParser",
    "YOLOParser",
    "ClassificationParser",
]
