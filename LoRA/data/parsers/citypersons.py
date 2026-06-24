"""CityPersons-specific parser extensions.

Handles CityPersons-specific annotation format if needed.
Currently, most CityPersons datasets on Kaggle are in YOLO format,
so this module can provide CityPersons-specific logic like
occlusion handling, ignore regions, etc.
"""

from pathlib import Path
from typing import Dict, Optional
from .yolo import YOLOParser


class CityPersonsParser(YOLOParser):
    """CityPersons-aware YOLO parser.

    Extends YOLOParser with CityPersons-specific conventions:
    - Ignore regions handling
    - Occlusion levels
    - Visibility annotations
    """

    def __init__(
        self,
        source_id: str,
        mount_path: Path,
        splits: Dict[str, str],
        label_dirs: Dict[str, str],
        class_mapping: Optional[Dict[int, str]] = None,
    ):
        # CityPersons YOLO class mapping (example)
        if class_mapping is None:
            class_mapping = {
                0: "pedestrian",
                1: "ignore",
            }

        super().__init__(
            source_id=source_id,
            mount_path=mount_path,
            splits=splits,
            label_dirs=label_dirs,
            class_mapping=class_mapping,
        )

    def _compute_group_id(self, stem: str, split_name: str) -> str:
        """CityPersons-specific group ID computation.

        CityPersons images follow naming:
            {city}_{sequence}_{frame}_leftImg8bit

        Group by city and sequence to ensure frames from the same
        video sequence stay together during splits.
        """
        parts = stem.split("_")

        if len(parts) >= 2:
            # Extract city and sequence
            city = parts[0]
            sequence = parts[1]
            group_id = f"{self.source_id}_{split_name}_{city}_{sequence}"
        else:
            # Fallback
            group_id = f"{self.source_id}_{split_name}_{stem}"

        return group_id
