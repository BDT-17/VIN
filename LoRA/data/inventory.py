"""Raw dataset inventory and discovery.

Scans all configured sources and builds the initial raw inventory
before normalization and deduplication.
"""

from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
from tqdm import tqdm

from .config import PipelineConfig, SourceDefinition
from .schema import ImageRecord, InstanceRecord
from .parsers import MOTParser, YOLOParser, ClassificationParser
from .parsers.citypersons import CityPersonsParser


class InventoryBuilder:
    """Build raw inventory from all configured sources."""

    def __init__(self, config: PipelineConfig, working_dir: Path):
        self.config = config
        self.working_dir = Path(working_dir)
        self.inventory_dir = self.working_dir / "raw_inventory"
        self.inventory_dir.mkdir(parents=True, exist_ok=True)

    def build(self) -> Dict[str, Path]:
        """Build inventory for all sources.

        Returns:
            Dictionary mapping source_id to inventory parquet paths.
        """
        inventory_paths = {}

        for source in tqdm(self.config.sources, desc="Building inventory"):
            print(f"\nProcessing source: {source.source_id}")

            try:
                images, instances = self._parse_source(source)

                # Save to parquet
                img_path = self._save_images(source.source_id, images)
                inst_path = self._save_instances(source.source_id, instances)

                inventory_paths[source.source_id] = {
                    "images": img_path,
                    "instances": inst_path,
                }

                print(f"  Images: {len(images)}")
                print(f"  Instances: {len(instances)}")

            except Exception as e:
                print(f"  ERROR: Failed to process {source.source_id}: {e}")
                raise

        return inventory_paths

    def _parse_source(
        self,
        source: SourceDefinition
    ) -> Tuple[List[ImageRecord], List[InstanceRecord]]:
        """Parse a single source using the appropriate parser."""

        parser_name = source.parser.lower()
        mount_path = Path(source.kaggle_mount)

        if parser_name == "mot":
            parser = MOTParser(
                source_id=source.source_id,
                mount_path=mount_path,
                sequence_dir=source.sequence_dir,
                gt_dir=source.gt_dir,
                temporal_sampling_fps=source.temporal_sampling_fps,
                temporal_window_seconds=source.temporal_window_seconds,
            )
        elif parser_name == "yolo":
            parser = YOLOParser(
                source_id=source.source_id,
                mount_path=mount_path,
                splits=source.splits,
                label_dirs=source.label_dirs,
            )
        elif parser_name == "citypersons":
            parser = CityPersonsParser(
                source_id=source.source_id,
                mount_path=mount_path,
                splits=source.splits,
                label_dirs=source.label_dirs,
            )
        elif parser_name == "classification_folders":
            parser = ClassificationParser(
                source_id=source.source_id,
                mount_path=mount_path,
                positive_dir=source.positive_dir,
                negative_dir=source.negative_dir,
            )
        else:
            raise ValueError(f"Unknown parser: {parser_name}")

        return parser.parse()

    def _save_images(self, source_id: str, images: List[ImageRecord]) -> Path:
        """Save image records to parquet."""
        if not images:
            print(f"  Warning: No images for {source_id}")
            return None

        df = pd.DataFrame([vars(img) for img in images])
        path = self.inventory_dir / f"{source_id}_images.parquet"
        df.to_parquet(path, index=False)
        return path

    def _save_instances(self, source_id: str, instances: List[InstanceRecord]) -> Path:
        """Save instance records to parquet."""
        if not instances:
            print(f"  Warning: No instances for {source_id}")
            # Create empty parquet with correct schema
            df = pd.DataFrame(columns=[
                "instance_id", "image_id", "class_name",
                "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                "visible_bbox_x", "visible_bbox_y", "visible_bbox_w", "visible_bbox_h",
                "track_id", "occlusion_level", "ignore_flag", "confidence",
                "annotation_origin",
            ])
        else:
            df = pd.DataFrame([vars(inst) for inst in instances])

        path = self.inventory_dir / f"{source_id}_instances.parquet"
        df.to_parquet(path, index=False)
        return path

    def validate_inventory(self) -> Dict[str, List[str]]:
        """Validate all inventory records.

        Returns:
            Dictionary mapping source_id to list of validation errors.
        """
        errors = {}

        for source in self.config.sources:
            source_errors = []

            img_path = self.inventory_dir / f"{source.source_id}_images.parquet"
            inst_path = self.inventory_dir / f"{source.source_id}_instances.parquet"

            if not img_path.exists():
                source_errors.append(f"Missing image inventory: {img_path}")
                errors[source.source_id] = source_errors
                continue

            # Load and validate
            img_df = pd.read_parquet(img_path)

            for idx, row in img_df.iterrows():
                record = ImageRecord(**row.to_dict())
                record_errors = record.validate()
                if record_errors:
                    source_errors.extend([f"Image {record.image_id}: {e}" for e in record_errors])

            # Validate instances if present
            if inst_path.exists():
                inst_df = pd.read_parquet(inst_path)

                # Create image dimension lookup
                img_dims = {row['image_id']: (row['width'], row['height'])
                           for _, row in img_df.iterrows()}

                for idx, row in inst_df.iterrows():
                    record = InstanceRecord(**row.to_dict())
                    width, height = img_dims.get(record.image_id, (0, 0))
                    record_errors = record.validate(width, height)
                    if record_errors:
                        source_errors.extend([f"Instance {record.instance_id}: {e}" for e in record_errors])

            if source_errors:
                errors[source.source_id] = source_errors

        return errors
