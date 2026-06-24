"""Dataset ingestion orchestrator.

Coordinates the full ingestion pipeline from raw sources to normalized manifests.
"""

from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime

from .config import PipelineConfig, load_sources_config
from .inventory import InventoryBuilder


class DatasetIngestor:
    """Orchestrate dataset ingestion pipeline."""

    def __init__(self, config_path: Path, working_dir: Path):
        self.config_path = Path(config_path)
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self.config = load_sources_config(config_path)

        # Create subdirectories
        self.raw_inventory_dir = self.working_dir / "raw_inventory"
        self.normalized_dir = self.working_dir / "normalized"
        self.reports_dir = self.working_dir / "reports" / "data" / self.config.release_name

        for d in [self.raw_inventory_dir, self.normalized_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def run_inventory(self) -> Dict[str, Path]:
        """Run inventory phase: discover and parse all sources."""
        print("=" * 60)
        print("PHASE 1: INVENTORY")
        print("=" * 60)

        builder = InventoryBuilder(self.config, self.working_dir)
        inventory_paths = builder.build()

        # Validate
        print("\nValidating inventory...")
        errors = builder.validate_inventory()

        if errors:
            print("\n⚠️  VALIDATION ERRORS FOUND:")
            for source_id, source_errors in errors.items():
                print(f"\n{source_id}:")
                for error in source_errors[:10]:  # Show first 10
                    print(f"  - {error}")
                if len(source_errors) > 10:
                    print(f"  ... and {len(source_errors) - 10} more errors")

            # Save error report
            error_path = self.reports_dir / "inventory_errors.json"
            with open(error_path, 'w') as f:
                json.dump(errors, f, indent=2)

            raise ValueError(f"Inventory validation failed. See {error_path}")

        print("\n✓ Inventory validation passed")

        # Save inventory report
        self._save_inventory_report(inventory_paths)

        return inventory_paths

    def _save_inventory_report(self, inventory_paths: Dict[str, Path]):
        """Generate and save inventory report."""
        import pandas as pd

        report = {
            "release_name": self.config.release_name,
            "generated_at": datetime.now().isoformat(),
            "sources": {},
        }

        for source_id, paths in inventory_paths.items():
            img_df = pd.read_parquet(paths["images"])
            inst_df = pd.read_parquet(paths["instances"]) if paths["instances"] else pd.DataFrame()

            report["sources"][source_id] = {
                "image_count": len(img_df),
                "instance_count": len(inst_df),
                "total_size_mb": img_df["file_size_bytes"].sum() / (1024 * 1024),
                "splits": img_df["original_split"].value_counts().to_dict() if "original_split" in img_df.columns else {},
            }

        report_path = self.reports_dir / "inventory.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nInventory report saved: {report_path}")

        # Print summary
        print("\nInventory Summary:")
        print("-" * 60)
        for source_id, stats in report["sources"].items():
            print(f"{source_id}:")
            print(f"  Images: {stats['image_count']}")
            print(f"  Instances: {stats['instance_count']}")
            print(f"  Size: {stats['total_size_mb']:.1f} MB")
            if stats['splits']:
                print(f"  Splits: {stats['splits']}")


def run_ingestion(config_path: Path, working_dir: Path):
    """Run the full ingestion pipeline."""
    ingestor = DatasetIngestor(config_path, working_dir)

    # Phase 1: Inventory
    inventory_paths = ingestor.run_inventory()

    print("\n" + "=" * 60)
    print("✓ INGESTION COMPLETE")
    print("=" * 60)

    return ingestor
