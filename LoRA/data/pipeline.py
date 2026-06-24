"""Complete LoRA data pipeline orchestration.

Runs the full pipeline from raw sources to validated release.
"""

from pathlib import Path
from typing import Dict, Optional
import hashlib
import subprocess

from .config import load_sources_config
from .inventory import InventoryBuilder
from .dedupe import ImageDeduplicator
from .normalize import DatasetNormalizer
from .splits import DatasetSplitter, SplitValidator, create_split_locks
from .quality import QualityFilter
from .captions import CaptionGenerator, validate_captions
from .crops import build_all_crops
from .export import DatasetExporter
from .validate import validate_release
from .report import generate_release_report


class LoRAPipeline:
    """Complete LoRA data pipeline orchestrator."""

    def __init__(self, config_path: Path, working_dir: Path):
        self.config_path = Path(config_path)
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self.config = load_sources_config(config_path)

        print("=" * 70)
        print("VIN LORA DATA PIPELINE")
        print("=" * 70)
        print(f"Release: {self.config.release_name}")
        print(f"Working directory: {self.working_dir}")
        print(f"Sources: {len(self.config.sources)}")
        print()

    def run_full_pipeline(self) -> Path:
        """Execute complete pipeline.

        Returns:
            Path to validated dataset release
        """
        # Phase 1: Inventory
        print("\n" + "=" * 70)
        print("PHASE 1: INVENTORY")
        print("=" * 70)
        inventory_builder = InventoryBuilder(self.config, self.working_dir)
        inventory_paths = inventory_builder.build()
        errors = inventory_builder.validate_inventory()
        if errors:
            raise ValueError(f"Inventory validation failed: {errors}")

        # Phase 2: Deduplication
        print("\n" + "=" * 70)
        print("PHASE 2: DEDUPLICATION")
        print("=" * 70)
        deduplicator = ImageDeduplicator(self.working_dir)
        source_priorities = {src.source_id: src.duplicate_priority for src in self.config.sources}
        groups_path = deduplicator.deduplicate_all_sources(source_priorities)

        # Phase 3: Normalization
        print("\n" + "=" * 70)
        print("PHASE 3: NORMALIZATION")
        print("=" * 70)
        import pandas as pd
        groups_df = pd.read_parquet(groups_path)
        normalizer = DatasetNormalizer(self.working_dir)
        images_path = normalizer.normalize_images(groups_df)
        instances_path = normalizer.normalize_instances()

        # Phase 4: Quality Filtering
        print("\n" + "=" * 70)
        print("PHASE 4: QUALITY FILTERING")
        print("=" * 70)
        images_df = pd.read_parquet(images_path)
        instances_df = pd.read_parquet(instances_path) if instances_path else pd.DataFrame()

        quality_filter = QualityFilter(
            min_bbox_height_px=self.config.quality_thresholds.min_bbox_height_px,
            min_visible_ratio=self.config.quality_thresholds.min_visible_ratio,
            min_quality_score=self.config.quality_thresholds.min_quality_score,
            max_source_share=self.config.quality_thresholds.max_source_share,
        )

        filtered_instances = quality_filter.filter_samples(instances_df, images_df)
        balanced_instances = quality_filter.balance_sources(filtered_instances)

        # Phase 5: Split Assignment
        print("\n" + "=" * 70)
        print("PHASE 5: SPLIT ASSIGNMENT")
        print("=" * 70)
        splitter = DatasetSplitter(
            working_dir=self.working_dir,
            train_ratio=self.config.split_config.train_ratio,
            val_ratio=self.config.split_config.val_ratio,
            test_ratio=self.config.split_config.test_ratio,
            split_seed=self.config.split_config.split_seed,
        )

        splits_df = splitter.assign_splits(images_df, groups_df)

        # Merge split assignments into instances
        balanced_instances = balanced_instances.merge(
            splits_df[['image_id', 'assigned_split']],
            on='image_id',
            how='left'
        )

        # Create sample records
        samples = []
        for idx, row in balanced_instances.iterrows():
            sample_id = f"{self.config.release_name}_{row['instance_id']}"
            samples.append({
                'sample_id': sample_id,
                'image_id': row['image_id'],
                'instance_id': row['instance_id'],
                'role': 'lora_positive',  # Will be updated for benchmark-locked
                'split': row['assigned_split'],
                'crop_path': '',  # Will be filled during crop building
                'crop_width': 0,
                'crop_height': 0,
                'bbox_height_ratio': row['bbox_height_ratio'],
                'visible_ratio': row['visible_ratio'],
                'occlusion_level': row.get('occlusion_level'),
                'source_id': row['source_id'],
                'quality_score': row['quality_score'],
                'caption': '',  # Will be filled during captioning
                'trigger_token': self.config.caption_config.trigger_token,
                'duplicate_cluster_id': '',  # Will be filled from groups
                'split_group_id': '',  # Will be filled from groups
            })

        samples_df = pd.DataFrame(samples)

        # Merge group IDs
        samples_df = samples_df.merge(
            groups_df[['image_id', 'duplicate_cluster_id', 'split_group_id']],
            on='image_id',
            how='left',
            suffixes=('', '_from_groups')
        )
        samples_df['duplicate_cluster_id'] = samples_df['duplicate_cluster_id_from_groups']
        samples_df['split_group_id'] = samples_df['split_group_id_from_groups']
        samples_df = samples_df.drop(columns=['duplicate_cluster_id_from_groups', 'split_group_id_from_groups'])

        # Phase 6: Benchmark Locks
        print("\n" + "=" * 70)
        print("PHASE 6: BENCHMARK LOCKS")
        print("=" * 70)
        benchmark_sources = [src.source_id for src in self.config.sources if src.benchmark_lock]
        if benchmark_sources:
            print(f"Benchmark-locked sources: {benchmark_sources}")
            samples_df = create_split_locks(samples_df, benchmark_sources, images_df)
        else:
            print("No benchmark-locked sources")

        # Phase 7: Caption Generation
        print("\n" + "=" * 70)
        print("PHASE 7: CAPTION GENERATION")
        print("=" * 70)
        caption_gen = CaptionGenerator(
            trigger_token=self.config.caption_config.trigger_token,
            template_version=self.config.caption_config.template_version,
        )
        samples_df = caption_gen.generate_all_captions(samples_df, images_df)

        # Validate captions
        caption_errors = validate_captions(
            samples_df,
            trigger_token=self.config.caption_config.trigger_token,
            min_tokens=self.config.caption_config.min_caption_tokens,
            max_tokens=self.config.caption_config.max_caption_tokens,
        )
        if caption_errors:
            print(f"\n⚠️  Caption validation warnings: {len(caption_errors)}")
            for err in caption_errors[:5]:
                print(f"  - {err}")

        # Phase 8: Crop Building
        print("\n" + "=" * 70)
        print("PHASE 8: CROP BUILDING")
        print("=" * 70)
        curated_dir = self.working_dir / "curated"
        curated_dir.mkdir(exist_ok=True)
        samples_df = build_all_crops(
            images_df=images_df,
            instances_df=balanced_instances,
            samples_df=samples_df,
            output_dir=curated_dir,
            context_ratio=self.config.export_config.crop_context_ratio,
        )

        # Save curated samples
        samples_df.to_parquet(curated_dir / "samples.parquet", index=False)

        # Phase 9: Split Validation
        print("\n" + "=" * 70)
        print("PHASE 9: SPLIT VALIDATION")
        print("=" * 70)
        validation_result = SplitValidator.validate_split_safety(
            images_df=images_df,
            groups_df=groups_df,
            samples_df=samples_df,
            benchmark_locked_sources=benchmark_sources,
        )

        if not validation_result['valid']:
            print("\n❌ SPLIT VALIDATION FAILED:")
            for error in validation_result['errors']:
                print(f"  - {error}")
            raise ValueError("Split validation failed. Cannot proceed to export.")

        print("\n✓ Split validation passed (zero leakage)")

        # Phase 10: Export Release
        print("\n" + "=" * 70)
        print("PHASE 10: EXPORT RELEASE")
        print("=" * 70)

        # Compute config hash
        config_hash = hashlib.sha256(
            self.config_path.read_bytes()
        ).hexdigest()[:16]

        # Get git commit if available
        git_commit = None
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                git_commit = result.stdout.strip()
        except Exception:
            pass

        exporter = DatasetExporter(
            working_dir=self.working_dir,
            release_name=self.config.release_name,
        )

        release_dir = exporter.export_release(
            samples_df=samples_df,
            images_df=images_df,
            groups_df=groups_df,
            config_hash=config_hash,
            git_commit=git_commit,
        )

        # Phase 11: Final Validation
        print("\n" + "=" * 70)
        print("PHASE 11: FINAL VALIDATION")
        print("=" * 70)

        final_validation = validate_release(release_dir)

        if final_validation['valid']:
            print("\n✅ RELEASE VALIDATED")
            print(f"  Train samples: {final_validation['stats']['train_count']}")
            print(f"  Val samples: {final_validation['stats']['val_count']}")
            print(f"  Zero leakage: ✓")
        else:
            print("\n❌ RELEASE VALIDATION FAILED:")
            for error in final_validation['errors']:
                print(f"  - {error}")
            raise ValueError("Release validation failed")

        # Phase 12: Generate Report
        print("\n" + "=" * 70)
        print("PHASE 12: GENERATE REPORT")
        print("=" * 70)

        reports_dir = self.working_dir / "reports" / "data" / self.config.release_name
        generate_release_report(release_dir, reports_dir)

        # Done
        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETE")
        print("=" * 70)
        print(f"Release: {release_dir}")
        print(f"Reports: {reports_dir}")
        print()

        return release_dir
