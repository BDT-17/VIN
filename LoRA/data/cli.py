"""CLI entry point for LoRA data pipeline."""

import argparse
from pathlib import Path
import sys

from .config import load_sources_config
from .ingest import run_ingestion


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="VIN LoRA Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build dataset from sources.yaml
  python -m LoRA.data.cli build --config LoRA/data/sources.yaml --release pedestrian_lora_v1

  # Validate a release
  python -m LoRA.data.cli validate --release /kaggle/working/vin_data/releases/pedestrian_lora_v1

  # Generate report
  python -m LoRA.data.cli report --release /kaggle/working/vin_data/releases/pedestrian_lora_v1
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build dataset from sources")
    build_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to sources.yaml configuration file"
    )
    build_parser.add_argument(
        "--release",
        type=str,
        help="Release name (overrides config)"
    )
    build_parser.add_argument(
        "--working-dir",
        type=Path,
        default=Path("/kaggle/working/vin_data"),
        help="Working directory for pipeline outputs"
    )

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a dataset release")
    validate_parser.add_argument(
        "--release",
        type=Path,
        required=True,
        help="Path to release directory"
    )

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate dataset report")
    report_parser.add_argument(
        "--release",
        type=Path,
        required=True,
        help="Path to release directory"
    )

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "report":
        cmd_report(args)
    else:
        parser.print_help()
        sys.exit(1)


def cmd_build(args):
    """Execute build command."""
    from .pipeline import LoRAPipeline

    print("=" * 70)
    print("VIN LoRA DATA PIPELINE - BUILD")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Working directory: {args.working_dir}")
    print()

    # Run full pipeline
    pipeline = LoRAPipeline(
        config_path=args.config,
        working_dir=args.working_dir,
    )

    release_dir = pipeline.run_full_pipeline()

    print(f"\n✓ Build complete.")
    print(f"Release: {release_dir}")
    print(f"\nNext steps:")
    print(f"  1. Validate: python -m LoRA.data.cli validate --release {release_dir}")
    print(f"  2. Train: python -m LoRA.sd35_lora_training --dataset_release {release_dir}")


def cmd_validate(args):
    """Execute validate command."""
    from .validate import validate_release

    print("=" * 70)
    print("VIN LoRA DATA PIPELINE - VALIDATE")
    print("=" * 70)
    print(f"Release: {args.release}")
    print()

    result = validate_release(args.release)

    if result['valid']:
        print("\n✅ VALIDATION PASSED")
        print("\nStats:")
        for key, value in result['stats'].items():
            print(f"  {key}: {value}")
    else:
        print("\n❌ VALIDATION FAILED")
        print("\nErrors:")
        for error in result['errors']:
            print(f"  - {error}")

        if result['warnings']:
            print("\nWarnings:")
            for warning in result['warnings']:
                print(f"  - {warning}")

        sys.exit(1)


def cmd_report(args):
    """Execute report command."""
    from .report import generate_release_report

    print("=" * 70)
    print("VIN LoRA DATA PIPELINE - REPORT")
    print("=" * 70)
    print(f"Release: {args.release}")
    print()

    # Generate report in release's report directory
    report_dir = args.release / "reports"
    generate_release_report(args.release, report_dir)

    print(f"\n✓ Report generated: {report_dir}")


if __name__ == "__main__":
    main()
