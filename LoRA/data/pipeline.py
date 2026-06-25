"""Thin ETL orchestrator: raw datasets -> validated LoRA release + eval cases.

Notebooks call run_full_etl(); they hold no ETL logic themselves.
"""

from pathlib import Path

import pandas as pd

from .config import load_sources_config, load_prompt_config
from .ingest import run_ingest
from .normalize import run_normalize
from .dedupe import run_dedupe
from .curate import run_curate
from .captions import run_captions
from .splits import assign_splits
from .export import run_export
from .validate import validate_release
from .build_eval_cases import run_build_eval_cases


def run_full_etl(work_dir, sources_path=None, prompts_path=None, repo_dir=None) -> dict:
    work_dir = Path(work_dir)
    config = load_sources_config(sources_path)
    prompts = load_prompt_config(prompts_path)

    print("=" * 70); print(f"LoRA ETL — release: {config.release_name}"); print("=" * 70)

    print("\n[00] ingest");        inv_dir = run_ingest(config, work_dir)
    print("[01] normalize");        norm_dir = run_normalize(inv_dir, work_dir)
    print("[02] dedupe/group");     cur_dir = run_dedupe(norm_dir, work_dir)
    print("[03] build eval cases"); eval_root = run_build_eval_cases(norm_dir, work_dir, config)
    print("[04] filter/crop");      run_curate(norm_dir, work_dir, config)

    candidates = pd.read_parquet(cur_dir / "lora_candidates.parquet")
    groups = pd.read_parquet(cur_dir / "groups.parquet")

    print("[05] caption")
    captioned = run_captions(candidates, prompts, config.caption_config)
    print("[06] split")
    samples = assign_splits(captioned, config.split_config, groups)

    print("[07] export")
    release_dir = run_export(samples, work_dir, config, repo_dir=repo_dir)

    print("[08] validate")
    eval_manifest = None
    em_path = eval_root / "eval_manifest.parquet"
    if em_path.exists():
        eval_manifest = pd.read_parquet(em_path)
    report = validate_release(release_dir, eval_manifest=eval_manifest)

    print("\n" + ("VALID" if report["valid"] else "INVALID"), report["stats"])
    return {"release_dir": release_dir, "eval_root": eval_root, "report": report}
