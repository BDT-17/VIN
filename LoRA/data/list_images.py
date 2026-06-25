"""List background/source images straight from the datasets in sources.yaml.

Lets notebooks pull real images from the mounted Kaggle datasets (the same
`kaggle_mount` paths the ETL uses) instead of asking the user to upload files.
Also resolves the alternate mount forms (/kaggle/input/<slug> vs
/kaggle/input/datasets/<user>/<slug>) so it works regardless of how the dataset
was attached.
"""

from pathlib import Path
from typing import List, Optional

from .config import load_sources_config, SourceDefinition

_IMG_EXT = (".jpg", ".jpeg", ".png")


def _resolve_mount(mount: str) -> Optional[Path]:
    """Return an existing mount path, trying the configured one and common
    alternates (with/without the `datasets/<user>/` segment)."""
    p = Path(mount)
    if p.exists():
        return p
    parts = p.parts
    # /kaggle/input/datasets/<user>/<slug>/...  <->  /kaggle/input/<slug>/...
    if "datasets" in parts:
        i = parts.index("datasets")
        # drop "datasets/<user>" -> /kaggle/input/<slug>/...
        alt = Path(*parts[:i], *parts[i + 2:])
        if alt.exists():
            return alt
    else:
        # try inserting datasets/* by globbing the slug under /kaggle/input/datasets
        try:
            idx = parts.index("input")
            tail = Path(*parts[idx + 1:])
            for cand in Path("/kaggle/input/datasets").glob("*/" + str(tail)):
                if cand.exists():
                    return cand
        except (ValueError, OSError):
            pass
    return None


def _source_image_dirs(src: SourceDefinition, base: Path, splits: List[str]) -> List[Path]:
    """Image directories for a source given which splits to read."""
    dirs = []
    if src.parser in ("yolo", "citypersons"):
        for sp in splits:
            sub = (src.splits or {}).get(sp)
            if sub:
                dirs.append(base / sub)
    elif src.parser == "mot":
        if src.sequence_dir:
            dirs.append(base / src.sequence_dir)
    elif src.parser == "classification_folders":
        if src.positive_dir:
            dirs.append(base / src.positive_dir)
    return dirs


def list_source_images(source_id: str, splits: Optional[List[str]] = None,
                       limit: Optional[int] = None, sources_path=None) -> List[Path]:
    """Return image paths for one source in sources.yaml.

    Args:
        source_id: e.g. 'citypersons', 'mot17_02', 'human_detection'
        splits: which splits to read (default: the source's eval_splits, else
                lora_splits, else ['train']). Ignored for mot/classification.
        limit: cap the number of paths returned.
    """
    cfg = load_sources_config(sources_path)
    src = next((s for s in cfg.sources if s.source_id == source_id), None)
    if src is None:
        raise ValueError(f"source_id '{source_id}' not in sources.yaml")

    base = _resolve_mount(src.kaggle_mount)
    if base is None:
        raise FileNotFoundError(
            f"Dataset for '{source_id}' not mounted. Expected near "
            f"{src.kaggle_mount}. Add it via Kaggle 'Add Data'.")

    splits = splits or src.eval_splits or src.lora_splits or ["train"]
    paths: List[Path] = []
    for d in _source_image_dirs(src, base, splits):
        if d.exists():
            paths += [p for p in sorted(d.rglob("*")) if p.suffix.lower() in _IMG_EXT]
    if limit is not None:
        paths = paths[:limit]
    return paths


def list_any_source_images(source_ids: Optional[List[str]] = None,
                           per_source_limit: Optional[int] = None,
                           sources_path=None) -> List[Path]:
    """Concatenate images from several sources (first that resolve)."""
    cfg = load_sources_config(sources_path)
    ids = source_ids or [s.source_id for s in cfg.sources]
    out: List[Path] = []
    for sid in ids:
        try:
            out += list_source_images(sid, limit=per_source_limit, sources_path=sources_path)
        except FileNotFoundError:
            continue
    return out
