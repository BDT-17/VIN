"""list_images: source lookup + mount-form resolution (no real Kaggle mounts)."""

import pytest

from LoRA.data import list_images
from LoRA.data.config import load_sources_config


def test_unknown_source_raises():
    with pytest.raises(ValueError):
        list_images.list_source_images("nope_not_a_source")


def test_missing_mount_raises_filenotfound():
    # citypersons mount won't exist on the dev box -> clear error, not a crash
    with pytest.raises(FileNotFoundError):
        list_images.list_source_images("citypersons")


def test_resolve_mount_prefers_existing(tmp_path):
    # configured path missing, but the /datasets/<user>/ -> /<slug> alternate exists
    slug_dir = tmp_path / "citypersons-x"
    (slug_dir).mkdir()
    configured = tmp_path / "datasets" / "someuser" / "citypersons-x"
    got = list_images._resolve_mount(str(configured))
    # alternate (drop datasets/<user>) resolves to slug_dir
    assert got == slug_dir


def test_source_image_dirs_yolo(tmp_path):
    cfg = load_sources_config()
    src = next(s for s in cfg.sources if s.source_id == "citypersons")
    base = tmp_path
    dirs = list_images._source_image_dirs(src, base, ["val"])
    assert dirs == [base / "valid/images"]


def test_source_image_dirs_classification(tmp_path):
    cfg = load_sources_config()
    src = next(s for s in cfg.sources if s.source_id == "human_detection")
    dirs = list_images._source_image_dirs(src, tmp_path, ["train"])
    assert dirs == [tmp_path / "1"]
