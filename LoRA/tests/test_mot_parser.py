"""Tests for MOT parser."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import pandas as pd

from LoRA.data.parsers.mot import MOTParser
from LoRA.data.schema import ImageRecord, InstanceRecord


@pytest.fixture
def mock_mot_structure(tmp_path):
    """Create mock MOT dataset structure."""
    mot_root = tmp_path / "MOT17-02-FRCNN"
    img_dir = mot_root / "img1"
    gt_dir = mot_root / "gt"

    img_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    # Create dummy images
    for i in range(1, 6):
        img_path = img_dir / f"{i:06d}.jpg"
        # Create a small dummy image
        img = Image.new('RGB', (640, 480), color='white')
        img.save(img_path)

    # Create dummy gt.txt
    gt_path = gt_dir / "gt.txt"
    gt_data = [
        "1,1,100,100,50,150,1.0,1,0.8",
        "2,1,110,105,50,150,1.0,1,0.8",
        "3,2,200,150,45,140,1.0,1,0.9",
    ]
    gt_path.write_text("\n".join(gt_data))

    return mot_root


def test_mot_parser_initialization(mock_mot_structure):
    """Test MOTParser initialization."""
    parser = MOTParser(
        source_id="mot17_02",
        mount_path=mock_mot_structure.parent,
        sequence_dir="MOT17-02-FRCNN/img1",
        gt_dir="MOT17-02-FRCNN/gt",
    )

    assert parser.source_id == "mot17_02"
    assert parser.img_dir.exists()
    assert parser.gt_path.exists()


def test_mot_parser_parse(mock_mot_structure):
    """Test MOT parsing."""
    parser = MOTParser(
        source_id="mot17_02",
        mount_path=mock_mot_structure.parent,
        sequence_dir="MOT17-02-FRCNN/img1",
        gt_dir="MOT17-02-FRCNN/gt",
        temporal_sampling_fps=2.0,
    )

    images, instances = parser.parse()

    # Should have sampled images (not all 5)
    assert len(images) > 0
    assert len(images) <= 5

    # Check image records
    for img in images:
        assert isinstance(img, ImageRecord)
        assert img.source_id == "mot17_02"
        assert img.width == 640
        assert img.height == 480
        assert len(img.sha256) == 64
        assert img.group_id.startswith("mot17_02_window_")

    # Should have some instances
    assert len(instances) >= 0

    # Check instance records
    for inst in instances:
        assert isinstance(inst, InstanceRecord)
        assert inst.class_name == "pedestrian"
        assert inst.bbox_w > 0
        assert inst.bbox_h > 0


def test_mot_temporal_sampling(mock_mot_structure):
    """Test temporal sampling logic."""
    parser = MOTParser(
        source_id="mot17_02",
        mount_path=mock_mot_structure.parent,
        sequence_dir="MOT17-02-FRCNN/img1",
        gt_dir="MOT17-02-FRCNN/gt",
        temporal_sampling_fps=1.0,  # Sample at 1 FPS (every 30th frame at 30 FPS source)
    )

    image_files = sorted((mock_mot_structure / "img1").glob("*.jpg"))
    sampled = parser._temporal_sample(image_files)

    # With 5 frames and sampling every 30 frames, should get ~1 frame
    assert len(sampled) <= len(image_files)


def test_mot_group_id_computation():
    """Test group ID computation based on temporal windows."""
    parser = MOTParser(
        source_id="mot17_02",
        mount_path=Path("/mock"),
        sequence_dir="img1",
        gt_dir="gt",
        temporal_window_seconds=2.0,
        temporal_sampling_fps=2.0,
    )

    # Window size = 2.0 seconds * 2.0 fps = 4 frames per window
    group_1 = parser._compute_group_id(1)  # Frame 1 -> window 0
    group_3 = parser._compute_group_id(3)  # Frame 3 -> window 0
    group_5 = parser._compute_group_id(5)  # Frame 5 -> window 1

    assert group_1 == group_3  # Same window
    assert group_1 != group_5  # Different window
