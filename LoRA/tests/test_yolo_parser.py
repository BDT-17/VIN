"""Tests for YOLO parser."""

import pytest
from pathlib import Path
from PIL import Image

from LoRA.data.parsers.yolo import YOLOParser
from LoRA.data.schema import ImageRecord, InstanceRecord


@pytest.fixture
def mock_yolo_structure(tmp_path):
    """Create mock YOLO dataset structure."""
    dataset_root = tmp_path / "citypersons_yolo"

    # Create train split
    train_img_dir = dataset_root / "train" / "images"
    train_label_dir = dataset_root / "train" / "labels"
    train_img_dir.mkdir(parents=True)
    train_label_dir.mkdir(parents=True)

    # Create dummy images and labels
    for i in range(3):
        img_path = train_img_dir / f"aachen_000000_{i:06d}_leftImg8bit.jpg"
        img = Image.new('RGB', (1024, 512), color='gray')
        img.save(img_path)

        # Create corresponding label
        label_path = train_label_dir / f"aachen_000000_{i:06d}_leftImg8bit.txt"
        # YOLO format: class_id center_x center_y width height (normalized)
        label_data = [
            "0 0.5 0.7 0.05 0.15",  # pedestrian at center-bottom
            "0 0.3 0.65 0.04 0.12",  # another pedestrian
        ]
        label_path.write_text("\n".join(label_data))

    return dataset_root


def test_yolo_parser_initialization(mock_yolo_structure):
    """Test YOLOParser initialization."""
    parser = YOLOParser(
        source_id="citypersons",
        mount_path=mock_yolo_structure,
        splits={"train": "train/images"},
        label_dirs={"train": "train/labels"},
    )

    assert parser.source_id == "citypersons"
    assert parser.class_mapping[0] == "pedestrian"


def test_yolo_parser_parse(mock_yolo_structure):
    """Test YOLO parsing."""
    parser = YOLOParser(
        source_id="citypersons",
        mount_path=mock_yolo_structure,
        splits={"train": "train/images"},
        label_dirs={"train": "train/labels"},
    )

    images, instances = parser.parse()

    # Should have 3 images
    assert len(images) == 3

    # Check image records
    for img in images:
        assert isinstance(img, ImageRecord)
        assert img.source_id == "citypersons"
        assert img.width == 1024
        assert img.height == 512
        assert img.camera_domain == "citypersons_surveillance"

    # Should have 6 instances (2 per image * 3 images)
    assert len(instances) == 6

    # Check instance records
    for inst in instances:
        assert isinstance(inst, InstanceRecord)
        assert inst.class_name == "pedestrian"
        assert 0 <= inst.bbox_x < 1024
        assert 0 <= inst.bbox_y < 512
        assert inst.bbox_w > 0
        assert inst.bbox_h > 0


def test_yolo_coordinate_conversion(mock_yolo_structure):
    """Test YOLO normalized coordinate conversion to absolute pixels."""
    parser = YOLOParser(
        source_id="citypersons",
        mount_path=mock_yolo_structure,
        splits={"train": "train/images"},
        label_dirs={"train": "train/labels"},
    )

    images, instances = parser.parse()

    # Find a specific instance
    inst = instances[0]

    # YOLO label: "0 0.5 0.7 0.05 0.15"
    # Image size: 1024 x 512
    # Expected absolute bbox:
    #   width = 0.05 * 1024 = 51.2
    #   height = 0.15 * 512 = 76.8
    #   center_x = 0.5 * 1024 = 512
    #   center_y = 0.7 * 512 = 358.4
    #   top_left_x = 512 - 51.2/2 = 486.4
    #   top_left_y = 358.4 - 76.8/2 = 320

    # Check that conversion is approximately correct
    assert inst.bbox_w > 40
    assert inst.bbox_w < 60
    assert inst.bbox_h > 70
    assert inst.bbox_h < 85


def test_yolo_group_id_computation(mock_yolo_structure):
    """Test group ID computation for CityPersons-style filenames."""
    parser = YOLOParser(
        source_id="citypersons",
        mount_path=mock_yolo_structure,
        splits={"train": "train/images"},
        label_dirs={"train": "train/labels"},
    )

    images, _ = parser.parse()

    # All images from "aachen_000000" should have the same group
    group_ids = [img.group_id for img in images]

    # All should start with same prefix (same scene)
    assert all(gid.startswith("citypersons_train_aachen_000000") for gid in group_ids)
