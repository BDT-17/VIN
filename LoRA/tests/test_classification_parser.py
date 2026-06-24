"""Tests for Classification parser."""

import pytest
from pathlib import Path
from PIL import Image

from LoRA.data.parsers.classification import ClassificationParser
from LoRA.data.schema import ImageRecord


@pytest.fixture
def mock_classification_structure(tmp_path):
    """Create mock classification dataset structure."""
    dataset_root = tmp_path / "human_detection"

    # Create positive folder
    pos_dir = dataset_root / "1"
    pos_dir.mkdir(parents=True)

    for i in range(5):
        img_path = pos_dir / f"positive_{i:04d}.jpg"
        img = Image.new('RGB', (640, 480), color='blue')
        img.save(img_path)

    # Create negative folder
    neg_dir = dataset_root / "0"
    neg_dir.mkdir(parents=True)

    for i in range(3):
        img_path = neg_dir / f"negative_{i:04d}.jpg"
        img = Image.new('RGB', (640, 480), color='red')
        img.save(img_path)

    return dataset_root


def test_classification_parser_initialization(mock_classification_structure):
    """Test ClassificationParser initialization."""
    parser = ClassificationParser(
        source_id="human_detection",
        mount_path=mock_classification_structure,
        positive_dir="1",
        negative_dir="0",
    )

    assert parser.source_id == "human_detection"
    assert parser.positive_dir == "1"
    assert parser.negative_dir == "0"


def test_classification_parser_parse(mock_classification_structure):
    """Test classification parsing."""
    parser = ClassificationParser(
        source_id="human_detection",
        mount_path=mock_classification_structure,
        positive_dir="1",
        negative_dir="0",
    )

    images, instances = parser.parse()

    # Should have 8 images (5 positive + 3 negative)
    assert len(images) == 8

    # Instances should be EMPTY (pseudo-labeling required)
    assert len(instances) == 0

    # Check image records
    positive_images = [img for img in images if "positive" in img.image_id]
    negative_images = [img for img in images if "negative" in img.image_id]

    assert len(positive_images) == 5
    assert len(negative_images) == 3

    for img in images:
        assert isinstance(img, ImageRecord)
        assert img.source_id == "human_detection"
        assert img.width == 640
        assert img.height == 480
        assert len(img.sha256) == 64
        assert img.camera_domain == "human_detection_cctv"


def test_classification_split_assignment(mock_classification_structure):
    """Test that original_split is set to folder label."""
    parser = ClassificationParser(
        source_id="human_detection",
        mount_path=mock_classification_structure,
        positive_dir="1",
        negative_dir="0",
    )

    images, _ = parser.parse()

    positive_images = [img for img in images if "positive" in img.image_id]
    negative_images = [img for img in images if "negative" in img.image_id]

    # Check split assignment
    for img in positive_images:
        assert img.original_split == "positive"

    for img in negative_images:
        assert img.original_split == "negative"
