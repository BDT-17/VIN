from types import SimpleNamespace

from sd35_utils import (
    load_person_bboxes_for_crop,
    person_overlap_depth_ok,
)


def test_rejects_front_person_that_is_smaller_than_occluded_person():
    small_front = (120, 270, 170, 390)
    larger_occluded = (140, 230, 220, 410)

    ok, overlap = person_overlap_depth_ok(small_front, larger_occluded)

    assert overlap > 0
    assert ok is False


def test_allows_small_overlap_when_front_person_is_larger_and_lower():
    larger_front = (120, 210, 190, 430)
    smaller_occluded = (187, 300, 237, 390)

    ok, overlap = person_overlap_depth_ok(larger_front, smaller_occluded)

    assert 0 < overlap <= 0.08
    assert ok is True


def test_loads_mot_gt_person_boxes_for_frame(tmp_path):
    sequence = tmp_path / "MOT17-02-FRCNN"
    img_dir = sequence / "img1"
    gt_dir = sequence / "gt"
    img_dir.mkdir(parents=True)
    gt_dir.mkdir()
    image_path = img_dir / "000045.jpg"
    image_path.write_bytes(b"")
    (gt_dir / "gt.txt").write_text(
        "45,1,100,200,40,120,1,1,1\n"
        "46,1,10,20,30,40,1,1,1\n"
        "45,2,1,2,3,4,0,1,1\n",
        encoding="utf-8",
    )
    record = SimpleNamespace(path=image_path, label_path=None)

    bboxes = load_person_bboxes_for_crop(record, original_size=(640, 480), resolution=512)

    assert len(bboxes) == 1
    assert bboxes[0][2] > bboxes[0][0]
    assert bboxes[0][3] > bboxes[0][1]
