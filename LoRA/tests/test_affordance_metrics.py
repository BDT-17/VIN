from PIL import Image, ImageDraw

from sd35_metrics import compute_affordance_score, summarize_manifest_metrics


def _person_mask(bbox, size=(512, 512)):
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(bbox, fill=255)
    return mask


def test_affordance_score_accepts_grounded_reasonable_scale():
    mask = _person_mask((220, 280, 270, 430))
    metrics = compute_affordance_score(
        (512, 512),
        pasted_mask=mask,
        insert_bbox=(220, 280, 270, 430),
        metadata={"ground_y": 430, "expected_height": 150, "final_person_person_overlap_ratio": 0.02},
        variant="add_single_pedestrian",
    )

    assert metrics["affordance_valid"] is True
    assert metrics["placement_score"] >= 0.6
    assert metrics["scale_score"] >= 0.6
    assert metrics["occlusion_score"] >= 0.5


def test_affordance_score_rejects_bad_grounding():
    mask = _person_mask((220, 70, 270, 190))
    metrics = compute_affordance_score(
        (512, 512),
        pasted_mask=mask,
        insert_bbox=(220, 280, 270, 430),
        metadata={"ground_y": 430, "expected_height": 120},
        variant="add_single_pedestrian",
    )

    assert metrics["affordance_valid"] is False
    assert metrics["placement_score"] < 0.6


def test_summarize_manifest_metrics_counts_acceptance_and_means():
    rows = [
        {"accepted": True, "placement_score": 0.8, "scale_score": 0.7, "occlusion_score": 0.9, "affordance_score": 0.78},
        {"accepted": True, "placement_score": 0.6, "scale_score": 0.9, "occlusion_score": 0.8, "affordance_score": 0.76},
        {"accepted": False, "reject_reason": "bad_scale_too_large"},
    ]

    summary = summarize_manifest_metrics(rows, reject_histogram={"bad_scale_too_large": 1})

    assert summary["num_generated"] == 3
    assert summary["num_accepted"] == 2
    assert summary["num_rejected"] == 1
    assert summary["mean_affordance_score"] == 0.77
