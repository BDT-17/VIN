"""Concept-LoRA caption builder: imperative stripping + trigger/prefix (no HF)."""

from LoRA.train.concept_lora_dataset import instruction_to_subject, build_caption


def test_strips_leading_imperative():
    assert instruction_to_subject("add a man wearing a hat") == "a man wearing a hat"
    assert instruction_to_subject("Place the woman on the left") == "the woman on the left"
    assert instruction_to_subject("put a child near the car") == "a child near the car"
    assert instruction_to_subject("insert in a pedestrian") == "a pedestrian"


def test_empty_instruction_falls_back():
    assert instruction_to_subject("") == "a person"
    assert instruction_to_subject("   ") == "a person"
    assert instruction_to_subject("add ") == "a person"   # bare verb, no object


def test_non_imperative_left_unchanged():
    # already a description -> untouched
    assert instruction_to_subject("a woman in a red dress") == "a woman in a red dress"


def test_build_caption_prefix_only():
    assert build_caption("add a person", caption_prefix="a photo of ") == "a photo of a person"


def test_build_caption_with_trigger():
    cap = build_caption("add a man on a bike", caption_prefix="a photo of ",
                        trigger_token="<vin_ped>")
    assert cap == "a photo of <vin_ped> a man on a bike"


def test_build_caption_no_prefix_no_trigger():
    assert build_caption("add a person", caption_prefix="", trigger_token="") == "a person"
