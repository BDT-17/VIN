"""D1 tests for the inpaint-edit trainer — the parts that need no GPU/SD3.5."""

import pytest

torch = pytest.importorskip("torch")  # InputAdapter needs torch; skip if absent

from LoRA.train.spike_inpaint_edit import InputAdapter
from LoRA.data.config import load_train_config


def test_input_adapter_shapes_and_identity_init():
    """33->16; identity-init means cond starts == the noisy latent."""
    ad = InputAdapter()
    noisy = torch.randn(1, 16, 8, 8)
    source = torch.randn(1, 16, 8, 8)
    mask = torch.randn(1, 1, 8, 8)
    out = ad(noisy, source, mask)
    assert out.shape == (1, 16, 8, 8)
    # zero-init weights except identity on the noisy channels -> out == noisy
    assert torch.allclose(out, noisy, atol=1e-5)


def test_input_adapter_state_dict_roundtrip(tmp_path):
    ad = InputAdapter()
    with torch.no_grad():
        ad.proj.weight.add_(0.123)  # perturb so it's not the init
    p = tmp_path / "input_adapter.pt"
    torch.save(ad.state_dict(), p)

    ad2 = InputAdapter()
    ad2.load_state_dict(torch.load(p, weights_only=True))
    x = (torch.randn(1, 16, 4, 4), torch.randn(1, 16, 4, 4), torch.randn(1, 1, 4, 4))
    assert torch.allclose(ad(*x), ad2(*x), atol=1e-6)


def test_edit_train_config_loads():
    cfg = load_train_config(
        __import__("pathlib").Path(__file__).resolve().parent.parent
        / "configs" / "inpaint_edit_train.yaml")
    assert cfg["model_name"] == "vinped_sd35m_edit_v1"
    assert cfg["weighting_scheme"] in ("logit_normal", "mode", "cosmap", "none")
    assert cfg["num_train_samples"] > 0
    assert cfg["rank"] >= 1


def test_pipe_pair_prompt_has_trigger(monkeypatch):
    """iter_pipe_pairs builds prompts with the trigger token, without hitting HF."""
    from LoRA.train import inpaint_edit_dataset as ds
    import numpy as np
    from PIL import Image

    src = Image.fromarray(np.full((32, 32, 3), 100, np.uint8))
    tgt = np.array(src); tgt[8:24, 8:24] = 200
    tgt = Image.fromarray(tgt)

    fake_row = {"Instruction_VLM-LLM": "a person walking", "Instruction_Class": "person",
                "source_img": src, "target_img": tgt}

    def fake_load_dataset(*a, **k):
        return [fake_row, fake_row]
    monkeypatch.setattr(ds, "load_dataset", fake_load_dataset, raising=False)
    # inject the symbol the function imports locally
    import datasets as _d
    monkeypatch.setattr(_d, "load_dataset", fake_load_dataset)

    items = list(ds.iter_pipe_pairs(num_samples=1, dilate_px=0, min_mask_pixels=1))
    assert len(items) == 1
    assert "<vin_ped>" in items[0]["prompt"]
    assert items[0]["mask"].size == src.size
