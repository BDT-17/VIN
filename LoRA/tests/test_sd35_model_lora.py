import importlib
import sys
import types
from pathlib import Path

import pytest


LORA_ROOT = Path(__file__).resolve().parents[1]


class DummyPipe:
    def __init__(self):
        self.load_calls = []
        self.adapter_calls = []
        self.fuse_calls = []

    def load_lora_weights(self, path, **kwargs):
        self.load_calls.append((path, kwargs))

    def set_adapters(self, adapter_names, adapter_weights=None):
        self.adapter_calls.append((adapter_names, adapter_weights))

    def fuse_lora(self, **kwargs):
        self.fuse_calls.append(kwargs)


class DummyPipeline(DummyPipe):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


def install_lightweight_stubs(monkeypatch):
    numpy = types.ModuleType("numpy")
    monkeypatch.setitem(sys.modules, "numpy", numpy)

    matplotlib = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("matplotlib.pyplot")
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)

    pil = types.ModuleType("PIL")
    for name in ("Image", "ImageOps", "ImageDraw", "ImageFilter", "ImageChops"):
        module = types.ModuleType(f"PIL.{name}")
        setattr(pil, name, module)
        monkeypatch.setitem(sys.modules, f"PIL.{name}", module)
    monkeypatch.setitem(sys.modules, "PIL", pil)

    torch = types.ModuleType("torch")
    torch.float16 = "float16"
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        device_count=lambda: 0,
        set_device=lambda *_args, **_kwargs: None,
    )
    torch.device = lambda device: types.SimpleNamespace(index=0 if str(device).startswith("cuda") else None)
    monkeypatch.setitem(sys.modules, "torch", torch)

    sd35_utils = types.ModuleType("sd35_utils")
    sd35_utils.clear_cuda = lambda: None
    monkeypatch.setitem(sys.modules, "sd35_utils", sd35_utils)


def import_sd35_model(monkeypatch):
    install_lightweight_stubs(monkeypatch)

    diffusers = types.ModuleType("diffusers")
    diffusers.StableDiffusion3Img2ImgPipeline = DummyPipeline
    diffusers.StableDiffusion3Pipeline = DummyPipeline
    diffusers.StableDiffusion3InpaintPipeline = DummyPipeline

    diffusers_utils = types.ModuleType("diffusers.utils")
    diffusers_logging = types.SimpleNamespace(set_verbosity_error=lambda: None)
    diffusers_utils.logging = diffusers_logging

    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.utils", diffusers_utils)
    sys.path.insert(0, str(LORA_ROOT))
    for name in ("sd35_model", "sd35_config", "sd35_utils"):
        sys.modules.pop(name, None)
    return importlib.import_module("sd35_model")


def configure_lora(model, *, enabled=True, fuse=False):
    model.LORA_ENABLED = enabled
    model.LORA_PATH = Path("/tmp/citypersons-lora")
    model.LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"
    model.LORA_ADAPTER_NAME = "citypersons_lora"
    model.LORA_SCALE = 0.7
    model.LORA_FUSE = fuse


def test_apply_lora_adapter_noops_when_disabled(monkeypatch):
    model = import_sd35_model(monkeypatch)
    pipe = DummyPipe()
    configure_lora(model, enabled=False)

    assert model.apply_lora_adapter(pipe) is pipe
    assert pipe.load_calls == []
    assert pipe.adapter_calls == []


def test_apply_lora_adapter_loads_named_adapter(monkeypatch):
    model = import_sd35_model(monkeypatch)
    pipe = DummyPipe()
    configure_lora(model)

    assert model.apply_lora_adapter(pipe) is pipe
    assert pipe.load_calls == [
        (
            str(Path("/tmp/citypersons-lora")),
            {
                "adapter_name": "citypersons_lora",
                "weight_name": "pytorch_lora_weights.safetensors",
            },
        )
    ]
    assert pipe.adapter_calls == [(["citypersons_lora"], [0.7])]


def test_apply_lora_adapter_requires_diffusers_lora_api(monkeypatch):
    model = import_sd35_model(monkeypatch)
    configure_lora(model)

    with pytest.raises(RuntimeError, match="load_lora_weights"):
        model.apply_lora_adapter(object())
