import importlib
import sys
import types
from pathlib import Path


LORA_ROOT = Path(__file__).resolve().parents[1]


def import_training(monkeypatch):
    sys.path.insert(0, str(LORA_ROOT))
    sys.modules.pop("sd35_lora_training", None)
    sys.modules.pop("sd35_config", None)
    return importlib.import_module("sd35_lora_training")


def test_export_lora_pt_writes_model_artifact(monkeypatch, tmp_path):
    training = import_training(monkeypatch)
    adapter = tmp_path / "pytorch_lora_weights.bin"
    adapter.write_bytes(b"adapter")
    saved = {}

    torch = types.ModuleType("torch")
    torch.load = lambda path, map_location=None: {"layer.weight": [1, 2, 3]}

    def fake_save(payload, path):
        saved["payload"] = payload
        Path(path).write_bytes(b"pt-model")

    torch.save = fake_save
    monkeypatch.setitem(sys.modules, "torch", torch)

    pt_path = training.export_lora_pt(adapter_path=adapter, output_dir=tmp_path)

    assert pt_path.name == "pytorch_lora_weights.pt"
    assert pt_path.exists()
    assert saved["payload"]["format"] == "sd35_lora_state_dict"
    assert saved["payload"]["source_adapter"] == str(adapter)
    assert saved["payload"]["state_dict"] == {"layer.weight": [1, 2, 3]}


def test_build_training_command_uses_t4_safe_defaults(monkeypatch):
    training = import_training(monkeypatch)

    command = training.build_diffusers_training_command(script_path="train_dreambooth_lora_sd3.py")

    assert "--mixed_precision" in command
    assert command[command.index("--mixed_precision") + 1] == "fp16"
    assert "--train_batch_size" in command
    assert command[command.index("--train_batch_size") + 1] == "1"
    assert "--gradient_checkpointing" in command
    assert "--use_8bit_adam" in command
    assert "--train_text_encoder=false" not in command
