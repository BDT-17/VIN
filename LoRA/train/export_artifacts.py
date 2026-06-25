"""Adapter discovery, .pt export, and load-back verification.

Canonical artifact: pytorch_lora_weights.safetensors
The .pt is a secondary handoff artifact.
"""

import json
from pathlib import Path

_ADAPTER_NAMES = ("pytorch_lora_weights.safetensors", "pytorch_lora_weights.bin")


def find_adapter(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    for name in _ADAPTER_NAMES:
        for base in (output_dir / "adapter", output_dir):
            if (base / name).exists():
                return base / name
    matches = sorted(output_dir.rglob("*lora*.safetensors")) + sorted(output_dir.rglob("*lora*.bin"))
    if matches:
        return matches[-1]
    raise FileNotFoundError(f"No LoRA adapter found under {output_dir}")


def export_pt(adapter_path: Path, out_path: Path) -> Path:
    """Re-serialize a .safetensors adapter into a torch .pt state dict."""
    adapter_path, out_path = Path(adapter_path), Path(out_path)
    import torch
    if adapter_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(adapter_path))
    else:
        state = torch.load(str(adapter_path), map_location="cpu", weights_only=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(out_path))
    return out_path


def verify_adapter_loadable(adapter_path: Path) -> dict:
    """Parse the adapter with safetensors; does not need a running pipeline."""
    adapter_path = Path(adapter_path)
    result = {"adapter_path": str(adapter_path), "exists": adapter_path.exists(),
              "key_count": 0, "loadable": False, "error": ""}
    if not adapter_path.exists():
        result["error"] = "adapter file not found"
        return result
    try:
        if adapter_path.suffix == ".safetensors":
            from safetensors import safe_open
            with safe_open(str(adapter_path), framework="pt", device="cpu") as f:
                keys = list(f.keys())
            result["key_count"] = len(keys)
            result["loadable"] = len(keys) > 0
            if not keys:
                result["error"] = "adapter has 0 keys"
        else:
            import torch
            sd = torch.load(str(adapter_path), map_location="cpu", weights_only=True)
            result["key_count"] = len(sd) if hasattr(sd, "__len__") else -1
            result["loadable"] = True
    except Exception as exc:
        result["error"] = str(exc)
    return result


def write_adapter_verification(out_dir: Path, adapter_path: Path) -> dict:
    v = verify_adapter_loadable(adapter_path)
    (Path(out_dir) / "adapter_verification.json").write_text(
        json.dumps(v, indent=2), encoding="utf-8")
    return v
