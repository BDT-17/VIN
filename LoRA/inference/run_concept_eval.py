"""Standalone concept-LoRA eval entrypoint — run in a FRESH process.

Training + a second full-pipeline load in the same kernel exhausts RAM, so eval
runs as its own process. Loads base + LoRA and generates one image per validation
prompt, then tiles them into a contact sheet.

Usage:
    python -m LoRA.inference.run_concept_eval --run-dir RUN --base-model SD35 \
        [--steps 28] [--guidance 7.0] [--seed 42]

Validation prompts come from the run's training_config.json (validation_prompts),
falling back to a default person prompt. Writes images + contact_sheet.png under
RUN/concept_eval/.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from PIL import Image
    from LoRA.inference.sd35_concept_runner import load_concept_runner_from_run

    run_dir = Path(args.run_dir)
    out = Path(args.out_dir) if args.out_dir else run_dir / "concept_eval"
    (out / "images").mkdir(parents=True, exist_ok=True)

    cfg = json.loads((run_dir / "training_config.json").read_text(encoding="utf-8"))
    prompts = cfg.get("validation_prompts") or ["a photo of a person walking on a city street"]
    print(f"[eval] {len(prompts)} validation prompts")

    runner, prov = load_concept_runner_from_run(run_dir, base_model_id=args.base_model, hf_token=args.hf_token)
    assert not prov.get("requires_input_proj"), "concept flow must not require input_proj"
    print("[eval] concept adapter loaded")

    import time
    paths = []
    for i, p in enumerate(prompts):
        te = time.time()
        img = runner.generate(p, seed=args.seed + i, num_inference_steps=args.steps,
                              guidance_scale=args.guidance, resolution=args.resolution)
        ip = out / "images" / f"sample_{i:02d}.png"
        img.save(ip)
        paths.append(ip)
        print(f"[eval] {i+1}/{len(prompts)}  {time.time()-te:.1f}s  {p}", flush=True)

    # contact sheet: one column per generated sample
    tiles = [Image.open(p).convert("RGB").resize((256, 256)) for p in paths]
    n = max(1, len(tiles))
    sheet = Image.new("RGB", (256 * n, 256))
    for i, t in enumerate(tiles):
        sheet.paste(t, (256 * i, 0))
    sheet.save(out / "contact_sheet.png")
    print(f"[eval] contact sheet -> {out/'contact_sheet.png'}")


if __name__ == "__main__":
    main()
