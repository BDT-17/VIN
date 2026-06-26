"""Standalone mask-free edit-eval entrypoint — run in a FRESH process.

Training + a second full-pipeline load in the same kernel exhausts RAM, so eval
runs as its own process. Loads base + LoRA + input_proj, runs the mask-free edit
(IP2P-style CFG) on the PIPE eval set, and writes a contact sheet comparing the
edit against the REAL PIPE target (ground truth).

Usage:
    python -m LoRA.inference.run_maskfree_eval --run-dir RUN --eval-dir EVAL \
        --base-model SD35 [--steps 30] [--s-image 1.5] [--s-text 7.5]

Writes images + contact_sheet.png under RUN/maskfree_eval/. The contact sheet is
source | edit | target (no mask column — this flow is mask-free).
"""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None, help="cap number of eval cases")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--s-image", type=float, default=1.5)
    ap.add_argument("--s-text", type=float, default=7.5)
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from PIL import Image
    from LoRA.inference.sd35_maskfree_runner import load_maskfree_runner_from_run

    run_dir = Path(args.run_dir)
    eval_dir = Path(args.eval_dir)
    out = Path(args.out_dir) if args.out_dir else run_dir / "maskfree_eval"
    (out / "images").mkdir(parents=True, exist_ok=True)

    cases = [json.loads(l) for l in (eval_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit is not None:
        cases = cases[:args.limit]
    print(f"[eval] {len(cases)} cases")

    runner, prov = load_maskfree_runner_from_run(run_dir, base_model_id=args.base_model, hf_token=args.hf_token)
    assert prov.get("requires_input_proj"), "provenance missing input-proj flag"
    print("[eval] mask-free adapter loaded")

    def case_prompt(c):
        return (c["prompt_fields"].get("instruction", "") or "add a person").strip()

    runner.precompute_embeds([case_prompt(c) for c in cases])

    import time
    for i, c in enumerate(cases):
        # mask-free uses the object-erased source as input (PIPE 'image_path')
        src = Image.open(eval_dir / c["image_path"]).convert("RGB")
        te = time.time()
        img = runner.edit(src, case_prompt(c), seed=args.seed,
                          num_inference_steps=args.steps,
                          s_image=args.s_image, s_text=args.s_text)
        img.save(out / "images" / f"{c['case_id']}.png")
        print(f"[eval] {i+1}/{len(cases)} {c['case_id']}  {time.time()-te:.1f}s", flush=True)
    print(f"[eval] generated {len(cases)} edits")

    # contact sheet: source | edit | target(real)
    strips = []
    for c in cases:
        s_ = Image.open(eval_dir / c["image_path"]).convert("RGB").resize((256, 256))
        rs = Image.open(out / "images" / f"{c['case_id']}.png").convert("RGB").resize((256, 256))
        tg = Image.open(eval_dir / c["reference_path"]).convert("RGB").resize((256, 256))
        strip = Image.new("RGB", (768, 256))
        strip.paste(s_, (0, 0)); strip.paste(rs, (256, 0)); strip.paste(tg, (512, 0))
        strips.append(strip)
    sheet = Image.new("RGB", (768, 256 * max(1, len(strips))))
    for i, s_ in enumerate(strips):
        sheet.paste(s_, (0, 256 * i))
    sheet.save(out / "contact_sheet.png")
    print(f"[eval] contact sheet (source | edit | target) -> {out/'contact_sheet.png'}")


if __name__ == "__main__":
    main()
