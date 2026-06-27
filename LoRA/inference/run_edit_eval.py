"""Standalone edit-eval entrypoint — run in a FRESH process.

Training + a second full-pipeline load in the same kernel exhausts system RAM
(SD3.5 is ~10GB; the trainer's leftover RAM + a reload OOMs the kernel). Running
eval as its own process starts from a clean slate, so the only resident model is
the inference pipeline.

Usage (from a notebook cell, after training has saved RUN_DIR):
    !python -m LoRA.inference.run_edit_eval --run-dir RUN_DIR --eval-dir EVAL \
        --base-model SD35_MODEL [--limit 12] [--steps 30]

Writes images, edit_metrics.csv, and contact_sheet.png under RUN/runs/edit_eval/.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="training run dir (has adapter/, training_provenance.json)")
    ap.add_argument("--eval-dir", required=True, help="PIPE eval set dir (has cases.jsonl)")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None, help="cap number of eval cases")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--detector-weights", default="yolov8n.pt",
                    help="YOLO weights for person metrics; on Kaggle point at a dataset mount")
    ap.add_argument("--no-detector", action="store_true",
                    help="skip person metrics (background/seam only)")
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # ensure repo root on path when invoked as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from PIL import Image
    from LoRA.inference.sd35_edit_runner import load_edit_runner_from_run
    from LoRA.inference.inpaint_metrics import compute_case_metrics

    run_dir = Path(args.run_dir)
    eval_dir = Path(args.eval_dir)
    out = Path(args.out_dir) if args.out_dir else run_dir / "edit_eval"
    (out / "images").mkdir(parents=True, exist_ok=True)

    cases = [json.loads(l) for l in (eval_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit is not None:
        cases = cases[:args.limit]
    print(f"[eval] {len(cases)} cases")

    runner, prov = load_edit_runner_from_run(run_dir, base_model_id=args.base_model, hf_token=args.hf_token)
    assert prov.get("requires_input_adapter")
    print("[eval] adapter loaded")

    detector = None
    if not args.no_detector:
        from LoRA.inference.person_detector import maybe_load_person_detector
        detector = maybe_load_person_detector(weights=args.detector_weights)
        if detector is not None:
            print("[eval] person detector loaded")

    def case_prompt(c):
        return "a photo of <vin_ped> pedestrian, " + (c["prompt_fields"].get("instruction", "") or "a person")

    runner.precompute_embeds([case_prompt(c) for c in cases])

    import time
    rows = []
    for i, c in enumerate(cases):
        src = Image.open(eval_dir / c["image_path"]); msk = Image.open(eval_dir / c["mask_path"])
        te = time.time()
        img = runner.edit(src, msk, case_prompt(c), seed=args.seed, num_inference_steps=args.steps)
        p = out / "images" / f"{c['case_id']}.png"; img.save(p)
        m = compute_case_metrics(eval_dir / c["reference_path"], p, eval_dir / c["mask_path"],
                                 c["expected_bbox_xyxy"], detector=detector)
        m["case_id"] = c["case_id"]; rows.append(m)
        print(f"[eval] {i+1}/{len(cases)} {c['case_id']}  {time.time()-te:.1f}s  "
              f"det={m.get('person_detected')} inside_mask={m.get('person_inside_mask_ratio')} "
              f"scale={m.get('scale_ratio')}", flush=True)
    print(f"[eval] generated {len(rows)} edits")

    if rows:
        with open(out / "edit_metrics.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader(); w.writerows(rows)

    # contact sheet: source | mask | edit
    strips = []
    for c in cases:
        s_ = Image.open(eval_dir / c["image_path"]).convert("RGB").resize((256, 256))
        mk = Image.open(eval_dir / c["mask_path"]).convert("RGB").resize((256, 256))
        rs = Image.open(out / "images" / f"{c['case_id']}.png").convert("RGB").resize((256, 256))
        strip = Image.new("RGB", (768, 256)); strip.paste(s_, (0, 0)); strip.paste(mk, (256, 0)); strip.paste(rs, (512, 0))
        strips.append(strip)
    sheet = Image.new("RGB", (768, 256 * max(1, len(strips))))
    for i, s_ in enumerate(strips):
        sheet.paste(s_, (0, 256 * i))
    sheet.save(out / "contact_sheet.png")
    print(f"[eval] contact sheet -> {out/'contact_sheet.png'}")


if __name__ == "__main__":
    main()
