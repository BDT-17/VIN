"""Build the frozen SD3.5 inpaint evaluation set from eval-locked source splits.

For every pedestrian instance in an eval-locked split (e.g. CityPersons valid),
emit a case: original image, a white mask over the (dilated) pedestrian bbox,
and a reference copy. Cases are partitioned group-disjoint into:
    inpaint_eval_v1        (dev/tuning)
    final_inpaint_test_v1  (touch once)

Output:
    <work>/eval/inpaint_eval_v1/{cases.jsonl,images/,masks/,reference/}
    <work>/eval/final_inpaint_test_v1/{...}
    <work>/eval/eval_manifest.parquet   (image_id, group_id, eval_set)
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SourcesConfig, EvalConfig
from .schema import EVAL_SET_DEV, EVAL_SET_FINAL


def _eval_lock_keys(config: SourcesConfig):
    keys = set()
    for s in config.sources:
        for sp in (s.eval_splits or []):
            keys.add((s.source_id, sp))
    return keys


def _mask_dilate_px(bbox_w, bbox_h, ratio=0.08):
    return int(max(bbox_w, bbox_h) * ratio)


def run_build_eval_cases(normalized_dir: Path, work_dir: Path, config: SourcesConfig) -> Path:
    normalized_dir = Path(normalized_dir)
    eval_root = Path(work_dir) / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    images_df = pd.read_parquet(normalized_dir / "images.parquet")
    instances_df = pd.read_parquet(normalized_dir / "instances.parquet")
    keys = _eval_lock_keys(config)
    if not keys:
        print("  build_eval_cases: no eval-locked splits configured")
        return eval_root

    img_meta = images_df.set_index("image_id")
    eval_imgs = images_df[
        images_df.apply(lambda r: (r["source_id"], r["original_split"]) in keys, axis=1)
    ].copy()
    if eval_imgs.empty:
        print("  build_eval_cases: no images in eval-locked splits (check mounts)")
        return eval_root

    # group-disjoint partition into dev vs final
    ec: EvalConfig = config.eval_config
    groups = eval_imgs["group_id"].dropna().unique().tolist()
    rng = np.random.RandomState(ec.eval_seed)
    rng.shuffle(groups)
    dev_end = int(len(groups) * ec.eval_v1_ratio)
    group_set = {g: (EVAL_SET_DEV if i < dev_end else EVAL_SET_FINAL) for i, g in enumerate(groups)}

    counts = {EVAL_SET_DEV: 0, EVAL_SET_FINAL: 0}
    manifest_rows = []
    cases_by_set = {EVAL_SET_DEV: [], EVAL_SET_FINAL: []}

    for _, inst in instances_df.iterrows():
        iid = inst["image_id"]
        if iid not in img_meta.index:
            continue
        img = img_meta.loc[iid]
        if (img["source_id"], img["original_split"]) not in keys:
            continue
        eval_set = group_set.get(img["group_id"], EVAL_SET_DEV)
        set_dir = eval_root / eval_set
        for sub in ("images", "masks", "reference"):
            (set_dir / sub).mkdir(parents=True, exist_ok=True)

        case_id = f"{img['source_id']}_{img['original_split']}_{inst['instance_id']}"
        ok, bbox_xyxy = _emit_case(Path(img["raw_path"]), inst, set_dir, case_id)
        if not ok:
            continue
        counts[eval_set] += 1
        cases_by_set[eval_set].append({
            "case_id": case_id,
            "image_path": f"images/{case_id}.png",
            "mask_path": f"masks/{case_id}.png",
            "reference_path": f"reference/{case_id}.png",
            "expected_bbox_xyxy": bbox_xyxy,
            "prompt_fields": {"pose": "walking", "view": "side view", "scene": "urban street scene"},
            "source_split": img["original_split"],
            "eval_set": eval_set,
            "frozen": True,
        })
        manifest_rows.append({"image_id": iid, "group_id": img["group_id"], "eval_set": eval_set})

    for eval_set, cases in cases_by_set.items():
        if not cases:
            continue
        lines = "\n".join(json.dumps(c, ensure_ascii=False) for c in cases)
        (eval_root / eval_set / "cases.jsonl").write_text(lines + "\n", encoding="utf-8")

    pd.DataFrame(manifest_rows).to_parquet(eval_root / "eval_manifest.parquet", index=False)
    print(f"  build_eval_cases: {counts[EVAL_SET_DEV]} {EVAL_SET_DEV}, "
          f"{counts[EVAL_SET_FINAL]} {EVAL_SET_FINAL}")
    return eval_root


def _emit_case(image_path: Path, inst, set_dir: Path, case_id: str):
    try:
        from PIL import Image, ImageDraw
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            W, H = im.size
            x, y, w, h = (float(inst["bbox_x"]), float(inst["bbox_y"]),
                          float(inst["bbox_w"]), float(inst["bbox_h"]))
            d = _mask_dilate_px(w, h)
            x1, y1 = max(0, int(x - d)), max(0, int(y - d))
            x2, y2 = min(W, int(x + w + d)), min(H, int(y + h + d))
            if x2 - x1 < 4 or y2 - y1 < 4:
                return False, []
            mask = Image.new("L", (W, H), 0)
            ImageDraw.Draw(mask).rectangle([x1, y1, x2, y2], fill=255)
            im.save(set_dir / "images" / f"{case_id}.png")
            im.save(set_dir / "reference" / f"{case_id}.png")
            mask.save(set_dir / "masks" / f"{case_id}.png")
            return True, [float(x), float(y), float(x + w), float(y + h)]
    except Exception:
        return False, []
