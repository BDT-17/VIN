"""Stage 03/04 — lock eval split + filter + crop.

- Eval lock: images whose (source_id, original_split) is an eval split are tagged
  for the inpaint eval subsystem and EXCLUDED from LoRA candidates.
- Filter: pedestrian instances passing quality thresholds become LoRA candidates.
- Crop: pedestrian bbox + context_ratio context, clamped to crop_min/max size.
  Crops keep surrounding context (NOT transparent cutouts).

Outputs:
    <work>/etl/curated/lora_candidates.parquet
    <work>/etl/crops/<sample_id>.jpg
"""

from pathlib import Path

import pandas as pd

from .config import SourcesConfig
from .schema import PEDESTRIAN


def _eval_lock_lookup(config: SourcesConfig):
    lora_ok, eval_only = set(), set()
    for s in config.sources:
        for sp in (s.lora_splits or []):
            lora_ok.add((s.source_id, sp))
        for sp in (s.eval_splits or []):
            eval_only.add((s.source_id, sp))
    return lora_ok, eval_only


def run_curate(normalized_dir: Path, work_dir: Path, config: SourcesConfig) -> Path:
    normalized_dir = Path(normalized_dir)
    out_dir = Path(work_dir) / "etl" / "curated"
    crops_dir = Path(work_dir) / "etl" / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    images_df = pd.read_parquet(normalized_dir / "images.parquet")
    instances_df = pd.read_parquet(normalized_dir / "instances.parquet")
    qt = config.quality_thresholds
    ctx = config.export_config.crop_context_ratio
    cmin, cmax = config.export_config.crop_min_size, config.export_config.crop_max_size

    lora_ok, eval_only = _eval_lock_lookup(config)
    img_meta = images_df.set_index("image_id")

    candidates = []
    for _, inst in instances_df.iterrows():
        if inst["image_id"] not in img_meta.index:
            continue
        img = img_meta.loc[inst["image_id"]]
        key = (img["source_id"], img["original_split"])
        # eval-locked: never a LoRA candidate
        if key in eval_only:
            continue
        if lora_ok and key not in lora_ok:
            continue

        bbox_h = float(inst["bbox_h"])
        if bbox_h < qt.min_bbox_height_px:
            continue
        visible_ratio = _visible_ratio(inst)
        if visible_ratio < qt.min_visible_ratio:
            continue
        quality = _quality_score(inst, bbox_h, visible_ratio)
        if quality < qt.min_quality_score:
            continue

        sample_id = f"{config.release_name}_{inst['instance_id']}"
        crop_path = crops_dir / f"{sample_id}.jpg"
        cw, ch = _build_crop(Path(img["raw_path"]), inst, ctx, cmin, cmax, crop_path)
        if cw == 0:
            continue

        candidates.append({
            "sample_id": sample_id,
            "image_id": inst["image_id"],
            "instance_id": inst["instance_id"],
            "source_id": img["source_id"],
            "group_id": img["group_id"],
            "crop_path": str(crop_path),
            "crop_width": cw, "crop_height": ch,
            "bbox_height_px": bbox_h,
            "visible_ratio": visible_ratio,
            "quality_score": quality,
        })

    cand_df = pd.DataFrame(candidates)
    cand_df.to_parquet(out_dir / "lora_candidates.parquet", index=False)
    print(f"  curate: {len(cand_df)} LoRA candidate crops "
          f"(eval-locked splits excluded)")
    return out_dir


def _visible_ratio(inst) -> float:
    area = float(inst["bbox_w"]) * float(inst["bbox_h"])
    vis = float(inst.get("visible_bbox_w", inst["bbox_w"])) * float(
        inst.get("visible_bbox_h", inst["bbox_h"]))
    return min(1.0, vis / area) if area > 0 else 0.0


def _quality_score(inst, bbox_h, visible_ratio) -> float:
    # simple, transparent proxy: visibility weighted, size-aware
    occ = inst.get("occlusion_level")
    occ_term = 1.0 if occ is None or (isinstance(occ, float) and occ != occ) else max(0.0, 1.0 - float(occ))
    size_term = min(1.0, bbox_h / 256.0)
    return round(0.5 * visible_ratio + 0.3 * occ_term + 0.2 * size_term, 4)


def _build_crop(image_path: Path, inst, ctx, cmin, cmax, out_path: Path):
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            W, H = im.size
            x, y, w, h = (float(inst["bbox_x"]), float(inst["bbox_y"]),
                          float(inst["bbox_w"]), float(inst["bbox_h"]))
            mx, my = w * ctx, h * ctx
            left = max(0, int(x - mx)); top = max(0, int(y - my))
            right = min(W, int(x + w + mx)); bottom = min(H, int(y + h + my))
            if right - left < 8 or bottom - top < 8:
                return 0, 0
            crop = im.crop((left, top, right, bottom))
            cw, ch = crop.size
            longest = max(cw, ch)
            if longest > cmax:
                scale = cmax / longest
                crop = crop.resize((max(1, int(cw * scale)), max(1, int(ch * scale))))
            cw, ch = crop.size
            if max(cw, ch) < cmin:
                scale = cmin / max(cw, ch)
                crop = crop.resize((max(1, int(cw * scale)), max(1, int(ch * scale))))
            cw, ch = crop.size
            out_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(out_path, quality=95)
            return cw, ch
    except Exception:
        return 0, 0
