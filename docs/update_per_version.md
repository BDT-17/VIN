# BÁO CÁO SO SÁNH CÁC PHIÊN BẢN PIPELINE SD3.5 CITYPERSONS AUGMENTATION


---

## TÓM TẮT EXECUTIVE

Dự án "Evaluation-Guided SD3.5 CityPersons Augmentation" đã trải qua **5 phiên bản pipeline chính** (V1 → V5) trong quá trình phát triển. Mỗi phiên bản đại diện cho một bước tiến trong việc giải quyết bài toán cốt lõi: **thêm object vào ảnh mà giữ nguyên background gốc**. Báo cáo này phân tích chi tiết từng phiên bản, nguyên nhân chuyển đổi, lỗi gặp phải, và insight kỹ thuật rút ra.

**Kết luận tổng quan:**
- V1 (Full Image Img2Img): Pipeline cơ bản, đơn giản nhưng phá background nặng
- V2 (Patch Blend): Giữ background tốt hơn nhưng khó kiểm soát tỉ lệ người
- V3 (Human Mask Inpaint): Chuyển sang inpainting chuẩn, nhưng mask hình người quá hẹp
- V4 (BBox Inpaint): Mask rộng hơn, sinh người rõ hơn nhưng vẫn còn vấn đề scale
- V5 (Context Person Composite): Phiên bản hiện tại - tách biệt "sinh người" và "giữ background" bằng YOLO-seg segmentation

---

## 1 BỐI CẢNH DỰ ÁN VÀ MỤC TIÊU

### 1.1 Bài toán nghiên cứu

**Tên dự án:** Evaluation-Guided SD3.5 CityPersons Augmentation Pipeline
**Mục tiêu:** Tạo ảnh augmentation cho dataset CityPersons bằng Stable Diffusion 3.5 Medium, tập trung vào việc thêm pedestrian (người đi bộ) vào ảnh đô thị nhưng giữ background gần như nguyên vẹn.

**Ràng buộc kỹ thuật chính:**
- Chạy trên Kaggle T4 x2 (16GB VRAM mỗi GPU)
- Không load cùng lúc SD3.5 + YOLO + SmolVLM (memory constraint)
- Chạy sequential: generate → unload SD3.5 → load evaluator → evaluate
- Ưu tiên baseline SD3.5 + prompt/parameter tuning trước khi fine-tune LoRA

### 1.2 Dataset

Dataset CityPersons/Roboflow YOLO format:
```
Dataset/
  train/images  7025 ảnh
  train/labels  7025 labels
  valid/images  289 ảnh
  valid/labels  289 labels
  test/images   294 ảnh
  test/labels   294 labels
  data.yaml
```

### 1.3 Quyết định chiến lược quan trọng

**Quyết định #1: Baseline-first, LoRA-only-if-needed**
Thay vì fine-tune LoRA ngay, team quyết định optimize prompt + parameter + compositing của SD3.5 base trước. Chỉ train LoRA nếu baseline fail có hệ thống.

**Quyết định #2: Không dùng loss function từ evaluator**
Evaluator (YOLO, SSIM, LPIPS) dùng để rank/filter ảnh, không đưa trực tiếp vào loss function training vì:
- Detector/CLIP/SSIM thường không differentiable qua diffusion generation
- Dễ bị "hack reward" - ảnh score cao nhưng thực tế xấu
- Tốn GPU nếu giữ computation graph qua denoise/decode

---

## 2 PHIÊN BẢN V1 - FULL IMAGE IMG2IMG

### 2.1 Kiến trúc V1

```
CityPersons Image
→ Resize center crop 448x448
→ SD3.5 Img2Img toàn ảnh
→ Save augmented image
→ Lưu manifest.csv
```

**Core pipeline:**
```python
image = pipe(
    image=source,
    prompt=prompt,
    strength=0.54,
    guidance_scale=7.6,
    num_inference_steps=36,
).images[0]
image.save(output_path)
```

### 2.2 Prompt V1

```python
BASE_CAPTION = "CityPersons urban street scene, fixed camera, preserved road geometry"

VARIANT_PROMPTS = {
    "add_single_pedestrian": "add one realistic pedestrian, correct scale, grounded",
    "add_two_pedestrians": "add two realistic pedestrians, correct scale, grounded",
    "add_small_group": "add a small pedestrian group, correct scale, grounded",
    "add_occluded_pedestrian": "add one partially occluded pedestrian, correct scale",
    "add_distant_pedestrian": "add one small distant pedestrian, correct perspective",
    "add_near_pedestrian": "add one foreground pedestrian, correct scale, grounded"
}

NEGATIVE_PROMPT = (
    "changed layout, changed camera, wrong scale, floating person, bad anatomy, "
    "extra limbs, warped body, no ground contact, text, watermark, cartoon, blur, artifacts"
)
```

### 2.3 Parameters V1

```python
VARIANT_STRENGTHS = {
    "add_single_pedestrian": 0.54,
    "add_two_pedestrians": 0.58,
    "add_small_group": 0.60,
    "add_occluded_pedestrian": 0.58,
    "add_distant_pedestrian": 0.52,
    "add_near_pedestrian": 0.62,
}

VARIANT_GUIDANCE_SCALES = {
    "add_single_pedestrian": 7.6,
    "add_two_pedestrians": 8.0,
    "add_small_group": 8.2,
    "add_occluded_pedestrian": 8.0,
    "add_distant_pedestrian": 7.4,
    "add_near_pedestrian": 8.4,
}

VARIANT_NUM_INFERENCE_STEPS = {
    "add_single_pedestrian": 36,
    "add_two_pedestrians": 40,
    "add_small_group": 42,
    "add_occluded_pedestrian": 40,
    "add_distant_pedestrian": 34,
    "add_near_pedestrian": 42,
}
```

### 2.4 Kết quả V1

**Ưu điểm:**
- Đơn giản, dễ triển khai
- SD3.5 có toàn quyền sửa ảnh nên pedestrian sinh ra tự nhiên
- Không cần mask/composite phức tạp

**Nhược điểm nghiêm trọng:**
- **Phá background toàn cục**: strength 0.54-0.62 đủ mạnh để thay đổi road, building, cars, camera viewpoint
- Không kiểm soát được vùng nào bị thay đổi
- Ảnh augmented có thể khác ảnh gốc ở mọi pixel
- Khó đánh giá "có thêm người" hay "đổi cả scene"

**Ví dụ lỗi:**
```
Original: road A, building B, car C
Augmented: road D, building E, car F + pedestrian
```
→ Không còn là "thêm người", mà là "vẽ lại ảnh"

### 2.5 Nguyên nhân gốc rễ

SD3.5 img2img hoạt động bằng cách:
1. Encode ảnh gốc thành latent
2. Thêm noise theo strength
3. Denoise toàn bộ latent theo prompt
4. Decode thành ảnh mới

Vì denoise toàn bộ latent, model có quyền thay đổi mọi pixel. Prompt "preserve background" là yếu đuối so với quá trình diffusion tự nhiên.

---

## 3: PHIÊN BẢN V2 - PATCH BLEND

### 3.1 Động lực chuyển đổi

Sau khi V1 phá background quá nhiều, team quyết định: **"Không cho SD3.5 thấy toàn ảnh nữa, chỉ cho nó thấy patch nhỏ"**.

### 3.2 Kiến trúc V2

```
Original Image (448x448)
→ Smart Placement chọn insert_bbox
→ Expand thành patch có context
→ SD3.5 img2img trên patch (strength cao)
→ Chỉ paste vùng insert_bbox nhỏ về ảnh gốc
→ Feather blend ở viền paste
→ Save final image
```

**Core pipeline:**
```python
# Chỉ paste insert_bbox, không paste toàn patch
ix1, iy1, ix2, iy2 = insert_bbox
px1, py1, _, _ = patch_bbox
rel_insert_bbox = (ix1 - px1, iy1 - py1, ix2 - px1, iy2 - py1)
generated_insert = aug_patch.crop(rel_insert_bbox)

result = source.copy()
result.paste(generated_insert, (ix1, iy1), feather_mask(generated_insert.size))
```

### 3.3 Smart Placement V1

```python
def find_insertion_region(...):
    # Sample 80 candidates
    # Scale người theo ground_y/perspective
    # Tránh overlap person cũ
    # Tránh sát mép ảnh
    # Chọn bbox điểm cao nhất
```

**Perspective scale:**
```python
PERSPECTIVE_SCALE_FAR = 0.28
PERSPECTIVE_SCALE_NEAR = 1.28
```

### 3.4 Parameters V2

```python
BACKGROUND_PRESERVATION_MODE = "patch_blend"
PATCH_CONTEXT_RATIO = 0.18
PATCH_MIN_SIZE = 128
PATCH_FEATHER_RADIUS = 3

VARIANT_STRENGTHS = {
    "add_single_pedestrian": 0.70,
    "add_two_pedestrians": 0.74,
    "add_small_group": 0.76,
    "add_occluded_pedestrian": 0.72,
    "add_distant_pedestrian": 0.62,
    "add_near_pedestrian": 0.76,
}
```

### 3.5 Kết quả V2

**Ưu điểm:**
- Background ngoài patch giữ nguyên 100%
- Chỉ ~18% pixel bị thay đổi (với patch 192x192 trên ảnh 448x448)
- Kiểm soát được vùng thay đổi

**Nhược điểm:**
- **Không sinh được người rõ**: img2img trên patch nền trống không có cấu trúc người, model chỉ "beautify" texture nền
- Tỉ lệ người không đúng: bbox quá to so với phối cảnh
- Vùng paste vẫn có thể làm méo road/building trong bbox
- Khó kiểm soát "người đứng đúng chỗ" vì model tự quyết trong patch

**Ví dụ lỗi:**
```
Patch input: road texture + sidewalk
Patch output: road texture đẹp hơn, KHÔNG có người
Final image: giống input, không có pedestrian mới
```

### 3.6 Cố gắng cải thiện V2

**Thêm person guide (skeleton):**
```python
DRAW_INSERTION_GUIDE = True
INSERTION_GUIDE_ALPHA = 0.34
```

Vẽ silhouette người mờ trong vùng insert trước khi đưa patch vào SD3.5, hy vọng model refine thành pedestrian.

**Kết quả:** Vẫn không hiệu quả. Model hoặc giữ skeleton như bóng mờ, hoặc xóa hẳn.

---

##  4: PHIÊN BẢN V3 - HUMAN MASK INPAINT

### 4.1 Động lực chuyển đổi

V2 fail vì img2img không có mask thật. Team quyết định chuyển sang **inpainting pipeline** để model chỉ được phép thay đổi trong mask.

### 4.2 Kiến trúc V3

```
Original Image
→ Smart Placement chọn insert_bbox
→ Tạo human-shaped mask (head + torso + legs)
→ SD3.5 InpaintPipeline với mask
→ Composite chỉ vùng mask về ảnh gốc
→ Add contact shadow
→ Save final image
```

**Core pipeline:**
```python
mask_image = human_mask_for_bbox(source.size, insert_bbox, variant)
generated = pipe(
    image=source,
    prompt=prompt,
    mask_image=mask_image,
    strength=0.42,
    guidance_scale=5.0,
).images[0]

result = source.copy()
result.paste(generated, (0, 0), mask_image)
```

### 4.3 Human Mask

```python
def human_mask_for_bbox(image_size, insert_bbox, variant):
    # Tạo mask hình người: head oval + torso rectangle + legs
    # Không dùng rectangle bbox thô
```

### 4.4 Parameters V3

```python
BACKGROUND_PRESERVATION_MODE = "human_mask_inpaint"
HUMAN_MASK_PADDING = 2
HUMAN_MASK_BLUR_RADIUS = 2

VARIANT_STRENGTHS = {
    "add_single_pedestrian": 0.42,
    "add_two_pedestrians": 0.46,
    "add_small_group": 0.48,
    "add_occluded_pedestrian": 0.44,
    "add_distant_pedestrian": 0.36,
    "add_near_pedestrian": 0.50,
}

VARIANT_GUIDANCE_SCALES = {
    "add_single_pedestrian": 5.0,
    "add_two_pedestrians": 5.4,
    "add_small_group": 5.6,
    "add_occluded_pedestrian": 5.2,
    "add_distant_pedestrian": 4.6,
    "add_near_pedestrian": 5.6,
}
```

### 4.5 Kết quả V3

**Ưu điểm:**
- Background ngoài mask giữ nguyên tuyệt đối (inpaint chỉ thay mask)
- Không còn phá road/building ngoài vùng người

**Nhược điểm nghiêm trọng:**
- **Mask quá hẹp**: human-shaped mask chỉ ~60-80% diện tích bbox, model không có đủ "canvas" để vẽ quần áo, biên người
- **Chỉ thấy bóng mờ/skeleton**: SD3.5 không thể tạo full-body pedestrian rõ ràng trong vùng hẹp
- Tỉ lệ người vẫn sai: mask không encode đúng "gần xa"
- Không phân biệt variant: "add_two_pedestrians" vẫn chỉ có 1 mask người

**Ví dụ lỗi:**
```
Mask: hình người nhỏ trong bbox lớn
Generated: bóng mờ, limb mờ, không rõ là người
Final: giống input với bóng mờ lạ
```

### 4.6 Insight quan trọng từ V3

**"Human mask không phải lúc nào cũng tốt hơn bbox mask"**

Với diffusion model, mask càng chi tiết (hình người) thì càng hạn chế model. Model cần không gian để:
- Tạo quần áo, giày dép
- Tạo biên người tự nhiên
- Tạo shadow transition
- Tự điều chỉnh scale

Mask hình người quá chặt → model bị "bóp" → output yếu.

---

##  5: PHIÊN BẢN V4 - BBOX INPAINT

### 5.1 Động lực chuyển đổi

Sau khi V3 fail vì mask quá hẹp, team quyết định: **"Bỏ human mask, dùng bbox/ellipse mask rộng hơn để model có không gian sinh người"**.

### 5.2 Kiến trúc V4

```
Original Image
→ Smart Placement V2 chọn insert_bbox
→ Tạo rounded rectangle mask (bbox + padding)
→ SD3.5 InpaintPipeline
→ Composite chỉ vùng mask về ảnh gốc
→ Add contact shadow
→ Save final image
```

**Core pipeline:**
```python
mask_image = bbox_mask_for_bbox(source.size, insert_bbox, variant)
generated = pipe(
    image=source,
    prompt=prompt,
    mask_image=mask_image,
    strength=0.78,
    guidance_scale=7.4,
).images[0]

result = source.copy()
result.paste(generated, (0, 0), mask_image)
```

### 5.3 BBox Mask

```python
def bbox_mask_for_bbox(image_size, insert_bbox, padding=8, blur=3):
    x1, y1, x2, y2 = insert_bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((x1, y1, x2, y2), radius=8, fill=255)
    return mask.filter(ImageFilter.GaussianBlur(radius=blur))
```

### 5.4 Smart Placement V2

**Tăng đa dạng vị trí:**
```python
INSERTION_CENTER_BIAS = 0.00  # Bỏ lực hút về trung tâm
PLACEMENT_SLOT_BIAS = 1.15
PLACEMENT_SLOT_XS = (0.20, 0.36, 0.52, 0.68, 0.82)
PLACEMENT_SLOT_YS = (0.68, 0.74, 0.80, 0.86, 0.91)
```

**Scale theo reference object:**
```python
USE_REFERENCE_PERSON_SCALE = True
REFERENCE_SCALE_BLEND = 0.48
CAR_HEIGHT_TO_PERSON_HEIGHT_RATIO = 0.75
PATCH_VEHICLE_CLASS_IDS = {2, 5, 7}  # car, bus, truck
```

### 5.5 Parameters V4

```python
BACKGROUND_PRESERVATION_MODE = "bbox_inpaint"
BBOX_MASK_PADDING = 8
BBOX_MASK_BLUR_RADIUS = 3

VARIANT_STRENGTHS = {
    "add_single_pedestrian": 0.78,
    "add_two_pedestrians": 0.82,
    "add_small_group": 0.84,
    "add_occluded_pedestrian": 0.80,
    "add_distant_pedestrian": 0.70,
    "add_near_pedestrian": 0.84,
}

VARIANT_GUIDANCE_SCALES = {
    "add_single_pedestrian": 7.4,
    "add_two_pedestrians": 7.8,
    "add_small_group": 8.0,
    "add_occluded_pedestrian": 7.6,
    "add_distant_pedestrian": 6.8,
    "add_near_pedestrian": 8.0,
}
```

### 5.6 Kết quả V4

**Ưu điểm:**
- Sinh được người rõ ràng hơn V3 (mask rộng hơn)
- Background ngoài mask vẫn giữ nguyên
- Đa dạng vị trí: trái/phải/giữa, xa/gần

**Nhược điểm:**
- **Tỉ lệ người vẫn sai**: gần camera quá to, xa camera quá nhỏ hoặc ngược lại
- **Màu người không đồng nhất**: người generated có màu/lighting khác với ảnh gốc
- **Outline lộ**: khi paste person vào ảnh gốc, có viền giữa người và nền
- Vẫn có thể phá background trong bbox (dù ngoài bbox giữ nguyên)

**Ví dụ lỗi:**
```
Generated person: màu ấm, ánh sáng từ trái
Original scene: màu lạnh, ánh sáng từ phải
→ Người trông như "dán" vào ảnh
```

---

##  6: PHIÊN BẢN V5 - CONTEXT PERSON COMPOSITE (HIỆN TẠI)

### 6.1 Động lực chuyển đổi

Sau khi V4 vẫn có 3 vấn đề:
1. Tỉ lệ người sai
2. Màu người không đồng nhất
3. Outline lộ

Team quyết định **tách biệt hoàn toàn** "sinh người" và "giữ background":
- SD3.5 chỉ lo sinh người đẹp trong context crop
- Ảnh gốc giữ background
- YOLO-seg tách riêng người mới
- Chỉ paste người vào ảnh gốc

### 6.2 Kiến trúc V5

```
Original Image (512x512)
→ Smart Placement V2 chọn insert_bbox
→ Expand context crop 3.4x quanh bbox
→ SD3.5 img2img trên context crop (default clean notebook)
→ YOLOv8m-seg detect/segment person trong generated crop
→ Chọn person mask gần insert_bbox nhất
→ Color-match person pixels theo scene
→ Paste chỉ person pixels vào ảnh gốc
→ Add contact shadow
→ Save final image + debug strip
```

**Core pipeline:**
```python
# 1. Context crop
crop_bbox = expand_bbox_with_context(insert_bbox, source.size, ratio=3.4)
crop = source.crop(crop_bbox)

# 2. Context generation on crop
generated_crop = pipe(
    image=crop,
    prompt=prompt,
    strength=0.72,
    guidance_scale=6.8,
).images[0]

# 3. YOLO-seg detect person
results = yolo_seg_model.predict(generated_crop, conf=0.12)
person_mask_crop = select_person_mask(results, insert_bbox_relative)

# 4. Color match
person_matched = color_match_person(
    person_pixels=generated_crop,
    person_mask=person_mask_crop,
    scene_region=source_region_around_insert
)

# 5. Composite
result = source.copy()
result.paste(person_matched, (insert_x, insert_y), person_mask_crop)
result = add_contact_shadow(result, insert_bbox, variant)
```

### 6.3 YOLOv8m-seg Integration

```python
from ultralytics import YOLO

yolo_seg_model = YOLO("yolov8m-seg.pt")

# Detect + segment person in generated crop
results = yolo_seg_model.predict(
    generated_crop,
    conf=CONTEXT_PERSON_MIN_CONFIDENCE,  # 0.12
    verbose=False,
    save=False,
)

# Select mask closest to target insert_bbox
person_mask_crop = select_generated_person_mask(
    results, insert_bbox_relative, crop.size
)
```

### 6.4 Color Matching

```python
COLOR_MATCH_PERSON_TO_SCENE = True
COLOR_MATCH_STRENGTH = 0.72

def color_match_person(person_pixels, person_mask, scene_region):
    # Lấy mean/std màu của vùng scene quanh người
    scene_mean = np.mean(scene_region, axis=(0,1))
    scene_std = np.std(scene_region, axis=(0,1))

    # Lấy mean/std màu của person generated
    person_mean = np.mean(person_pixels[person_mask > 0], axis=0)
    person_std = np.std(person_pixels[person_mask > 0], axis=0)

    # Blend person color về scene color
    matched = (person_pixels - person_mean) / (person_std + 1e-6)
    matched = matched * scene_std + scene_mean
    matched = np.clip(matched, 0, 255).astype(np.uint8)

    return matched * COLOR_MATCH_STRENGTH + person_pixels * (1 - COLOR_MATCH_STRENGTH)
```

### 6.5 Smart Placement V2 (đã cải tiến)

```python
# Variant-specific placement
def ground_y_range_for_variant(variant, height):
    if variant == "add_distant_pedestrian":
        return y_min, y_min + int(span * 0.34)  # Vùng xa/cao
    elif variant == "add_near_pedestrian":
        return y_min + int(span * 0.58), y_max   # Vùng gần/thấp
    else:
        return y_min + int(span * 0.25), y_max - int(span * 0.08)

# Scale theo reference person/car + perspective
PERSPECTIVE_SCALE_FAR = 0.34
PERSPECTIVE_SCALE_NEAR = 1.05
REFERENCE_SCALE_BLEND = 0.48
CAR_REFERENCE_SCALE_BLEND = 0.38

# Flexible scale envelope
FLEXIBLE_SCALE_INPAINT = True
SCALE_ENVELOPE_HEIGHT_MULT = 1.18
SCALE_ENVELOPE_WIDTH_MULT = 1.34
```

### 6.6 Parameters V5

```python
BACKGROUND_PRESERVATION_MODE = "context_person_composite"
CONTEXT_PERSON_GENERATION_PIPELINE = "img2img"
CONTEXT_CROP_EXPAND = 3.4
CONTEXT_CROP_MIN_SIZE = 192
CONTEXT_INPAINT_MASK_PADDING = 46
CONTEXT_GENERATION_RETRIES = 4
RESOLUTION = 512

# Current clean-notebook mask/compositing defaults
CONTEXT_PERSON_MASK_THRESHOLD = 0.40
PERSON_MASK_TRIM_FRINGE_PIXELS = 1
PERSON_MASK_DILATE_FOR_ACCESSORIES = 2
ACCESSORY_KEEP_COMPONENTS = 12
ACCESSORY_MIN_COMPONENT_AREA_RATIO = 0.002
USE_SEAMLESS_CLONE = False
EDGE_HALO_COLOR_MATCH_STRENGTH = 0.38
EDGE_HORIZON_BG_BLEND = 0.70
EDGE_BG_CONTEXT_PAD = 18
MAX_PERSON_PERSON_OVERLAP_RATIO = 0.08
SMOKE_IMAGES = 10

VARIANT_STRENGTHS = {
    "add_single_pedestrian": 0.72,
    "add_two_pedestrians": 0.74,
    "add_small_group": 0.76,
    "add_occluded_pedestrian": 0.74,
    "add_distant_pedestrian": 0.68,
    "add_near_pedestrian": 0.76,
}

VARIANT_GUIDANCE_SCALES = {
    "add_single_pedestrian": 6.8,
    "add_two_pedestrians": 6.9,
    "add_small_group": 7.0,
    "add_occluded_pedestrian": 6.8,
    "add_distant_pedestrian": 6.6,
    "add_near_pedestrian": 7.3,
}

VARIANT_NUM_INFERENCE_STEPS = {
    "add_single_pedestrian": 36,
    "add_two_pedestrians": 36,
    "add_small_group": 38,
    "add_occluded_pedestrian": 36,
    "add_distant_pedestrian": 34,
    "add_near_pedestrian": 38,
}
```

### 6.7 Reject/Retry Mechanism

```python
# Reject nếu crop thay đổi quá ít
CONTEXT_MIN_GENERATED_MASK_DIFF = 0.003

# Reject nếu YOLO mask quá nhỏ
CONTEXT_MIN_PERSON_MASK_AREA_RATIO = 0.00045

# Reject nếu final vẫn giống input
CONTEXT_MIN_FINAL_PERSON_DIFF = 0.01

# Retry tối đa 4 lần ở base config; smoke/quality/batch preset có thể override
CONTEXT_GENERATION_RETRIES = 4
```

### 6.8 Kết quả V5 (hiện tại)

**Ưu điểm:**
- **Background giữ nguyên tuyệt đối**: chỉ paste person pixels, không paste cả vùng context
- **Người sinh ra rõ ràng**: SD3.5 có context rộng để hiểu scene, nhưng output được cắt gọn
- **Color matching**: người có màu/lighting đồng bộ với scene
- **Edge matching hiện tại**: alpha paste mặc định, trim fringe 1px, inner-ring edge correction, và match màu viền bằng background pixels trong vùng local 10-20px quanh người thay vì toàn ảnh
- **Tỉ lệ cải thiện**: scale theo reference person/car + perspective clamp
- **Retry mechanism**: không lưu ảnh yếu, tự động thử lại
- **Debug strip**: 5 panel (source | mask | generated | person_mask | final) để chẩn đoán

**Vẫn đang cải thiện:**
- Tỉ lệ người vẫn cần tinh chỉnh thêm dù clean notebook đã có Depth Anything scale correction
- Outline/halo vẫn cần kiểm tra thủ công, nhưng notebook hiện ưu tiên trim generated-background fringe và tắt seamlessClone mặc định để giảm halo
- Retry rate còn cao với variant khó (occluded, group)
- Semantic placement (SegFormer) chưa chạy ổn định trên Kaggle

---

##  7: BẢNG SO SÁNH CHI TIẾT

### 7.1 So sánh kiến trúc

| Tiêu chí | V1 Full Img2Img | V2 Patch Blend | V3 Human Mask | V4 BBox Inpaint | V5 Context Composite |
|----------|-----------------|----------------|---------------|-----------------|---------------------|
| **Pipeline core** | Img2Img toàn ảnh | Img2Img patch | Inpaint mask người | Inpaint bbox | Img2Img context crop + YOLO-seg |
| **Background preservation** | Kém (toàn ảnh đổi) | Tốt (ngoài patch) | Tốt (ngoài mask) | Tốt (ngoài mask) | Xuất sắc (chỉ paste người) |
| **Pedestrian clarity** | Tốt | Kém (bóng mờ) | Kém (skeleton) | Tốt | Tốt |
| **Scale accuracy** | Trung bình | Kém | Kém | Trung bình | Khá |
| **Color consistency** | Tốt (tự nhiên) | Trung bình | Trung bình | Kém | Tốt (color-match) |
| **Position diversity** | Thấp | Thấp | Thấp | Khá | Khá |
| **Outline artifacts** | Không | Có (patch viền) | Ít | Có (mask blur) | Có (seg mask) |
| **Implementation complexity** | Đơn giản | Trung bình | Trung bình | Trung bình | Phức tạp |
| **Inference time** | Nhanh | Nhanh | Nhanh | Nhanh | Chậm hơn (YOLO-seg) |

### 7.2 So sánh parameters qua các version

| Parameter | V1 | V2 | V3 | V4 | V5 |
|-----------|-----|-----|-----|-----|-----|
| **Mode** | full_img2img | patch_blend | human_mask_inpaint | bbox_inpaint | context_person_composite |
| **Strength range** | 0.52-0.62 | 0.62-0.76 | 0.36-0.50 | 0.70-0.84 | 0.68-0.76 |
| **Guidance range** | 7.4-8.4 | 6.8-7.7 | 4.6-5.6 | 6.8-8.0 | 6.6-7.3 |
| **Steps range** | 34-42 | 32-38 | 26-32 | 32-38 | 34-38 |
| **Mask type** | None | None | Human-shaped | Rounded rectangle | BBox + YOLO-seg |
| **Composite method** | Full replace | Patch paste | Mask paste | Mask paste | Person-only alpha paste + local edge match |
| **Context awareness** | Toàn ảnh | Patch local | Toàn ảnh | Toàn ảnh | Crop 3.4x |

### 7.3 So sánh prompt evolution

| Version | Prompt đặc trưng | Độ dài | Focus chính |
|---------|------------------|--------|-------------|
| **V1** | "add one realistic pedestrian, correct scale, grounded" | Ngắn | Thêm người |
| **V2** | "local edit, preserve background, road, buildings, cars, viewpoint" | Trung bình | Giữ background |
| **V3** | "complete realistic full-body pedestrian inside mask, standing on road" | Dài | Full body + mask |
| **V4** | "place pedestrian only inside masked box, full body, feet on road" | Trung bình | Đúng vị trí |
| **V5** | "urban street photo. Add one pedestrian in empty road or sidewalk. Keep scene unchanged. full body visible, grounded feet, natural scale, matching light" | Ngắn (<77 CLIP tokens) | Thêm người, giữ scene, tránh token overflow |

### 7.4 So sánh Smart Placement

| Feature | V1 | V2 | V3/V4 | V5 |
|---------|-----|-----|-------|-----|
| **Placement algorithm** | Random + overlap check | Candidate scoring (80) | Candidate scoring (120) | Candidate scoring (180) + slots |
| **Perspective scale** | Không | Có (0.28-1.28) | Có (0.28-1.28) | Có (0.34-1.05) |
| **Reference person scale** | Không | Không | Có (blend 0.25) | Có (blend 0.48) |
| **Reference car scale** | Không | Không | Không | Có (blend 0.38) |
| **Variant-specific y-range** | Không | Không | Có (distant/near) | Có (distant/near + clamp) |
| **Position diversity** | Thấp | Thấp | Khá | Cao (5 slots x 5 y-levels) |
| **Semantic segmentation** | Không | Không | Không | Có (SegFormer, tạm tắt) |

---

##  8: LỖI ĐÃ GẶP QUA CÁC PHIÊN BẢN

### 8.1 Lỗi V1 → V2

**Lỗi: Background bị phá toàn cục**
- Nguyên nhân: img2img denoise toàn ảnh
- Cách phát hiện: SSIM thấp, LPIPS cao, ảnh augmented khác hoàn toàn ảnh gốc
- Cách fix: Chuyển sang patch-based, chỉ cho SD3.5 thấy vùng nhỏ

### 8.2 Lỗi V2 → V3

**Lỗi #1: Không sinh được người (chỉ thay texture nền)**
- Nguyên nhân: img2img trên patch nền trống không có cấu trúc người
- Cách phát hiện: Output giống input, không có pedestrian mới
- Cách fix: Thêm person guide/skeleton, sau đó chuyển sang inpainting

**Lỗi #2: Tỉ lệ người quá to/nhỏ**
- Nguyên nhân: Bbox scale không theo perspective
- Cách fix: Thêm perspective_scale_for_ground_y()

### 8.3 Lỗi V3 → V4

**Lỗi: Chỉ thấy bóng mờ/skeleton, không thành người**
- Nguyên nhân: Human mask quá hẹp, model không có đủ không gian
- Cách phát hiện: Debug strip show generated panel chỉ có bóng mờ
- Cách fix: Bỏ human mask, dùng bbox/ellipse mask rộng hơn

### 8.4 Lỗi V4 → V5

**Lỗi #1: Tỉ lệ người vẫn sai**
- Nguyên nhân: Scale envelope quá rộng, SD3.5 tự chọn size
- Cách phát hiện: Người gần camera quá to, người xa quá nhỏ
- Cách fix: Giảm PERSPECTIVE_SCALE_NEAR, tăng REFERENCE_SCALE_BLEND, thêm clamp

**Lỗi #2: Màu người không đồng nhất**
- Nguyên nhân: Person pixels từ generated crop có lighting khác ảnh gốc
- Cách phát hiện: Người trông như "dán" vào ảnh
- Cách fix: Color-match person pixels theo scene region trước khi paste

**Lỗi #3: Outline lộ**
- Nguyên nhân: YOLO-seg mask ăn cả vài pixel nền quanh người
- Cách phát hiện: Viền giữa người và nền
- Cách fix: Erode mask nhẹ, blur mask edge

**Lỗi #4: Output giống input (ảnh không đổi)**
- Nguyên nhân: SD3.5 inpaint quá "bảo thủ", giữ nguyên vùng mask
- Cách phát hiện: masked_rgb_mae < 0.012
- Cách fix: Tăng strength, thêm reject mechanism, retry với seed khác

### 8.5 Lỗi infrastructure

**Lỗi: CUDA OOM trên Kaggle T4**
- Nguyên nhân: Load 2 SD3.5 pipeline cùng lúc trên 2 GPU
- Cách fix: `enable_model_cpu_offload()`, dùng resolution 512 trong clean notebook, và dùng `low_cpu_mem_usage=True`

**Lỗi: Text encoder trên CPU**
- Nguyên nhân: `encode_prompt()` đưa tensor lên GPU nhưng text_encoder chưa `.to(device)`
- Cách fix: Move text encoders sang GPU trong training loop

**Lỗi: Token > 77 (CLIP limit)**
- Nguyên nhân: Prompt quá dài
- Cách fix: Rút gọn prompt, loại bỏ từ trùng lặp

**Lỗi: Output zip phình 4GB**
- Nguyên nhân: `shutil.make_archive` zip toàn bộ `/kaggle/working`
- Cách fix: Chỉ zip `OUTPUT_DIR`, không zip cả working directory

---

##  9: INSIGHT KỸ THUẬT QUAN TRỌNG

### 9.1 Insight #1: "Prompt không đủ mạnh để bảo toàn background"

Trong diffusion, prompt "preserve background" là yếu so với quá trình denoise. Muốn giữ background thật sự phải:
- Không cho model thấy background (crop/mask)
- Hoặc không dùng output của model cho background (composite)

### 9.2 Insight #2: "Human mask không phải lúc nào cũng tốt"

Mask càng chi tiết (hình người) thì càng hạn chế model. Với inpainting:
- Mask hình người → model bị "bóp" → output yếu
- Mask bbox rộng → model có không gian → output tốt hơn
- Quan trọng là sau đó chỉ lấy person pixels, không lấy cả bbox

### 9.3 Insight #3: "Tách biệt sinh và composite là hướng đúng"

Cố gắng bắt SD3.5 vừa sinh người đẹp vừa giữ background trong cùng một bước là sai lầm.

Hướng đúng:
```
SD3.5: sinh người (không cần giữ background)
Original: giữ background (không cần sinh người)
Segmentation: tách người từ generated
Composite: ghép người vào background
```

### 9.4 Insight #4: "Scale không thể chỉ dựa vào prompt"

Prompt "correct scale" là mơ hồ với diffusion. Cần:
- Geometry prior (perspective theo ground_y)
- Reference from existing objects (person/car bbox trong ảnh)
- Clamp/envelope để model không phóng đại quá mức

### 9.5 Insight #5: "Color matching cần làm post-process"

Không thể bắt SD3.5 tự match màu với ảnh gốc. Cần:
- Sinh người tự nhiên trong context
- Sau đó adjust mean/std của person pixels
- Blend với strength vừa phải (0.6-0.8) để không mất detail

### 9.6 Insight #6: "Reject mechanism quan trọng hơn perfect parameters"

Thay vì cố tìm parameters "sinh 100% đẹp", nên:
- Sinh nhiều candidates
- Reject ảnh yếu (no person, wrong scale, too similar)
- Retry với seed/config khác
- Chỉ giữ ảnh pass tất cả ngưỡng

---

##  10: KẾ HOẠCH TIẾP THEO

### 10.1 Cải thiện V5 hiện tại

**Ưu tiên 1: Tỉ lệ người chính xác hơn**
- Tinh chỉnh Depth Anything scale correction đã có trong clean notebook
- Dùng reject histogram và quality score để chỉnh soft/hard scale thresholds
- Hoặc dùng camera calibration từ CityPersons nếu có

**Ưu tiên 2: Giảm outline artifacts**
- Refine YOLO-seg mask bằng SAM (Segment Anything Model)
- Dùng matting (background removal) thay vì segmentation thô
- Feather edge mạnh hơn ở vùng paste

**Ưu tiên 3: Semantic placement hoạt động**
- Fix SegFormer load trên Kaggle (meta tensor issue)
- Hoặc dùng YOLO-seg để detect road/sidewalk
- Chỉ đặt người trên vùng road/sidewalk thật sự

**Ưu tiên 4: Variant enforcement**
- "add_two_pedestrians" cần 2 person masks riêng biệt
- "add_occluded_pedestrian" cần logic occlusion (đặt gần foreground object)
- "add_small_group" cần 3 person masks với spacing hợp lý

### 10.2 Evaluator Integration

Sau khi V5 ổn định, cần thêm evaluator layer:

```
Generated Images
→ YOLO person detection (count, confidence, bbox)
→ SSIM/LPIPS (scene preservation)
→ Histogram distance (color consistency)
→ Quality score aggregation
→ Accept/Reject
→ Manifest + Statistics
```

### 10.3 LoRA Decision Gate

```
Nếu V5 + evaluator cho kết quả:
- Background Preservation > 0.90
- Pedestrian Detection Rate > 85%
- Artifact Rate < 10%

→ KHÔNG cần LoRA. Báo cáo: "Parameter optimization sufficient."

Nếu:
- Pedestrian Detection Rate < 60%
- Background Rewrite Rate > 25%

→ Bắt đầu train LoRA trên accepted samples.
```

### 10.4 Xuất bản/Deliverable

- Notebook Kaggle hoàn chỉnh (V5)
- Script CLI độc lập (port từ notebook)
- Dataset augmentation với manifest đầy đủ
- Báo cáo so sánh các phiên bản (file này)
- Evaluation metrics trên downstream detector (mAP)

---

##  11: PHỤ LỤC - CODE EVOLUTION

### 11.1 V1 → V2: Từ full image sang patch

```python
# V1
image = pipe(image=source, strength=0.54).images[0]
image.save(output_path)

# V2
patch = source.crop(patch_bbox)
patch = draw_person_guide_on_patch(patch, ...)
aug_patch = pipe(image=patch, strength=0.70).images[0]
result = source.copy()
result.paste(aug_patch, patch_bbox[:2], feather_mask(...))
result.save(output_path)
```

### 11.2 V2 → V3: Từ patch sang inpaint

```python
# V2 (img2img patch)
aug_patch = pipe(image=patch, strength=0.70).images[0]

# V3 (inpaint with human mask)
mask_image = human_mask_for_bbox(source.size, insert_bbox, variant)
generated = pipe(image=source, mask_image=mask_image, strength=0.42).images[0]
result = source.copy()
result.paste(generated, (0, 0), mask_image)
```

### 11.3 V3 → V4: Từ human mask sang bbox mask

```python
# V3
mask_image = human_mask_for_bbox(source.size, insert_bbox, variant)

# V4
mask_image = bbox_mask_for_bbox(source.size, insert_bbox, variant)
```

### 11.4 V4 → V5: Từ mask paste sang person-only composite

```python
# V4
result = source.copy()
result.paste(generated, (0, 0), mask_image)

# V5
crop = source.crop(crop_bbox)
generated_crop = pipe(image=crop, mask_image=mask_crop, strength=0.88).images[0]
person_mask_crop = yolo_seg_select_person(generated_crop, insert_bbox_rel)
person_matched = color_match_person(generated_crop, person_mask_crop, scene_region)
result = source.copy()
result.paste(person_matched, (insert_x, insert_y), person_mask_crop)
```

---

## KẾT LUẬN

Dự án SD3.5 CityPersons Augmentation đã trải qua 5 phiên bản pipeline, mỗi phiên bản giải quyết một vấn đề cụ thể:

- **V1** chứng minh SD3.5 có thể sinh pedestrian, nhưng phá background
- **V2** chứng minh patch-based có thể giữ background, nhưng khó sinh người
- **V3** chứng minh inpainting đúng hướng, nhưng human mask quá hẹp
- **V4** chứng minh bbox mask rộng hơn sinh người tốt hơn, nhưng scale/color còn lỗi
- **V5** (hiện tại) tách biệt "sinh" và "composite", dùng YOLO-seg để chỉ paste person pixels

**Hướng đúng đã được xác định:** Tách biệt nhiệm vụ, không bắt một model làm tất cả.

**Việc cần làm tiếp:**
1. Cải thiện tỉ lệ người bằng cách tinh chỉnh depth-based scale correction
2. Giảm outline artifacts (SAM matting)
3. Hoàn thiện evaluator integration
4. Chạy batch lớn và đo downstream mAP

---

*Report được tổng hợp từ lịch sử chat session*
*Ngày tạo: 2026-06-09*
*Phiên bản notebook hiện tại: V5 - context_person_composite*
