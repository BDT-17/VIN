# CityPersons Pedestrian Augmentation With SD3.5

Notebook nghiên cứu để tạo ảnh augmentation cho pedestrian detection bằng **Stable Diffusion 3.5 Medium** và **YOLOv8m-seg**.

Pipeline hiện tại là **V5 context-person-composite bằng img2img**:

```text
Dataset scanner
-> Context crop img2img generation
-> YOLO person segmentation
-> Scale correction
-> Edge / color / brightness / shadow blending
-> Person-only alpha composite
-> YOLO validation + retry
-> Image outputs + manifest
```

## File Chính

- `sd35_run.ipynb`: runner notebook gọn để chạy project khi các module `.py` đã có sẵn.
- `sd3.5-agumentation-scale-correction-clean.ipynb`: notebook self-contained cho Kaggle, có các cell `%%writefile` để tự ghi module vào `/kaggle/working`.
- `sd35_config.py`: cấu hình.
- `sd35_data.py`: scan dataset và preview.
- `sd35_utils.py`: preprocessing, placement, mask, scale/depth helpers.
- `sd35_model.py`: load SD3.5 pipelines.
- `sd35_evaluation.py`: YOLO-seg, validation, retry policy.
- `sd35_pipeline.py`: generation, paste, edge correction, compositing.
- `sd35_runner.py`: build jobs, chạy augmentation, manifest, autotune, export.

## Cách Chạy Trên Kaggle

### Cách 1: Runner Gọn

Dùng `sd35_run.ipynb` nếu bạn upload/copy cả các file `sd35_*.py` lên cùng working directory.

Chạy lần lượt các cell:

```text
1. Install Dependencies
2. Autoreload And Imports
3. Runtime Check
4. Hugging Face Login
5. Dataset Scan
6. Smoke Run
7. Export Outputs nếu cần
```

Runner sẽ import module từ `Path.cwd()` hoặc `/kaggle/working`.

### Cách 2: Notebook Self-Contained

Dùng `sd3.5-agumentation-scale-correction-clean.ipynb` nếu bạn muốn upload một notebook duy nhất. Notebook này sẽ ghi các module Python vào `/kaggle/working` bằng `%%writefile`, sau đó import và chạy.

## Cấu Hình Quan Trọng

Chỉnh trong `sd35_config.py` hoặc trong cell `%%writefile /kaggle/working/sd35_config.py` của notebook self-contained.

```python
RUN_PRESET = "batch"  # smoke | quality | batch
PARAMETER_OVERRIDES = {}
USE_ALL_GPUS_FOR_AUGMENTATION = True
CONTEXT_PERSON_GENERATION_PIPELINE = "img2img"
SMOKE_IMAGES = 10
```

Các default chính:

- `BACKGROUND_PRESERVATION_MODE = "context_person_composite"`
- `CONTEXT_PERSON_GENERATION_PIPELINE = "img2img"`
- `RESOLUTION = 512`
- `USE_T5 = False`
- `USE_SEAMLESS_CLONE = False`
- `CONTEXT_PERSON_MASK_THRESHOLD = 0.40`
- `PERSON_MASK_TRIM_FRINGE_PIXELS = 1`

## Edge Handling

Sau khi paste người vào ảnh, pipeline mới lấy crop kết quả rồi chỉnh viền:

```text
paste person
-> crop pasted result
-> horizontal-row background mean around person
-> blur/mean edge correction
-> paste corrected crop back
```

Hiện edge correction chỉ dùng **horizontal mean**, không trộn local mean.

## GPU Memory

Runner xử lý từng device tuần tự để giữ VRAM ổn định trên Kaggle T4:

```text
load pipeline -> run shard -> del pipe -> clear_cuda() -> next device
```

## Outputs

Notebook tạo:

- augmented images;
- comparison pairs;
- optional debug images;
- manifest rows;
- rejection histogram;
- quality metrics;
- autotune snapshots;
- optional zip artifact từ `export_outputs()`.

## Trạng Thái

Đây là research baseline, tập trung vào scale correction, grounding, occlusion ordering, blending, validation và autotune. Chưa phải production augmentation pipeline.
