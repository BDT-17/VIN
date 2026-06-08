# CityPersons Pedestrian Augmentation With SD3.5

Notebook nghiên cứu để tạo ảnh augmentation cho pedestrian detection bằng **Stable Diffusion 3.5 Medium** và **YOLOv8m-seg**.

Notebook chính:

```text
sd3.5-agumentation-scale-correction-clean.ipynb
```

## Mục Tiêu

Pipeline chèn thêm pedestrian vào ảnh street-scene nhưng giữ background gốc. SD3.5 sinh candidate, YOLOv8m-seg tách mask người, sau đó chỉ phần pedestrian được composite lại vào ảnh gốc.

## Pipeline

```text
Dataset scanner
-> Placement selection
-> SD3.5 img2img / inpaint generation
-> YOLO person segmentation
-> Scale correction
-> Edge / color / brightness / shadow blending
-> Occlusion-aware compositing
-> YOLO validation + retry
-> Image outputs + manifest
```

## Notebook Flow

```text
1. Install
2. Runtime Check
3. HF Login
4. Configuration
5. Imports / Prompts
6. Dataset Scanner
7. Preview
8. Image Preprocessing
10. Img2Img Augmentation Pipeline
11. First Run
12. Metrics
```

## Cấu Hình Chính

Chỉnh cấu hình trong section `## 4. Configuration`.

Các biến quan trọng:

```python
RUN_PRESET = "batch"  # smoke | quality | batch
PARAMETER_OVERRIDES = {}
USE_ALL_GPUS_FOR_AUGMENTATION = True
```

Smoke test hiện chạy:

```python
SMOKE_IMAGES = 20
SMOKE_SPLITS = ["train"]
```

## Quality Score

Notebook tính quality score từ 4 nhóm metric:

```python
quality_score = (
    0.45 * person_score
    + 0.25 * scale_score
    + 0.20 * background_score
    + 0.10 * edge_score
)
```

Score này dùng để phân tích output và gợi ý autotune.

## Autotune

Autotune mặc định là dry-run, tức là chỉ in report và lưu snapshot, không tự đổi runtime config.

```python
autotune_report = autotune_from_last_run(apply=True, dry_run=True)
```

Muốn apply thật sau khi xem report:

```python
autotune_report = autotune_from_last_run(apply=True, dry_run=False)
```

Muốn khôi phục runtime config về default trong notebook:

```python
reset_runtime_config()
```

Snapshot autotune được lưu trong:

```text
autotune_snapshots/
```

## Multi-GPU

Khi `USE_ALL_GPUS_FOR_AUGMENTATION=True`, notebook tự detect các CUDA device có sẵn và chia augmentation jobs theo GPU.

Ví dụ trên máy có 2 GPU:

```python
["cuda:0", "cuda:1"]
```

Nếu chỉ có 1 GPU thì chạy như single-GPU. Nếu không có CUDA thì fallback CPU.

## Outputs

Notebook tạo các output chính:

- augmented images;
- comparison pairs;
- optional debug images;
- manifest rows;
- rejection histogram;
- quality metrics;
- autotune snapshots.

## Trạng Thái

Đây là research baseline trong notebook, tập trung vào scale correction, grounding, occlusion ordering, blending, validation và autotune. Chưa phải production augmentation pipeline.
