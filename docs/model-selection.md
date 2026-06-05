# Phân tích lựa chọn model

## Model được lựa chọn

**Stable Diffusion 3.5 Medium (BF16/FP16)** được chọn làm model chính cho pipeline **CityPersons Pedestrian Augmentation**.

Lưu ý quan trọng: hướng triển khai hiện tại **không còn LoRA/fine-tuning**. Model được dùng cho **inference/img2img augmentation**, kết hợp với YOLOv8m-seg để segment người mới và composite lại vào ảnh gốc. Vì vậy tiêu chí lựa chọn model tập trung vào:

- chất lượng generation;
- khả năng bám prompt;
- khả năng giữ bố cục gốc khi img2img;
- khả năng chạy ổn trên Kaggle T4 16GB;
- độ phù hợp với bài toán chèn pedestrian vào scene có sẵn;
- tính thực dụng trong notebook environment.

## 1. Bối cảnh bài toán

Bài toán không phải là sinh ảnh tự do từ prompt, mà là **augmentation dataset cho pedestrian detection**. Model cần tạo ra pedestrian mới trong ảnh CityPersons nhưng không được phá hỏng background gốc.

Pipeline hiện tại giải quyết bài toán theo hướng **V3 Context-Person Composite**:

1. SD3.5 Medium sinh candidate pedestrian bằng img2img.
2. YOLOv8m-seg segment người mới.
3. Pipeline chỉ paste pixel người đã segment trở lại ảnh gốc.
4. Background gốc luôn được giữ nguyên.
5. Các bước validation loại bỏ ghost person, scale sai, cropped body, thiếu số lượng người, hoặc foot-ground contact kém.

Vì vậy model được chọn phải hỗ trợ tốt generation trong bối cảnh có ảnh đầu vào, bám prompt đủ tốt, và không quá nặng để chạy nhiều lần với retry trên Kaggle.

## 2. Lý do lựa chọn SD3.5 Medium BF16/FP16

### 2.1. Phiên bản mới hơn và hoàn thiện hơn SD3 Medium

SD3.5 Medium là phiên bản cải tiến so với SD3 Medium, thuộc nhóm diffusion model dựa trên transformer hiện đại. So với SD3 Medium, SD3.5 Medium có các điểm mạnh hơn:

- prompt adherence ổn định hơn;
- text-image alignment tốt hơn;
- chất lượng generation cao hơn trong cùng mức tài nguyên;
- kiến trúc mới hơn, phù hợp hơn với hướng nghiên cứu diffusion transformer;
- ít lý do để quay lại SD3 Medium khi SD3.5 Medium đã khả dụng.

Với bài toán augmentation, prompt không chỉ mô tả "người đi bộ", mà còn cần ràng buộc: full-body, đúng scale, feet grounded, realistic street perspective, lighting phù hợp. Khả năng hiểu prompt tốt hơn giúp SD3.5 Medium có lợi thế.

### 2.2. Phù hợp với giới hạn phần cứng Kaggle T4 16GB

SD3.5 Medium có quy mô khoảng 2.5B parameters, nhẹ hơn đáng kể so với các model large-scale như FLUX.1-dev hoặc SD3.5 Large Turbo.

Trong notebook hiện tại, model chạy được trên Kaggle T4 bằng các kỹ thuật:

- BF16/FP16 inference;
- T5 disabled để giảm VRAM;
- model CPU offload;
- VAE slicing/tiling;
- attention slicing;
- resolution 448x448;
- batch size nhỏ theo từng sample;
- retry có kiểm soát.

Điểm quan trọng là model không chỉ cần load được một lần, mà phải chạy được trong pipeline có:

- SD3.5 img2img;
- YOLOv8m segmentation;
- semantic placement;
- quality validation;
- retry nhiều lần;
- debug/metrics.

SD3.5 Medium tạo được cân bằng tốt giữa chất lượng và khả năng chạy thực tế.

### 2.3. BF16/FP16 phù hợp hơn NF4 cho pipeline hiện tại

NF4/4-bit quantization hữu ích khi cần inference trong môi trường cực thấp VRAM, nhưng trong pipeline này BF16/FP16 phù hợp hơn vì:

- chất lượng output ổn định hơn;
- ít artifact hơn so với quantized inference;
- ít rủi ro sai khác do quantization;
- phù hợp hơn với img2img cần giữ texture và bố cục;
- thuận lợi hơn khi đánh giá công bằng giữa các biến thể prompt/strength/guidance.

Vì hiện tại không LoRA/fine-tuning, vấn đề không phải gradient stability nữa. Tuy nhiên, BF16/FP16 vẫn là lựa chọn tốt hơn NF4 vì mục tiêu chính là **generation quality và consistency**.

### 2.4. Phù hợp với bài toán img2img object insertion

Task augmentation này không cần model mạnh nhất cho text-to-image tự do; nó cần model ổn trong img2img, biết thêm object vào scene có sẵn mà không phá bố cục quá nhiều.

SD3.5 Medium phù hợp vì:

- img2img ổn định hơn dưới strength thấp/trung bình;
- giữ được layout đầu vào tốt hơn FLUX trong constraint notebook hiện tại;
- đủ mạnh để sinh pedestrian realistic;
- chạy được nhiều attempts trên T4;
- dễ kết hợp với segmentation-composite architecture.

Trong pipeline V3, background preservation không giao hoàn toàn cho model. Thay vào đó, background được bảo toàn bằng kiến trúc composite. Model chỉ cần sinh candidate pedestrian đủ tốt để segment và paste.

### 2.5. Practical hơn FLUX trong môi trường Kaggle

FLUX.1-dev có chất lượng generation rất mạnh, đặc biệt ở text-to-image, nhưng không phải lựa chọn thực dụng nhất cho pipeline này:

- VRAM requirement cao hơn;
- inference nặng hơn;
- khó chạy ổn định với retry nhiều lần trên Kaggle T4;
- control img2img/layout trong task chèn object không thuận lợi bằng SD3.5 Medium;
- workflow notebook phức tạp hơn.

Với dataset augmentation, model tốt nhất không nhất thiết là model sinh ảnh đẹp nhất trong benchmark tổng quát. Model tốt nhất là model tạo được output **đủ realistic, đúng bố cục, dễ kiểm soát, chạy được nhiều mẫu, và tương thích với validation pipeline**.

## 3. Lý do không lựa chọn các model còn lại

| Model | Lý do không ưu tiên |
|---|---|
| SD3 Medium BF16 | Kiến trúc cũ hơn, chất lượng và prompt adherence kém hơn SD3.5 Medium. Khi SD3.5 Medium đã chạy được trên T4, không có nhiều lý do để quay lại SD3 Medium. |
| SD3 Medium NF4 | NF4 phù hợp khi cần inference rất tiết kiệm VRAM, nhưng có thể làm output kém ổn định hơn. Không phải lựa chọn tốt nhất khi pipeline cần quality và consistency. |
| SD3.5 Medium NF4 | Có thể dùng nếu VRAM quá hạn chế, nhưng BF16/FP16 cho output ổn định hơn và phù hợp hơn với img2img augmentation. |
| SD3.5 Large Turbo NF4 | Model lớn hơn, vẫn nặng với T4, đồng thời bản Turbo/few-step đã distill mạnh nên không lý tưởng cho pipeline cần kiểm soát chi tiết qua strength/guidance/retry. |
| SD3.5 Large Turbo BF16 | Quá nặng cho 16GB VRAM trong notebook pipeline có segmentation, retry và metrics. |
| FLUX.1-dev FP16 | Chất lượng generation rất mạnh nhưng VRAM/runtime cao hơn, khó practical trên Kaggle T4 cho batch augmentation. |
| FLUX FP8/NF4 | Có thể chạy inference hoặc thử nghiệm, nhưng workflow phức tạp hơn, control img2img/object insertion chưa phù hợp bằng SD3.5 Medium trong setup hiện tại. |

## 4. So sánh SD3.5 Medium và FLUX.1 Dev cho task này

| Tiêu chí | SD3.5 Medium | FLUX.1 Dev |
|---|---|---|
| Hiểu prompt | Tốt | Rất tốt |
| Sinh người | Tốt | Rất tốt |
| Giữ bố cục gốc img2img | Tốt hơn trong setup hiện tại | Kém ổn định hơn |
| Control strength thấp | Tốt | Không ổn định bằng |
| Chèn object vào scene có sẵn | Phù hợp hơn | Trung bình |
| Inpainting/img2img workflow | Tốt | Chưa practical bằng trong notebook này |
| Kaggle T4 16GB | Chạy ổn với offload/slicing | Khá nặng |
| Batch augmentation dataset | Phù hợp hơn | Ít phù hợp hơn |
| Tích hợp với segmentation-composite | Dễ triển khai | Nặng và phức tạp hơn |

## 5. Kết luận

Stable Diffusion 3.5 Medium BF16/FP16 là lựa chọn phù hợp nhất cho pipeline hiện tại vì:

- là phiên bản mới và mạnh hơn SD3 Medium;
- có kiến trúc diffusion transformer hiện đại;
- chất lượng generation và prompt adherence tốt;
- chạy thực tế được trên Kaggle T4 16GB;
- phù hợp với img2img object insertion;
- giữ bố cục tốt hơn trong constraint hiện tại;
- practical hơn FLUX cho batch augmentation;
- tương thích tốt với V3 Context-Person Composite.

Do hiện tại không còn LoRA/fine-tuning, lựa chọn model được đánh giá theo hướng **inference quality, controllability và deployment practicality**. Trong các lựa chọn đã khảo sát, SD3.5 Medium BF16/FP16 là điểm cân bằng tốt nhất giữa chất lượng, tốc độ, khả năng kiểm soát, và tính thực dụng cho CityPersons pedestrian augmentation.
