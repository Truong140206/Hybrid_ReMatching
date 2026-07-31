# Ghi chú: Semantic-aware CTIRD bản top-k

## 1. Vì sao cần sửa lại semantic

Bản semantic trước đó đã được gắn vào CTIRD, nhưng còn khá đơn giản: code tạo embedding từ tên class bằng hashing, sau đó dùng semantic similarity để nhân trực tiếp vào ma trận relation target của CTIRD.

Kết quả chạy CIFAR100 10 task cho thấy bản này chạy đúng, có cải thiện nhẹ so với baseline gốc, nhưng chưa tốt hơn CFS-only. Nguyên nhân hợp lý là semantic bị đưa vào quá mạnh (`semantic_alpha=1.0`) và quan hệ semantic dựa trên hashing tên class chưa đủ chuẩn, nên có thể làm nhiễu target CTIRD.

## 2. Ý tưởng từ paper

Trong paper PMI-CFS, semantic-aware feature projection dùng thông tin ngữ nghĩa giữa class, thường lấy từ text feature của CLIP. Paper dùng các class gần nghĩa nhất, cụ thể là top 5 class tương tự nhất theo cosine similarity giữa text feature, và hệ số ảnh hưởng semantic nhỏ (`alpha=0.1`).

Vì HRM-PET hiện tại không dùng CLIP trực tiếp trong pipeline train, bản sửa này áp dụng ý tưởng theo hướng nhẹ và ổn định hơn:

- Chỉ dùng top-k class gần nghĩa nhất thay vì tác động lên toàn bộ ma trận.
- Dùng `semantic_alpha=0.1` mặc định, giống paper, để semantic chỉ đóng vai trò hiệu chỉnh nhẹ.
- Với CIFAR100, thêm superclass chuẩn của CIFAR100 để similarity có ý nghĩa hơn, thay vì chỉ dựa vào tên class đơn lẻ.

## 3. Code đã thay đổi gì

### `utils.py`

Thêm bảng superclass CIFAR100 gồm 20 nhóm coarse class. Ví dụ:

- aquatic mammals
- fish
- flowers
- vehicles
- people
- reptiles
- trees

Khi tạo semantic embedding cho class CIFAR100, code kết hợp:

```text
semantic_embedding = normalize(0.45 * name_embedding + 0.89 * superclass_embedding)
```

Nhờ vậy các class cùng nhóm, ví dụ `bus`, `train`, `bicycle`, sẽ gần nhau hơn trong semantic space.

### CTIRD semantic target mới

Trước đây:

```text
semantic_relation = normalize(old_relation * semantic_weight)
```

Bản mới:

```text
semantic_prior = normalize(top_k_semantic_weight)
new_target = normalize((1 - alpha) * old_relation + alpha * semantic_prior)
```

Nói đơn giản: CTIRD vẫn giữ target chính từ model cũ, nhưng trộn thêm một lượng nhỏ quan hệ semantic giữa các class gần nghĩa.

## 4. Tham số mới

Các config đã có thêm:

```bash
--semantic_top_k 5
--semantic_mode topk_mix
```

Các tham số semantic hiện tại:

```bash
--semantic_distill
--semantic_alpha 0.1
--semantic_top_k 5
--semantic_mode topk_mix
--semantic_floor 0.2
--semantic_sharpness 1.0
```

Ý nghĩa:

- `--semantic_distill`: bật semantic-aware CTIRD.
- `--semantic_alpha`: mức trộn semantic vào CTIRD. Nên bắt đầu bằng `0.1`.
- `--semantic_top_k`: số class gần nghĩa nhất được giữ lại. Theo paper là `5`.
- `--semantic_mode topk_mix`: dùng kiểu semantic prior top-k mới.
- `--semantic_mode weight`: quay lại kiểu cũ nếu cần ablation.
- `--semantic_floor`: chỉ dùng cho mode `weight`.
- `--semantic_sharpness`: làm similarity sắc hơn nếu tăng trên `1.0`.

## 5. Kỳ vọng kết quả

Bản mới có khả năng cao ổn định hơn bản semantic trước vì semantic không còn áp quá mạnh. Tuy nhiên không thể đảm bảo chắc chắn sẽ vượt CFS-only nếu chưa chạy ablation, vì CIFAR100 có đủ dữ liệu thật nên semantic chỉ là regularization phụ.

Thứ tự thử nghiệm nên là:

1. CFS-only: kết quả hiện đang tốt nhất trong log trước.
2. CFS + semantic top-k alpha 0.1: bản mới cần chạy lại.
3. Nếu chưa tăng, thử `semantic_alpha=0.05` hoặc `semantic_top_k=3`.

## 6. Câu trình bày ngắn

Sau khi nhận thấy semantic-aware CTIRD bản đầu chưa vượt CFS-only, phương pháp được điều chỉnh gần với paper hơn bằng cách dùng top-5 class semantic gần nhất và giảm hệ số tác động semantic xuống 0.1. Với CIFAR100, semantic similarity được cải thiện bằng cách kết hợp tên class với superclass chuẩn của CIFAR100. Nhờ đó CTIRD vẫn chủ yếu học quan hệ feature từ model cũ, nhưng được regularize nhẹ bởi quan hệ ngữ nghĩa giữa các class liên quan.