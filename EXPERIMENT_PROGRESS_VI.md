# Tiến độ thí nghiệm HRM-PET + CFS + Semantic

## 1. Mục tiêu

Mục tiêu hiện tại là cải tiến HRM-PET bằng hai hướng lấy ý tưởng từ paper PMI-CFS:

- CFS, viết tắt của Contrastive Feature Selection.
- Semantic-aware feature projection / semantic-aware relation distillation.

Trong HRM-PET, phần phù hợp nhất để áp dụng CFS là CRCT feature replay. Phần phù hợp nhất để áp dụng semantic là CTIRD, vì CTIRD đang distill quan hệ giữa các sample/class qua feature similarity.

## 2. Những gì đã làm

### 2.1. Thêm CFS vào CRCT

Đã thêm CFS vào bước sinh feature giả cho CRCT ở cả hai engine:

- TII / HidePrompt engine.
- LoRA HRM engine.

Logic mới:

```text
feature thật theo class
-> tính mean/covariance
-> train MLP contrastive nhỏ cho từng class
-> sample nhiều Gaussian candidates
-> đưa candidates qua MLP CFS
-> chọn subset đa dạng hơn
-> dùng subset đó cho CRCT
```

CFS có thể bật/tắt bằng:

```bash
--cfs_sampling
```

### 2.2. Thêm semantic-aware CTIRD bản đầu

Bản đầu dùng tên class để tạo semantic embedding bằng hashing, sau đó nhân semantic similarity vào target relation của CTIRD.

Cách này chạy được, nhưng đưa semantic vào khá mạnh, nhất là khi `semantic_alpha=1.0`.

### 2.3. Thêm semantic top-k theo hướng gần paper hơn

Sau khi đọc paper, semantic được sửa theo hướng:

- chỉ dùng top-5 class gần nghĩa nhất,
- giảm `semantic_alpha` xuống `0.1`, giống paper,
- thêm superclass CIFAR100 để similarity có ý nghĩa hơn.

Mode này là:

```bash
--semantic_mode topk_mix
--semantic_alpha 0.1
--semantic_top_k 5
```

### 2.4. Cải tiến tiếp: semantic adaptive gate

Kết quả top-k vẫn chưa tốt hơn CFS-only. Vì vậy code tiếp tục được sửa theo hướng an toàn hơn: `adaptive_gate`.

Ý tưởng:

- Không tạo semantic target mới thay thế CTIRD.
- Không ép model học theo semantic prior độc lập.
- Chỉ tăng nhẹ trọng số của những quan hệ mà CTIRD cũ đã có, đồng thời hai class cũng gần nghĩa.

Công thức ý tưởng:

```text
gated_target = normalize(old_relation * (1 + alpha * semantic_weight_topk))
```

Mode mới:

```bash
--semantic_mode adaptive_gate
--semantic_alpha 0.05
--semantic_top_k 5
```

Đây là hướng ít phá kết quả CFS-only hơn, vì semantic chỉ đóng vai trò điều chỉnh phụ.

## 3. Kết quả đã chạy

### 3.1. Baseline không CFS

```text
Acc@task: 87.0500
Acc@1:    86.9000
Acc@5:    97.5900
Loss:     0.5860
```

### 3.2. CFS-only

```text
Acc@task: 88.0600
Acc@1:    88.0000
Acc@5:    97.9400
Loss:     0.5232
Forgetting: 3.8889
Backward: -3.7000
```

Nhận xét: CFS-only là bản tốt nhất hiện tại.

### 3.3. CFS + semantic bản đầu alpha 1.0

```text
Acc@task: 87.3000
Acc@1:    87.2100
Acc@5:    97.6800
Loss:     0.5519
Forgetting: 3.4222
Backward: -3.0222
```

Nhận xét: tốt hơn baseline một chút, nhưng thấp hơn CFS-only.

### 3.4. CFS + semantic top-k alpha 0.1

```text
Acc@task: 87.1700
Acc@1:    86.8900
Acc@5:    97.5700
Loss:     0.5559
Forgetting: 3.8556
Backward: -3.4000
Total training time: 0:38:34
```

Nhận xét: không cải thiện, gần baseline và thấp hơn CFS-only.

## 4. Kết luận tạm thời

CFS đang là phần cải thiện chính. Việc chọn feature replay tốt hơn cho CRCT giúp classifier ổn định hơn và giảm loss rõ ràng.

Semantic-aware CTIRD đã chạy được nhưng chưa cải thiện trên CIFAR100. Khả năng cao là vì semantic từ class name/superclass còn yếu so với CLIP text feature trong paper, và nếu đưa semantic vào quá mạnh thì nó làm lệch target relation của CTIRD.

Vì vậy hướng cải tiến tiếp theo là dùng semantic rất nhẹ, như một gate hỗ trợ CTIRD thay vì tạo target mới. Đây là lý do thêm mode `adaptive_gate`.

## 5. Lệnh chạy tiếp nên thử

Chạy lại LoRA với CFS + semantic adaptive gate:

```bash
cd ~/Documents/truongnguyen/Hybrid_ReMatching

python -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port=29514 \
  main.py \
  cifar100_lora \
  --model vit_base_patch16_224 \
  --original_model vit_base_patch16_224 \
  --batch-size 24 \
  --epochs 5 \
  --data-path ~/datasets \
  --ca_lr 0.005 \
  --crct_epochs 10 \
  --seed 42 \
  --lr 0.03 \
  --con 0.2 \
  --lora_rank 5 \
  --En gen \
  --tau -10 \
  --K 5 \
  --sched cosine \
  --dataset Split-CIFAR100 \
  --lora_momentum 0.4 \
  --lora_type hide \
  --trained_original_model ~/Documents/truongnguyen/hrm-pet-output/cifar100_tii_10tasks_seed42 \
  --num_tasks 10 \
  --cfs_sampling \
  --cfs_epochs 50 \
  --cfs_train_max_samples 1024 \
  --cfs_candidate_multiplier 3 \
  --semantic_distill \
  --semantic_alpha 0.05 \
  --semantic_top_k 5 \
  --semantic_mode adaptive_gate \
  --output_dir ~/Documents/truongnguyen/hrm-pet-output/cifar100_lora_cfs_semantic_gate_a005_seed42 \
  2>&1 | tee ~/Documents/truongnguyen/hrm-pet-output/cifar100_lora_cfs_semantic_gate_a005_seed42.log
```

Kỳ vọng: nếu semantic có ích, mode này nên giữ được mức gần CFS-only và có thể cải thiện nhẹ loss/forgetting. Nếu vẫn thấp hơn CFS-only, nên báo cáo rằng semantic class-name chưa đủ mạnh và cần CLIP text embedding thật để bám sát paper hơn.
## 6. Semantic projection sát ý tưởng paper hơn

Sau khi thử `semantic_mode=topk_mix` và thấy kết quả không vượt CFS-only, code được cải tiến thêm một hướng sát ý tưởng gốc của paper hơn: `--semantic_projection`.

Ý tưởng gốc trong paper:

```text
text feature class c -> text feature class d
-> tạo projection/rotation c sang d
-> apply projection đó lên image feature của class c
-> tạo pseudo feature cho class d
```

Bản áp dụng vào HRM-PET hiện tại:

```text
semantic embedding class nguồn c
semantic embedding class đích d
-> tạo phép xoay/reflection trong feature space
-> sample feature từ Gaussian/CFS của class c
-> xoay phần residual của feature c sang hướng semantic của class d
-> đặt quanh mean của class d
-> dùng pseudo feature này trong CRCT của class d
```

Công thức ý tưởng:

```text
x_c = sample_from_class_c
r_c = x_c - mean_c
R_cd = semantic_rotation(text_c -> text_d)
x_d_pseudo = mean_d + R_cd(r_c)
```

Trong code, phép `R_cd` được triển khai bằng một phép biến đổi trực giao kiểu Householder để không cần tạo ma trận 768 x 768 đầy đủ. Đây là cách nhẹ hơn nhưng vẫn giữ tinh thần: dùng quan hệ semantic giữa class để project feature từ class nguồn sang class đích.

Tham số mới:

```bash
--semantic_projection
--semantic_projection_ratio 0.25
--semantic_projection_top_k 5
--semantic_projection_strength 1.0
```

Ý nghĩa:

- `--semantic_projection`: bật semantic feature projection trong CRCT.
- `--semantic_projection_ratio`: tỷ lệ feature CRCT của mỗi class được lấy từ semantic projection. Ví dụ `0.25` nghĩa là 25% projected feature, 75% Gaussian/CFS feature gốc.
- `--semantic_projection_top_k`: số class nguồn gần nghĩa nhất dùng để project sang class đích.
- `--semantic_projection_strength`: mức áp dụng phép projection. `1.0` là dùng projection đầy đủ, thấp hơn thì trộn nhẹ hơn.

Lưu ý quan trọng:

- Bản này sát paper hơn các bản semantic CTIRD trước vì nó thật sự có bước project feature từ class này sang class khác.
- Tuy nhiên vẫn chưa phải bản paper gốc tuyệt đối, vì paper dùng CLIP text feature thật và model inversion sinh ảnh. Bản hiện tại dùng semantic embedding nhẹ từ class name/superclass và áp dụng trực tiếp trong HRM feature space.
- Đây là ablation đáng thử tiếp theo. Nếu cải thiện, có thể trình bày là semantic-aware feature projection đã được chuyển hóa thành semantic-projected CRCT replay trong HRM-PET.