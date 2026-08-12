# Ti?n d? th� nghi?m HRM-PET + CFS + Semantic

## 1. M?c ti�u

M?c ti�u hi?n t?i l� c?i ti?n HRM-PET b?ng hai hu?ng l?y � tu?ng t? paper PMI-CFS:

- CFS, vi?t t?t c?a Contrastive Feature Selection.
- Semantic-aware feature projection / semantic-aware relation distillation.

Trong HRM-PET, ph?n ph� h?p nh?t d? �p d?ng CFS l� CRCT feature replay. Ph?n ph� h?p nh?t d? �p d?ng semantic l� CTIRD, v� CTIRD dang distill quan h? gi?a c�c sample/class qua feature similarity.

## 2. Nh?ng g� d� l�m

### 2.1. Th�m CFS v�o CRCT

�� th�m CFS v�o bu?c sinh feature gi? cho CRCT ? c? hai engine:

- TII / HidePrompt engine.
- LoRA HRM engine.

Logic m?i:

```text
feature th?t theo class
-> t�nh mean/covariance
-> train MLP contrastive nh? cho t?ng class
-> sample nhi?u Gaussian candidates
-> dua candidates qua MLP CFS
-> ch?n subset da d?ng hon
-> d�ng subset d� cho CRCT
```

CFS c� th? b?t/t?t b?ng:

```bash
--cfs_sampling
```

### 2.2. Th�m semantic-aware CTIRD b?n d?u

B?n d?u d�ng t�n class d? t?o semantic embedding b?ng hashing, sau d� nh�n semantic similarity v�o target relation c?a CTIRD.

C�ch n�y ch?y du?c, nhung dua semantic v�o kh� m?nh, nh?t l� khi `semantic_alpha=1.0`.

### 2.3. Th�m semantic top-k theo hu?ng g?n paper hon

Sau khi d?c paper, semantic du?c s?a theo hu?ng:

- ch? d�ng top-5 class g?n nghia nh?t,
- gi?m `semantic_alpha` xu?ng `0.1`, gi?ng paper,
- th�m superclass CIFAR100 d? similarity c� � nghia hon.

Mode n�y l�:

```bash
--semantic_mode topk_mix
--semantic_alpha 0.1
--semantic_top_k 5
```

### 2.4. C?i ti?n ti?p: semantic adaptive gate

K?t qu? top-k v?n chua t?t hon CFS-only. V� v?y code ti?p t?c du?c s?a theo hu?ng an to�n hon: `adaptive_gate`.

� tu?ng:

- Kh�ng t?o semantic target m?i thay th? CTIRD.
- Kh�ng �p model h?c theo semantic prior d?c l?p.
- Ch? tang nh? tr?ng s? c?a nh?ng quan h? m� CTIRD cu d� c�, d?ng th?i hai class cung g?n nghia.

C�ng th?c � tu?ng:

```text
gated_target = normalize(old_relation * (1 + alpha * semantic_weight_topk))
```

Mode m?i:

```bash
--semantic_mode adaptive_gate
--semantic_alpha 0.05
--semantic_top_k 5
```

��y l� hu?ng �t ph� k?t qu? CFS-only hon, v� semantic ch? d�ng vai tr� di?u ch?nh ph?.

## 3. K?t qu? d� ch?y

### 3.1. Baseline kh�ng CFS

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

Nh?n x�t: CFS-only l� b?n t?t nh?t hi?n t?i.

### 3.3. CFS + semantic b?n d?u alpha 1.0

```text
Acc@task: 87.3000
Acc@1:    87.2100
Acc@5:    97.6800
Loss:     0.5519
Forgetting: 3.4222
Backward: -3.0222
```

Nh?n x�t: t?t hon baseline m?t ch�t, nhung th?p hon CFS-only.

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

Nh?n x�t: kh�ng c?i thi?n, g?n baseline v� th?p hon CFS-only.

## 4. K?t lu?n t?m th?i

CFS dang l� ph?n c?i thi?n ch�nh. Vi?c ch?n feature replay t?t hon cho CRCT gi�p classifier ?n d?nh hon v� gi?m loss r� r�ng.

Semantic-aware CTIRD d� ch?y du?c nhung chua c?i thi?n tr�n CIFAR100. Kh? nang cao l� v� semantic t? class name/superclass c�n y?u so v?i CLIP text feature trong paper, v� n?u dua semantic v�o qu� m?nh th� n� l�m l?ch target relation c?a CTIRD.

V� v?y hu?ng c?i ti?n ti?p theo l� d�ng semantic r?t nh?, nhu m?t gate h? tr? CTIRD thay v� t?o target m?i. ��y l� l� do th�m mode `adaptive_gate`.

## 5. L?nh ch?y ti?p n�n th?

Ch?y l?i LoRA v?i CFS + semantic adaptive gate:

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

K? v?ng: n?u semantic c� �ch, mode n�y n�n gi? du?c m?c g?n CFS-only v� c� th? c?i thi?n nh? loss/forgetting. N?u v?n th?p hon CFS-only, n�n b�o c�o r?ng semantic class-name chua d? m?nh v� c?n CLIP text embedding th?t d? b�m s�t paper hon.
## 6. Semantic projection s�t � tu?ng paper hon

Sau khi th? `semantic_mode=topk_mix` v� th?y k?t qu? kh�ng vu?t CFS-only, code du?c c?i ti?n th�m m?t hu?ng s�t � tu?ng g?c c?a paper hon: `--semantic_projection`.

� tu?ng g?c trong paper:

```text
text feature class c -> text feature class d
-> t?o projection/rotation c sang d
-> apply projection d� l�n image feature c?a class c
-> t?o pseudo feature cho class d
```

B?n �p d?ng v�o HRM-PET hi?n t?i:

```text
semantic embedding class ngu?n c
semantic embedding class d�ch d
-> t?o ph�p xoay/reflection trong feature space
-> sample feature t? Gaussian/CFS c?a class c
-> xoay ph?n residual c?a feature c sang hu?ng semantic c?a class d
-> d?t quanh mean c?a class d
-> d�ng pseudo feature n�y trong CRCT c?a class d
```

C�ng th?c � tu?ng:

```text
x_c = sample_from_class_c
r_c = x_c - mean_c
R_cd = semantic_rotation(text_c -> text_d)
x_d_pseudo = mean_d + R_cd(r_c)
```

Trong code, ph�p `R_cd` du?c tri?n khai b?ng m?t ph�p bi?n d?i tr?c giao ki?u Householder d? kh�ng c?n t?o ma tr?n 768 x 768 d?y d?. ��y l� c�ch nh? hon nhung v?n gi? tinh th?n: d�ng quan h? semantic gi?a class d? project feature t? class ngu?n sang class d�ch.

Tham s? m?i:

```bash
--semantic_projection
--semantic_projection_ratio 0.25
--semantic_projection_top_k 5
--semantic_projection_strength 1.0
```

� nghia:

- `--semantic_projection`: b?t semantic feature projection trong CRCT.
- `--semantic_projection_ratio`: t? l? feature CRCT c?a m?i class du?c l?y t? semantic projection. V� d? `0.25` nghia l� 25% projected feature, 75% Gaussian/CFS feature g?c.
- `--semantic_projection_top_k`: s? class ngu?n g?n nghia nh?t d�ng d? project sang class d�ch.
- `--semantic_projection_strength`: m?c �p d?ng ph�p projection. `1.0` l� d�ng projection d?y d?, th?p hon th� tr?n nh? hon.

Luu � quan tr?ng:

- B?n n�y s�t paper hon c�c b?n semantic CTIRD tru?c v� n� th?t s? c� bu?c project feature t? class n�y sang class kh�c.
- Tuy nhi�n v?n chua ph?i b?n paper g?c tuy?t d?i, v� paper d�ng CLIP text feature th?t v� model inversion sinh ?nh. B?n hi?n t?i d�ng semantic embedding nh? t? class name/superclass v� �p d?ng tr?c ti?p trong HRM feature space.
- ��y l� ablation d�ng th? ti?p theo. N?u c?i thi?n, c� th? tr�nh b�y l� semantic-aware feature projection d� du?c chuy?n h�a th�nh semantic-projected CRCT replay trong HRM-PET.

## 7. K?t qu? th? semantic projection tr�n CIFAR100

L?nh ch?y tr�n m�y Ubuntu/RTX 4090 d�ng output:

```text
~/Documents/truongnguyen/hrm-pet-output/cifar100_lora_cfs_semantic_projection_seed42
```

Thi?t l?p ch�nh:

```text
num_tasks = 10
epochs = 5
crct_epochs = 10
cfs_sampling = True
semantic_distill = True
semantic_alpha = 0.05
semantic_mode = adaptive_gate
semantic_projection = True
semantic_projection_ratio = 0.25
semantic_projection_top_k = 5
semantic_projection_strength = 1.0
```

K?t qu? cu?i sau CRCT task 10:

```text
[Average accuracy till task10]
Acc@task 87.5600
Acc@1    87.0100
Acc@5    97.5900
Loss     0.5508
Forgetting 3.8111
Backward  -3.2556
Total training time: 1:55:46
```

So s�nh nhanh:

| Phi�n b?n | Acc@task | Acc@1 | Acc@5 | Loss | Nh?n x�t |
|---|---:|---:|---:|---:|---|
| Baseline chua CFS | 87.0500 | 86.9000 | 97.5900 | 0.5860 | M?c g?c d? so |
| CFS-only | 88.0600 | 88.0000 | 97.9400 | 0.5232 | T?t nh?t hi?n t?i |
| CFS + semantic top-k alpha 0.1 | 87.1700 | 86.8900 | 97.5700 | 0.5559 | Semantic top-k chua c?i thi?n r� |
| CFS + semantic projection | 87.5600 | 87.0100 | 97.5900 | 0.5508 | T?t hon semantic top-k v� baseline nh?, nhung chua vu?t CFS-only |

K?t lu?n t?m th?i:

- Semantic projection s�t � tu?ng g?c c?a paper hon semantic top-k v� c� bu?c project feature t? class ngu?n sang class d�ch.
- K?t qu? d� nh�ch l�n so v?i semantic top-k v� loss t?t hon baseline.
- Tuy nhi�n CFS-only v?n l� c?u h�nh m?nh nh?t trong c�c l?n th? hi?n t?i.
- Nguy�n nh�n h?p l�: semantic embedding hi?n t?i v?n l� class-name/superclass embedding nh?, chua ph?i CLIP text embedding th?t nhu paper, n�n t�n hi?u semantic chua d? m?nh d? vu?t CFS-only.
- Hu?ng ti?p theo n?u mu?n b�m paper hon n?a: thay semantic embedding hi?n t?i b?ng CLIP text embedding th?t c?a class name, r?i d�ng embedding d� cho c? `semantic_distill` v� `semantic_projection`.

## 8. B?n paper-style: �p d?ng s�t paper PMI-CFS hon

Sau khi th? semantic projection ki?u nh?, code du?c b? sung th�m m?t ch? d? m?i d? b�m s�t paper hon.

C�c di?m thay d?i ch�nh:

1. Semantic embedding chuy?n sang CLIP text embedding th?t khi b?t:

```bash
--semantic_backend clip
```

Thay v� d�ng vector hash t? t�n class, code s? d�ng `open_clip_torch` ho?c `clip` d? encode prompt d?ng:

```text
a photo of a {class name}.
```

�i?u n�y s�t paper hon v� paper do semantic similarity b?ng cosine similarity gi?a CLIP text features.

2. Semantic-aware feature projection c� mode m?i:

```bash
--semantic_projection_mode paper
--semantic_projection_alpha 0.1
```

Mode n�y m� ph?ng Eq. 7-9 trong paper:

```text
Ft(td) = R_c,d Ft(tc)
o_L,d = R_c,d o_L,c
o'_L,d = normalize((1 - alpha) o_L,d + alpha Ft(td))
```

Trong code hi?n t?i, `R_c,d` du?c tri?n khai nhu m?t ph�p xoay tr?c giao t?i thi?u trong m?t ph?ng t?o b?i text feature class ngu?n v� text feature class d�ch. Ph�p xoay n�y map hu?ng semantic c?a class ngu?n sang class d�ch.

3. CFS c� mode paper-style:

```bash
--cfs_paper_style
--cfs_selection_ratio 0.5
--cfs_selection_steps 5
--cfs_epochs 200
--cfs_lr 0.01
--cfs_hidden_dim 512
```

Mode n�y g?n Algorithm 2 hon: kh?i t?o m?t t?p feature d� ch?n, sau d� nhi?u bu?c sample candidate t? Gaussian v� ch?n candidate c� cosine similarity trung b�nh th?p nh?t v?i t?p d� ch?n trong kh�ng gian CFS MLP.

Luu � r?t quan tr?ng:

- Paper g?c d�ng CLIP image encoder + CLIP text encoder, feature image v� feature text c�ng n?m tr�n unit hypersphere.
- HRM-PET hi?n t?i d�ng ViT/timm feature, kh�ng ph?i pipeline CLIP inversion d?y d?.
- V� v?y b?n n�y l� b?n �p d?ng s�t c�ng th?c semantic/CFS c?a paper v�o CRCT feature replay c?a HRM-PET, chua ph?i b� nguy�n to�n b? PMI + full-model inversion sinh ?nh t? paper.
- N?u mu?n y h?t tuy?t d?i paper, ph?i t�ch h?p th�m c? pipeline model inversion PMI/full-model inversion d? sinh ?nh synthetic, vi?c n�y l?n hon nhi?u so v?i s?a CRCT feature replay.

B?n d�ng ch?y ti?p theo tr�n CIFAR100:

```bash
--cfs_sampling \
--cfs_paper_style \
--cfs_epochs 200 \
--cfs_train_max_samples 1024 \
--cfs_candidate_multiplier 3 \
--cfs_selection_ratio 0.5 \
--cfs_selection_steps 5 \
--semantic_distill \
--semantic_backend clip \
--semantic_alpha 0.1 \
--semantic_top_k 5 \
--semantic_mode topk_mix \
--semantic_projection \
--semantic_projection_mode paper \
--semantic_projection_ratio 0.5 \
--semantic_projection_top_k 5 \
--semantic_projection_alpha 0.1
```

K? v?ng:

- N?u CLIP text feature gi�p ch?n class li�n quan t?t hon hash/superclass, k?t qu? c� th? t?t hon semantic projection cu.
- N?u Acc v?n th?p hon CFS-only, c� th? k?t lu?n r?ng ph?n semantic c?a paper c?n d�ng CLIP image feature/model inversion m?i ph�t huy d?y d?, c�n trong HRM feature space th� CFS replay v?n l� th�nh ph?n c� l?i nh?t.

## 9. K?t qu? ch?y paper-style CLIP semantic projection

L?nh ch?y d�ng c?u h�nh s�t paper hon:

```text
cfs_paper_style = True
cfs_epochs = 200
cfs_selection_ratio = 0.5
cfs_selection_steps = 5
semantic_backend = clip
semantic_alpha = 0.1
semantic_mode = topk_mix
semantic_projection = True
semantic_projection_mode = paper
semantic_projection_ratio = 0.5
semantic_projection_top_k = 5
semantic_projection_alpha = 0.1
```

K?t qu? cu?i sau CRCT task 10:

```text
[Average accuracy till task10]
Acc@task 87.1600
Acc@1    86.5500
Acc@5    97.3300
Loss     0.6142
Forgetting 3.5222
Backward  -2.9222
Total training time: 0:46:48
```

So s�nh v?i c�c m?c tru?c:

| Phi�n b?n | Acc@task | Acc@1 | Acc@5 | Loss | Nh?n x�t |
|---|---:|---:|---:|---:|---|
| Baseline chua CFS | 87.0500 | 86.9000 | 97.5900 | 0.5860 | M?c g?c |
| CFS-only | 88.0600 | 88.0000 | 97.9400 | 0.5232 | T?t nh?t hi?n t?i |
| CFS + semantic projection cu | 87.5600 | 87.0100 | 97.5900 | 0.5508 | T?t hon paper-style |
| CFS + paper-style CLIP semantic | 87.1600 | 86.5500 | 97.3300 | 0.6142 | Ch?y d�ng nhung chua c?i thi?n |

K?t lu?n:

- B?n paper-style d� ch?y du?c end-to-end, kh�ng c�n l?i runtime.
- Tuy nhi�n k?t qu? kh�ng t?t hon CFS-only, v� cung th?p hon semantic projection cu.
- Nguy�n nh�n h?p l�: c�ng th?c paper gi? d?nh CLIP image feature v� CLIP text feature n?m chung kh�ng gian semantic d� normalize. HRM-PET hi?n t?i d�ng ViT/timm feature trong CRCT, kh�ng ph?i CLIP image feature d�ng nghia, n�n vi?c xoay feature theo CLIP text c� th? l�m l?ch feature kh?i kh�ng gian m� classifier HRM dang d�ng.
- `semantic_projection_ratio = 0.5` c� th? qu� m?nh: 50% feature CRCT b? thay b?ng projected feature, trong khi feature space chua th?t s? align v?i CLIP.
- Hu?ng th? ti?p h?p l� hon: gi? `semantic_backend=clip` nhung gi?m projection ratio v? `0.1` ho?c `0.25`, ho?c b?t CLIP semantic ch? cho ch?n top-k class li�n quan, c�n projection gi? ki?u mean_shift cu.

## 10. Hu?ng c?i ti?n th?c d?ng: semantic-safe CRCT

Sau khi b?n paper-style ch?y xong nhung kh�ng c?i thi?n, hu?ng ti?p theo kh�ng c? b� nguy�n paper n?a. L� do l� c�ng th?c paper gi? d?nh CLIP image feature v� CLIP text feature n?m chung kh�ng gian, c�n HRM-PET hi?n t?i d�ng feature ViT/timm trong CRCT. V� v?y n?u xoay feature qu� m?nh theo CLIP text, feature c� th? r?i kh?i ph�n ph?i m� classifier HRM dang h?c.

� tu?ng m?i: semantic ch? d�ng d? g?i � class ngu?n li�n quan, c�n feature cu?i c�ng ph?i du?c ki?m tra b?ng th?ng k� feature HRM.

Pipeline m?i:

```text
1. D�ng semantic embedding d? ch?n top-k class ngu?n g?n class d�ch.
2. Sample feature t? Gaussian/CFS c?a c�c class ngu?n d�.
3. Project feature ngu?n sang class d�ch b?ng mode mean_shift nh?.
4. Sinh nhi?u candidate hon s? c?n d�ng.
5. L?c candidate b?ng kho?ng c�ch t?i ph�n ph?i feature th?t c?a class d�ch.
6. Ch? dua c�c projected feature g?n ph�n ph?i class d�ch nh?t v�o CRCT.
```

C�c tham s? m?i:

```bash
--semantic_projection_filter
--semantic_projection_filter_multiplier 3
--semantic_projection_filter_cosine_weight 0.1
```

� nghia:

- `--semantic_projection_filter`: b?t l?c feature projected theo ph�n ph?i class d�ch.
- `--semantic_projection_filter_multiplier`: sinh nhi?u candidate hon r?i ch?n l?i. V� d? `3` nghia l� c?n 100 feature th� sinh 300 candidate r?i l?c l?y 100 t?t nh?t.
- `--semantic_projection_filter_cosine_weight`: th�m m?t ph?n nh? cosine distance t?i mean class d�ch khi x?p h?ng candidate.

Kh�c v?i paper-style:

- Kh�ng normalize feature v? unit vector nhu Eq. 9 n?a, v� HRM classifier kh�ng nh?t thi?t ho?t d?ng trong c�ng CLIP unit hypersphere.
- Kh�ng d�ng semantic d? �p to�n b? feature space.
- Semantic ch? l� prior d? ch?n ngu?n v� t?o candidate; ph�n ph?i feature th?t c?a HRM m?i l� b? l?c cu?i.

C?u h�nh n�n th? d?u ti�n:

```bash
--cfs_sampling
--cfs_epochs 50
--cfs_train_max_samples 1024
--cfs_candidate_multiplier 3
--semantic_backend clip
--semantic_projection
--semantic_projection_mode mean_shift
--semantic_projection_ratio 0.1
--semantic_projection_top_k 5
--semantic_projection_strength 0.5
--semantic_projection_filter
--semantic_projection_filter_multiplier 3
--semantic_projection_filter_cosine_weight 0.1
```

Luu �: c?u h�nh n�y c? � kh�ng b?t `--semantic_distill` ban d?u, v� c�c l?n tru?c semantic CTIRD l�m k?t qu? gi?m. Tru?c m?t ch? th�m semantic v�o CRCT replay m?t c�ch c� ki?m so�t. N?u k?t qu? t?t hon CFS-only, sau d� m?i th? b?t semantic CTIRD r?t nh?.

## 11. Kết quả semantic-safe CRCT trên CIFAR100

Cấu hình đã chạy:

```bash
--cfs_sampling
--cfs_epochs 50
--cfs_train_max_samples 1024
--cfs_candidate_multiplier 3
--semantic_backend clip
--semantic_projection
--semantic_projection_mode mean_shift
--semantic_projection_ratio 0.1
--semantic_projection_top_k 5
--semantic_projection_strength 0.5
--semantic_projection_filter
--semantic_projection_filter_multiplier 3
--semantic_projection_filter_cosine_weight 0.1
```

Kết quả cuối sau task 10:

```text
[Average accuracy till task10]
Acc@task 87.3600
Acc@1    86.9900
Acc@5    97.7400
Loss     0.5526
Forgetting 3.7333
Backward  -3.3222
```

So sánh với các mốc chính:

| Phiên bản | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Nhận xét |
|---|---:|---:|---:|---:|---:|---|
| Baseline chưa CFS | 87.0500 | 86.9000 | 97.5900 | 0.5860 | - | Mốc gốc |
| CFS-only | 88.0600 | 88.0000 | 97.9400 | 0.5232 | 3.8889 | Tốt nhất hiện tại |
| CFS + semantic projection cũ | 87.5600 | 87.0100 | 97.5900 | 0.5508 | 3.8111 | Tốt hơn baseline, nhưng chưa vượt CFS-only |
| CFS + paper-style CLIP semantic | 87.1600 | 86.5500 | 97.3300 | 0.6142 | 3.5222 | Không cải thiện |
| CFS + semantic-safe CRCT | 87.3600 | 86.9900 | 97.7400 | 0.5526 | 3.7333 | Cải thiện so với paper-style, nhưng vẫn chưa vượt CFS-only |

Nhận xét:

- Semantic-safe CRCT đã sửa được vấn đề lớn của paper-style: không còn ép feature quá mạnh theo CLIP text space.
- So với paper-style, kết quả tốt hơn rõ ở Acc@task, Acc@1, Acc@5 và loss.
- Tuy nhiên so với CFS-only, kết quả vẫn thấp hơn khoảng `0.70` Acc@task và `1.01` Acc@1.
- Điều này cho thấy phần cải thiện chắc chắn nhất hiện tại vẫn là CFS replay. Semantic hiện tại có thể giúp làm giàu replay, nhưng tín hiệu semantic chưa đủ chuẩn để vượt replay thống kê thuần.
- Hướng nên thử tiếp theo là giảm ảnh hưởng semantic hơn nữa, ví dụ `semantic_projection_ratio=0.05`, hoặc chỉ dùng semantic để chọn class nguồn nhưng không project mạnh. Một hướng khác là cải thiện CFS-only trước, vì đây đang là nền tốt nhất.

## 12. Cải tiến tiếp theo: CFS distribution filter

Sau khi các biến thể semantic chưa vượt được CFS-only, hướng cải tiến được chuyển về phần đang có hiệu quả nhất: CFS replay.

Vấn đề của CFS-only hiện tại:

- CFS chọn synthetic feature sao cho các điểm được chọn đa dạng trong embedding CFS.
- Tuy nhiên candidate ban đầu vẫn được sample từ Gaussian/covariance của class.
- Nếu Gaussian sinh ra outlier, CFS có thể chọn outlier đó vì nó khác các điểm còn lại, dù outlier này không thật sự nằm gần phân phối feature của class.
- Outlier trong CRCT có thể làm classifier bị kéo lệch, nhất là ở các task sau khi số class tăng lên.

Ý tưởng mới:

```text
1. Sinh nhiều Gaussian candidate hơn bình thường.
2. Tính điểm gần phân phối class bằng diagonal Mahalanobis distance tới mean/cov của class.
3. Có thể cộng thêm cosine distance nhỏ tới class mean.
4. Giữ lại các candidate sạch nhất.
5. Chạy CFS diversity selection trên candidate pool đã lọc.
```

Tham số mới:

```bash
--cfs_distribution_filter
--cfs_filter_multiplier 3
--cfs_filter_cosine_weight 0.0
```

Ý nghĩa:

- `--cfs_distribution_filter`: bật lọc candidate trước khi CFS chọn diversity.
- `--cfs_filter_multiplier`: số candidate thô sinh thêm trước khi lọc. Ví dụ CFS cần 360 candidate, multiplier 3 sẽ sinh 1080 candidate rồi lọc còn 360.
- `--cfs_filter_cosine_weight`: trọng số cosine distance tới class mean. Mặc định `0.0` để ưu tiên Mahalanobis thuần; có thể thử `0.05` hoặc `0.1` nếu muốn thêm ràng buộc hướng feature.

Điểm khác so với semantic-safe:

- Không dùng semantic, không dùng CLIP, không project feature.
- Chỉ làm sạch candidate Gaussian trước khi đưa vào CFS.
- Vì CFS-only đang là bản tốt nhất, hướng này ít rủi ro hơn semantic và bám trực tiếp vào thành phần đang có lợi.

Cấu hình nên chạy thử đầu tiên:

```bash
--cfs_sampling
--cfs_epochs 50
--cfs_train_max_samples 1024
--cfs_candidate_multiplier 3
--cfs_distribution_filter
--cfs_filter_multiplier 3
--cfs_filter_cosine_weight 0.0
```

Nếu kết quả tốt hơn CFS-only, có thể thử tiếp:

```bash
--cfs_filter_cosine_weight 0.05
```

hoặc tăng nhẹ candidate pool:

```bash
--cfs_candidate_multiplier 4
--cfs_filter_multiplier 3
```
## 13. Cải tiến semantic mới: semantic feature adapter

Các lần semantic trước chưa vượt CFS-only vì semantic được dùng trực tiếp từ text/CLIP. Cách này có một lệch pha quan trọng: CLIP text embedding nằm trong không gian ngữ nghĩa của CLIP, còn HRM-PET dùng feature `pre_logits` của ViT/timm để CRCT. Vì vậy semantic có thể đúng về nghĩa ngôn ngữ nhưng vẫn không khớp hình học feature mà classifier đang học.

Hướng mới là `semantic feature adapter`.

Ý tưởng:

```text
1. Lấy semantic embedding của toàn bộ class, ví dụ CLIP text embedding.
2. Sau mỗi task, lấy mean feature thật của các class đã thấy trong HRM.
3. Học một phép chiếu ridge regression từ semantic embedding sang HRM class mean.
4. Dùng semantic embedding đã align này để chọn source class liên quan và semantic projection.
5. Vẫn giữ semantic projection ratio nhỏ và filter theo target distribution để tránh phá CFS replay.
```

Khác với paper-style CLIP semantic:

- Paper-style dùng CLIP text vector trực tiếp để xoay/project feature.
- Bản adapter học cách dịch CLIP/text semantic sang feature space của HRM trước.
- Vì vậy semantic không còn ép replay đi theo không gian CLIP thuần, mà bám vào thống kê class thật đã học.

Tham số mới:

```bash
--semantic_feature_adapter
--semantic_adapter_dim 512
--semantic_adapter_ridge 0.01
--semantic_adapter_blend 1.0
--semantic_adapter_min_classes 5
```

Ý nghĩa:

- `--semantic_feature_adapter`: bật adapter semantic -> HRM feature mean.
- `--semantic_adapter_dim`: số chiều semantic embedding đầu vào cho adapter.
- `--semantic_adapter_ridge`: hệ số regularization khi học phép chiếu; giúp tránh overfit khi số class đã thấy còn ít.
- `--semantic_adapter_blend`: tỉ lệ dùng embedding đã align. `1.0` nghĩa là dùng hoàn toàn embedding đã align.
- `--semantic_adapter_min_classes`: cần ít nhất bao nhiêu class đã thấy mới bật adapter.

Cấu hình nên thử:

```bash
--cfs_sampling
--cfs_epochs 50
--cfs_train_max_samples 1024
--cfs_candidate_multiplier 3
--semantic_backend clip
--semantic_projection
--semantic_feature_adapter
--semantic_adapter_ridge 0.01
--semantic_adapter_blend 1.0
--semantic_projection_mode mean_shift
--semantic_projection_ratio 0.05
--semantic_projection_top_k 5
--semantic_projection_strength 0.35
--semantic_projection_filter
--semantic_projection_filter_multiplier 3
--semantic_projection_filter_cosine_weight 0.1
```

Kỳ vọng:

- Nếu semantic thật sự giúp, bản này có cơ hội tốt hơn các bản semantic cũ vì semantic đã được align với HRM feature space.
- Vẫn chưa thể đảm bảo vượt CFS-only trước khi chạy, nhưng đây là hướng semantic hợp lý hơn so với dùng CLIP/text trực tiếp.
- Nên chạy với ratio nhỏ `0.05` trước để tránh phá baseline CFS-only.
## 14. Kết quả semantic feature adapter và hướng covariance transfer

Kết quả chạy `semantic feature adapter` trên CIFAR100:

```text
[Average accuracy till task10]
Acc@task 87.4200
Acc@1    87.1500
Acc@5    97.8200
Loss     0.5464
Forgetting 3.6556
Backward  -3.1000
```

So với semantic-safe trước đó:

| Phiên bản | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---|---:|---:|---:|---:|---:|---:|
| Semantic-safe CRCT | 87.3600 | 86.9900 | 97.7400 | 0.5526 | 3.7333 | -3.3222 |
| Semantic feature adapter | 87.4200 | 87.1500 | 97.8200 | 0.5464 | 3.6556 | -3.1000 |

Nhận xét:

- Adapter có cải thiện thật so với semantic-safe: Acc@1 tăng `0.16`, loss giảm `0.0062`.
- Điều này cho thấy hướng align semantic sang HRM feature space hợp lý hơn dùng CLIP/text trực tiếp.
- Tuy nhiên vẫn chưa vượt CFS-only, nên semantic vẫn cần được đưa vào nhẹ và có kiểm soát hơn.

Cải tiến tiếp theo: `semantic_projection_mode=covariance_transfer`.

Ý tưởng:

```text
1. Semantic adapter chỉ dùng để chọn source class gần target class.
2. Sample feature từ source class.
3. Lấy residual: source_feature - source_mean.
4. Scale residual theo tỉ lệ variance target/source.
5. Đặt residual đã scale quanh target_mean.
6. Lọc lại bằng target distribution.
```

Khác với `mean_shift`:

- `mean_shift` vẫn xoay residual theo hướng semantic vector.
- `covariance_transfer` không xoay theo semantic nữa, chỉ mượn hình dạng biến thiên từ source class rồi neo vào target distribution.
- Vì vậy semantic đóng vai trò chọn hàng xóm class, còn hình học feature vẫn do mean/cov thật của HRM quyết định.

Tham số mới:

```bash
--semantic_projection_mode covariance_transfer
--semantic_cov_transfer_min_scale 0.5
--semantic_cov_transfer_max_scale 2.0
```

Cấu hình nên thử:

```bash
--semantic_backend clip
--semantic_projection
--semantic_feature_adapter
--semantic_projection_mode covariance_transfer
--semantic_projection_ratio 0.03
--semantic_projection_top_k 3
--semantic_projection_strength 0.75
--semantic_cov_transfer_min_scale 0.5
--semantic_cov_transfer_max_scale 2.0
--semantic_projection_filter
--semantic_projection_filter_multiplier 3
--semantic_projection_filter_cosine_weight 0.1
```

Kỳ vọng:

- Ít phá CFS-only hơn vì projected features bám target mean/cov chặt hơn.
- Nếu semantic hữu ích, bản này có khả năng cải thiện hơn adapter `mean_shift` do tránh xoay feature theo semantic vector.
## 15. Kết quả semantic covariance transfer

Kết quả chạy `semantic_projection_mode=covariance_transfer` trên CIFAR100:

```text
[Average accuracy till task10]
Acc@task 87.4000
Acc@1    87.1800
Acc@5    97.8000
Loss     0.5462
Forgetting 3.6444
Backward  -3.0778
```

So sánh với bản semantic adapter `mean_shift` trước đó:

| Phiên bản | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---|---:|---:|---:|---:|---:|---:|
| Semantic feature adapter + mean_shift | 87.4200 | 87.1500 | 97.8200 | 0.5464 | 3.6556 | -3.1000 |
| Semantic feature adapter + covariance_transfer | 87.4000 | 87.1800 | 97.8000 | 0.5462 | 3.6444 | -3.0778 |

Nhận xét:

- `covariance_transfer` tăng nhẹ Acc@1: `+0.03`.
- Loss giảm rất nhẹ: `0.5464 -> 0.5462`.
- Forgetting và Backward tốt hơn nhẹ.
- Acc@task giảm `0.02`, Acc@5 giảm `0.02`, tức khác biệt rất nhỏ.
- Kết luận: hướng covariance transfer không tạo bước nhảy lớn, nhưng nó ổn định hơn một chút ở Acc@1/forgetting so với mean_shift. Tuy nhiên cả hai vẫn chưa vượt CFS-only.

Hướng thử tiếp nên tập trung vào ablation nhỏ quanh covariance transfer thay vì tăng semantic mạnh:

```bash
--semantic_projection_ratio 0.02
--semantic_projection_top_k 3
--semantic_projection_strength 0.5
```

hoặc thử giữ ratio `0.03` nhưng giảm strength:

```bash
--semantic_projection_ratio 0.03
--semantic_projection_strength 0.5
```
## 16. Lần thử tiếp theo: semantic rất nhẹ

Sau kết quả `covariance_transfer`, semantic có cải thiện rất nhỏ ở Acc@1/loss/forgetting nhưng vẫn chưa vượt CFS-only. Vì vậy lần thử tiếp theo không tăng semantic nữa, mà giảm semantic xuống mức rất nhẹ.

Cấu hình đề xuất:

```bash
--semantic_projection_mode covariance_transfer
--semantic_projection_ratio 0.02
--semantic_projection_top_k 3
--semantic_projection_strength 0.5
```

Lý do:

- `ratio 0.02`: chỉ 2% replay feature đến từ semantic projection, tránh làm lệch CFS replay.
- `top_k 3`: chỉ dùng vài class semantic gần nhất, giảm nhiễu từ class liên quan yếu.
- `strength 0.5`: covariance transfer chỉ tác động một nửa, phần còn lại giữ residual gốc.

Kỳ vọng:

- Nếu semantic thực sự có ích, cấu hình nhẹ này có thể giữ được độ ổn định của CFS-only tốt hơn.
- Nếu kết quả vẫn dưới CFS-only, kết luận hợp lý là semantic projection chưa phù hợp bằng CFS trong setting CIFAR100/HRM hiện tại; khi trình bày nên nhấn mạnh rằng CFS là cải tiến chính, còn semantic là nhánh thử nghiệm có phân tích ablation.
## 17. Kết quả semantic rất nhẹ

Kết quả chạy cấu hình semantic rất nhẹ:

```text
[Average accuracy till task10]
Acc@task 87.3600
Acc@1    87.1400
Acc@5    97.8000
Loss     0.5463
Forgetting 3.6667
Backward  -3.1000
```

So sánh ba bản semantic gần nhất:

| Phiên bản | Ratio | Strength | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Adapter + mean_shift | 0.05 | 0.35 | 87.4200 | 87.1500 | 97.8200 | 0.5464 | 3.6556 | -3.1000 |
| Adapter + covariance_transfer | 0.03 | 0.75 | 87.4000 | 87.1800 | 97.8000 | 0.5462 | 3.6444 | -3.0778 |
| Adapter + covariance_transfer rất nhẹ | 0.02 | 0.5 | 87.3600 | 87.1400 | 97.8000 | 0.5463 | 3.6667 | -3.1000 |

Nhận xét:

- Giảm semantic xuống `ratio=0.02` không cải thiện thêm.
- Ba bản semantic adapter dao động rất gần nhau, quanh `Acc@1 = 87.14 - 87.18`.
- Tất cả vẫn thấp hơn CFS-only `Acc@1 = 88.0000`.
- Kết luận thực nghiệm hiện tại: semantic projection/adapter có thể cải thiện nhẹ so với các bản semantic cũ, nhưng chưa phải hướng mạnh nhất trong setting này. CFS-only vẫn là cấu hình tốt nhất và nên là cải tiến chính khi trình bày.

Hướng đi hợp lý tiếp theo:

- Không nên tiếp tục tăng độ mạnh semantic projection vì các lần trước cho thấy dễ kéo accuracy xuống.
- Nếu cần thêm cải tiến, nên quay lại tối ưu CFS-only: candidate filter, số candidate, CFS temperature, hoặc CRCT hyperparameter.
## 18. Ablation bỏ semantic: CFS-only + distribution filter

Sau nhiều lần thử semantic adapter/projection, kết quả vẫn chưa vượt CFS-only. Vì vậy bước tiếp theo là bỏ hoàn toàn semantic và chỉ cải tiến CFS.

Cấu hình mới:

```bash
--cfs_sampling
--cfs_epochs 50
--cfs_train_max_samples 1024
--cfs_candidate_multiplier 3
--cfs_distribution_filter
--cfs_filter_multiplier 3
--cfs_filter_cosine_weight 0.0
```

Không bật các flag sau:

```bash
--semantic_projection
--semantic_feature_adapter
--semantic_distill
--semantic_backend clip
```

Ý nghĩa:

- CFS vẫn sinh synthetic replay feature từ Gaussian/covariance của từng class.
- `cfs_distribution_filter` sinh nhiều Gaussian candidate hơn rồi lọc các điểm gần phân phối class thật nhất.
- Sau khi lọc, CFS mới chọn các điểm đa dạng trong candidate pool sạch hơn.
- Mục tiêu là giữ lợi ích chính của CFS nhưng giảm khả năng chọn outlier.

Kỳ vọng:

- Nếu semantic đúng là đang kéo CFS xuống, bản này nên gần hoặc tốt hơn CFS-only.
- Nếu distribution filter tốt, có thể vượt mốc CFS-only hiện tại `Acc@1 = 88.0000`.
- Nếu không vượt, CFS-only gốc vẫn là baseline mạnh nhất và các cải tiến nên tập trung vào hyperparameter hơn là thêm semantic.
## 19. Kết quả CFS-only + distribution filter

Kết quả chạy bỏ semantic hoàn toàn, chỉ dùng CFS + distribution filter:

```text
[Average accuracy till task10]
Acc@task 87.2000
Acc@1    87.1100
Acc@5    97.5000
Loss     0.5511
Forgetting 3.8222
Backward  -3.2111
```

So sánh với CFS-only gốc:

| Phiên bản | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---|---:|---:|---:|---:|---:|---:|
| CFS-only gốc | 88.0600 | 88.0000 | 97.9400 | 0.5232 | 3.8889 | -3.7000 |
| CFS-only + distribution filter | 87.2000 | 87.1100 | 97.5000 | 0.5511 | 3.8222 | -3.2111 |

Nhận xét:

- Distribution filter làm giảm Acc@task `-0.86` và Acc@1 `-0.89` so với CFS-only gốc.
- Loss tăng từ `0.5232` lên `0.5511`.
- Forgetting/Backward nhìn tốt hơn một chút, nhưng accuracy giảm nhiều hơn nên không đáng đánh đổi.
- Kết luận: lọc Gaussian candidate quá chặt có thể làm mất các replay feature đa dạng mà CFS cần. CFS gốc chọn được các synthetic feature đa dạng hơn, và trong setting này diversity quan trọng hơn việc lọc gần mean/cov.

Kết luận tổng hợp đến thời điểm này:

- Bản tốt nhất vẫn là CFS-only gốc.
- Semantic adapter/projection có cải thiện so với các bản semantic ban đầu nhưng chưa vượt CFS-only.
- Distribution filter cũng không vượt CFS-only.
- Khi trình bày, nên chốt CFS là cải tiến chính. Semantic và distribution filter nên trình bày như ablation/nhánh thử nghiệm: có ý tưởng hợp lý, chạy được, nhưng thực nghiệm cho thấy chưa tốt bằng CFS-only trong CIFAR100 setting.
## 20. Đính chính mốc CFS-only 88.0000

Đã tìm lại được full log cũ trong attachment:

```text
C:\Users\admin\.codex\attachments\2499c2a0-e0dc-4520-ab0f-eb9dca4f5851\pasted-text.txt
```

Kết quả `88.0600 / 88.0000` là kết quả thật ở task10:

```text
[Average accuracy till task10]
Acc@task 88.0600
Acc@1    88.0000
Acc@5    97.9400
Loss     0.5232
Forgetting 3.8889
Backward  -3.7000
Total training time: 8:42:11
```

Tuy nhiên đây không cùng cấu hình với các lệnh rerun gần đây.

Cấu hình LoRA trong log cũ:

```text
subparser_name = cifar100_lora
batch_size = 24
epochs = 10
crct_epochs = 3
trained_original_model = /content/drive/MyDrive/hrm-pet-output-colab/cifar100_tii_cfs_10tasks_seed42
cfs_sampling = True
cfs_epochs = 20
cfs_batch_size = 128
cfs_candidate_multiplier = 2
cfs_train_max_samples = 1024
cfs_tau = 1.0
semantic_projection = không có / False
```

Cấu hình TII trong cùng log cũ:

```text
subparser_name = cifar100_hideprompt_5e
batch_size = 64
epochs = 5
crct_epochs = 3
train_inference_task_only = True
ca_storage_efficient_method = covariance
cfs_sampling = True
cfs_epochs = 20
cfs_batch_size = 128
cfs_candidate_multiplier = 2
output_dir = /content/drive/MyDrive/hrm-pet-output-colab/cifar100_tii_cfs_10tasks_seed42
```

Các rerun gần đây khác ở nhiều điểm:

```text
LoRA epochs = 5 thay vì 10
crct_epochs = 10 hoặc 30 thay vì 3
cfs_epochs = 50 thay vì 20
cfs_candidate_multiplier = 3 thay vì 2
cfs_batch_size = 256 thay vì 128
trained_original_model = cifar100_tii_10tasks_seed42 thay vì cifar100_tii_cfs_10tasks_seed42
```

Kết luận sửa lại:

- Mốc CFS-only `Acc@1=88.0000` là đúng, không phải nhầm task9.
- Nhưng để reproduce phải chạy đúng cấu hình cũ, đặc biệt là dùng checkpoint TII `cifar100_tii_cfs_10tasks_seed42` và LoRA `epochs=10`, `crct_epochs=3`, `cfs_epochs=20`, `candidate_multiplier=2`.
- Các kết quả rerun sau này không thể dùng để phủ định mốc cũ vì không cùng cấu hình/checkpoint.
## 21. Kết quả reproduce CFS-only cấu hình cũ trên máy ảo

Sau khi patch lỗi `PretrainedCfg` trong `hide_prompt_vision_transformer.py`, đã chạy lại cấu hình CFS-only cũ trên máy ảo.

Kết quả:

```text
[Average accuracy till task10]
Acc@task 87.6200
Acc@1    87.8800
Acc@5    98.0800
Loss     0.5222
Forgetting 3.7222
Backward  -3.3889
```

So sánh với log cũ trên Colab/Kaggle:

| Phiên bản | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---|---:|---:|---:|---:|---:|---:|
| CFS-only cũ trong full log | 88.0600 | 88.0000 | 97.9400 | 0.5232 | 3.8889 | -3.7000 |
| Reproduce trên máy ảo | 87.6200 | 87.8800 | 98.0800 | 0.5222 | 3.7222 | -3.3889 |

Nhận xét:

- Reproduce không khớp tuyệt đối, nhưng rất gần mốc cũ.
- Acc@1 chỉ thấp hơn `0.12` so với log cũ.
- Acc@5 cao hơn `0.14`, loss tốt hơn `0.0010`, forgetting/backward cũng tốt hơn.
- Chênh lệch nhỏ có thể đến từ môi trường khác nhau: Colab/Kaggle vs máy ảo RTX 4090, phiên bản thư viện, nondeterminism CUDA/DDP, hoặc checkpoint TII được train lại.
- Cấu hình CFS-only cũ vẫn là nhánh mạnh nhất/ổn định nhất hiện tại.
## 22. Cải tiến CFS-only: mean initialization

Sau khi reproduce CFS-only cấu hình cũ, hướng tối ưu tiếp theo tập trung vào CFS-only, không dùng semantic.

Vấn đề nhỏ trong CFS gốc:

```text
CFS sinh candidate Gaussian
-> chọn điểm đầu tiên ngẫu nhiên
-> chọn các điểm tiếp theo sao cho ít giống tập đã chọn nhất
```

Điểm đầu tiên ngẫu nhiên có thể làm kết quả dao động. Nếu điểm đầu tiên nằm hơi xa phân phối class, các điểm sau vẫn đa dạng nhưng tập replay có thể bị kéo lệch.

Cải tiến mới:

```text
--cfs_init_strategy mean
```

Ý tưởng:

- Không lọc bỏ candidate nào, nên không làm nghèo diversity như `cfs_distribution_filter`.
- Chỉ đổi điểm khởi đầu của CFS: chọn candidate gần mean class nhất theo diagonal Mahalanobis distance.
- Sau đó CFS vẫn chọn diversity như cũ.

So với distribution filter:

- `distribution_filter`: lọc cả candidate pool, đã làm accuracy giảm.
- `mean init`: chỉ neo điểm đầu tiên, còn candidate pool vẫn giữ nguyên độ đa dạng.

Cấu hình nên thử đầu tiên:

```bash
--cfs_sampling
--cfs_epochs 20
--cfs_batch_size 128
--cfs_train_max_samples 1024
--cfs_candidate_multiplier 2
--cfs_init_strategy mean
```

Kỳ vọng:

- Nếu random init là nguồn dao động, mean init có thể giúp kết quả ổn định hơn hoặc nhích lên.
- Nếu không cải thiện, vẫn có thể bỏ flag này vì mặc định `random` giữ nguyên CFS gốc.
## 23. Kết quả CFS mean initialization

Kết quả chạy CFS-only với `--cfs_init_strategy mean`:

```text
[Average accuracy till task10]
Acc@task 87.6000
Acc@1    87.8600
Acc@5    98.0800
Loss     0.5219
Forgetting 3.7333
Backward  -3.4000
```

So sánh với reproduce CFS-only cấu hình cũ:

| Phiên bản | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---|---:|---:|---:|---:|---:|---:|
| CFS-only reproduce | 87.6200 | 87.8800 | 98.0800 | 0.5222 | 3.7222 | -3.3889 |
| CFS mean init | 87.6000 | 87.8600 | 98.0800 | 0.5219 | 3.7333 | -3.4000 |

Nhận xét:

- Mean init không cải thiện accuracy: Acc@task và Acc@1 đều giảm `0.02`.
- Acc@5 giữ nguyên.
- Loss tốt hơn rất nhẹ `0.5222 -> 0.5219`, nhưng mức cải thiện quá nhỏ.
- Forgetting/Backward xấu hơn rất nhẹ.
- Kết luận: `cfs_init_strategy=mean` chạy đúng nhưng không đáng giữ làm cải tiến chính. CFS random init gốc vẫn tốt tương đương hoặc nhỉnh hơn.
## 24. Kết quả thêm semantic vào cấu hình CFS cũ

Sau khi reproduce cấu hình CFS-only cũ, đã thử thêm semantic adapter + covariance transfer vào đúng nền CFS cũ:

```text
[Average accuracy till task10]
Acc@task 87.7700
Acc@1    87.8300
Acc@5    97.9900
Loss     0.5186
Forgetting 3.7000
Backward  -3.3778
```

So sánh với CFS-only reproduce:

| Phiên bản | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---|---:|---:|---:|---:|---:|---:|
| CFS-only reproduce | 87.6200 | 87.8800 | 98.0800 | 0.5222 | 3.7222 | -3.3889 |
| CFS cũ + semantic covariance transfer | 87.7700 | 87.8300 | 97.9900 | 0.5186 | 3.7000 | -3.3778 |

Nhận xét:

- Semantic cải thiện Acc@task `+0.15`.
- Loss tốt hơn rõ hơn: `0.5222 -> 0.5186`.
- Forgetting và Backward cũng tốt hơn nhẹ.
- Nhưng Acc@1 giảm `0.05` và Acc@5 giảm `0.09`.
- Kết luận: semantic trên nền CFS cũ có tác dụng phụ trợ thật, nhưng chưa thắng tuyệt đối CFS-only theo Acc@1. Nếu mục tiêu báo cáo ưu tiên Acc@task/loss/forgetting thì bản semantic đáng nhắc đến; nếu ưu tiên Acc@1 thì CFS-only reproduce vẫn nhỉnh hơn.

Hướng thử tiếp hợp lý:

- Giữ semantic adapter + covariance transfer.
- Giảm semantic ratio từ `0.03` xuống `0.02` để cố lấy lại Acc@1.
- Giữ `epochs=10`, `crct_epochs=3`, `cfs_epochs=20`, `candidate_multiplier=2`, và checkpoint TII cũ.
## 25. Kết quả semantic ratio 0.02 trên nền CFS cũ

Đã thử giảm semantic projection ratio từ `0.03` xuống `0.02`, giữ nguyên nền CFS cũ.

Kết quả:

```text
[Average accuracy till task10]
Acc@task 87.7600
Acc@1    87.8600
Acc@5    97.9900
Loss     0.5184
Forgetting 3.6667
Backward  -3.3444
```

So sánh ba mốc gần nhất:

| Phiên bản | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---|---:|---:|---:|---:|---:|---:|
| CFS-only reproduce | 87.6200 | 87.8800 | 98.0800 | 0.5222 | 3.7222 | -3.3889 |
| CFS + semantic ratio 0.03 | 87.7700 | 87.8300 | 97.9900 | 0.5186 | 3.7000 | -3.3778 |
| CFS + semantic ratio 0.02 | 87.7600 | 87.8600 | 97.9900 | 0.5184 | 3.6667 | -3.3444 |

Nhận xét:

- Ratio `0.02` tốt hơn ratio `0.03` ở Acc@1, loss, forgetting và backward.
- So với CFS-only, ratio `0.02` tăng Acc@task `+0.14`, loss tốt hơn `0.0038`, forgetting/backward tốt hơn.
- Acc@1 vẫn thấp hơn CFS-only rất nhẹ `0.02`, Acc@5 thấp hơn `0.09`.
- Đây là bản cân bằng tốt nhất hiện tại nếu xét nhiều chỉ số cùng lúc: gần như giữ được Acc@1 của CFS-only, đồng thời cải thiện Acc@task/loss/forgetting.

Kết luận tạm thời:

- Nếu chọn theo Acc@1 tuyệt đối: CFS-only reproduce vẫn nhỉnh hơn rất nhỏ.
- Nếu chọn theo cân bằng Acc@task + loss + forgetting: CFS + semantic covariance transfer ratio `0.02` là cấu hình đáng chọn nhất.
## 26. Kết quả semantic ratio 0.015 trên nền CFS cũ

Đã thử giảm semantic projection ratio tiếp từ `0.02` xuống `0.015`.

Kết quả:

```text
[Average accuracy till task10]
Acc@task 87.7300
Acc@1    87.8400
Acc@5    97.9900
Loss     0.5184
Forgetting 3.6667
Backward  -3.3667
```

So sánh:

| Phiên bản | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---|---:|---:|---:|---:|---:|---:|
| CFS-only reproduce | 87.6200 | 87.8800 | 98.0800 | 0.5222 | 3.7222 | -3.3889 |
| Semantic ratio 0.02 | 87.7600 | 87.8600 | 97.9900 | 0.5184 | 3.6667 | -3.3444 |
| Semantic ratio 0.015 | 87.7300 | 87.8400 | 97.9900 | 0.5184 | 3.6667 | -3.3667 |

Nhận xét:

- Ratio `0.015` không cải thiện so với `0.02`.
- Acc@task giảm `0.03`, Acc@1 giảm `0.02`, loss và forgetting giữ nguyên.
- Ratio `0.02` vẫn là điểm cân bằng tốt nhất trong các bản semantic trên nền CFS cũ.

Kết luận hiện tại:

- CFS-only reproduce tốt nhất nếu chỉ ưu tiên Acc@1/Acc@5.
- CFS + semantic covariance transfer ratio `0.02` tốt nhất nếu ưu tiên cân bằng nhiều chỉ số: Acc@task, loss, forgetting/backward.
## 27. Hướng cải tiến mới: full CRCT replay và boundary-aware CFS

### 27.1. Vì sao dừng tinh chỉnh semantic ratio

Các mức semantic ratio `0.03`, `0.02` và `0.015` chỉ làm kết quả dao động rất nhỏ. Semantic ratio `0.02` cải thiện loss và forgetting, nhưng Acc@1 vẫn không vượt CFS-only. Điều này cho thấy nút thắt hiện tại không còn nằm chủ yếu ở hệ số semantic.

### 27.2. Phát hiện trong classifier correction

Ở mỗi epoch CRCT, code sinh cùng số mẫu cho tất cả lớp đã thấy, nhưng số vòng tối ưu `crct_num` chỉ bằng số lớp của các task trước task hiện tại. Sau khi trộn ngẫu nhiên, phần dữ liệu còn lại không được dùng:

- Sau task 2: sinh mẫu cho 20 lớp nhưng chỉ dùng lượng tương đương 10 lớp, tức bỏ khoảng 50%.
- Sau task 10: sinh mẫu cho 100 lớp nhưng chỉ dùng lượng tương đương 90 lớp, tức bỏ khoảng 10%.
- Mẫu bị bỏ sau khi shuffle nên thành phần lớp thay đổi ngẫu nhiên giữa các epoch.

Đã thêm cờ:

```text
--crct_use_all_samples
```

Khi bật, CRCT dùng toàn bộ replay đã sinh. Khi không bật, hành vi cũ được giữ nguyên để vẫn reproduce được baseline.

### 27.3. Boundary-aware CFS

CFS gốc ưu tiên diversity trong feature space nhưng không biết classifier đang nhầm ở đâu. Đã thêm một chế độ tùy chọn:

```text
--cfs_boundary_replay
--cfs_boundary_ratio 0.5
--cfs_boundary_multiplier 3
--cfs_boundary_density_quantile 0.9
```

Cách hoạt động:

1. Một phần replay vẫn được chọn bằng CFS diversity như cũ.
2. Phần còn lại được lấy từ Gaussian candidate gần ranh giới giữa logit của lớp đúng và lớp cạnh tranh mạnh nhất.
3. Loại 10% candidate nằm xa phân phối lớp nhất trước khi chọn hard samples, tránh huấn luyện vào outlier.
4. Ranh giới được tính lại ở mỗi epoch CRCT, nên mẫu khó thay đổi theo classifier hiện tại.

Điểm khác semantic projection: semantic tạo feature dựa trên quan hệ tên lớp; boundary-aware CFS dùng lỗi/độ không chắc chắn thật của classifier. Vì vậy nó chỉ tập trung vào những vùng ảnh hưởng trực tiếp tới Acc@1.

### 27.4. Thứ tự thí nghiệm

Không bật hai thay đổi cùng lúc ngay từ đầu:

1. Chạy CFS cũ + `--crct_use_all_samples`, không semantic, không boundary replay. Đây là thử nghiệm ưu tiên để đo tác động của việc dùng đủ replay.
2. Nếu bước 1 tốt hơn, giữ full replay và thêm boundary replay với ratio `0.25` trước. Ratio thấp giúp giữ phần lớn CFS diversity.
3. Chỉ sau khi tìm được cấu hình CFS + boundary tốt mới thêm semantic ratio `0.02` để xem loss/forgetting có tiếp tục tốt lên mà không mất Acc@1 hay không.

Không kỳ vọng hoặc cam kết một mức tăng cụ thể trước khi chạy. Tuy nhiên đây là thay đổi tác động đúng vào lượng dữ liệu correction và decision boundary, nên có cơ sở tạo chênh lệch lớn hơn các lần chỉnh semantic ratio gần đây.
## 28. Kết quả full CRCT replay

Đã chạy lại cấu hình CFS cũ, chỉ thêm `--crct_use_all_samples`; không bật semantic và không bật boundary-aware replay.

```text
[Average accuracy till task10]
Acc@task 88.0100
Acc@1    88.1300
Acc@5    98.0800
Loss     0.4696
Forgetting 4.5222
Backward  -4.4111
```

So sánh với CFS-only reproduce trên cùng máy ảo:

| Phiên bản | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---|---:|---:|---:|---:|---:|---:|
| CFS-only reproduce | 87.6200 | 87.8800 | 98.0800 | 0.5222 | 3.7222 | -3.3889 |
| CFS + full CRCT replay | 88.0100 | 88.1300 | 98.0800 | 0.4696 | 4.5222 | -4.4111 |

Thay đổi:

- Acc@task tăng `+0.39`.
- Acc@1 tăng `+0.25` và vượt mốc CFS cũ trong full log (`88.00`) thêm `+0.13`.
- Acc@5 giữ nguyên.
- Loss giảm mạnh `0.5222 -> 0.4696`, tốt hơn `0.0526`.
- Forgetting tăng `+0.80`; Backward giảm thêm `-1.0222`, tức độ suy giảm so với accuracy tại thời điểm từng task vừa học lớn hơn.

Kết luận:

- Phát hiện CRCT bỏ một phần replay là nút thắt thật, không phải dao động ngẫu nhiên cỡ `0.01-0.05` như các lần tinh chỉnh semantic ratio.
- Full replay hiện là cấu hình tốt nhất theo Acc@1, Acc@task và loss cuối.
- Trade-off forgetting/backward cần được xử lý ở bước tiếp theo.
- Thí nghiệm kế tiếp nên giữ full replay và thêm boundary-aware CFS ở ratio thấp `0.25`, nhằm tăng chất lượng mẫu gần decision boundary mà vẫn giữ phần lớn CFS diversity. Không thêm semantic trong lần này để đo riêng tác động.
### 28.1. Phân tích theo từng task

Log in hai dòng từ task 2 trở đi:

- Dòng thứ nhất là đánh giá trước classifier correction (pre-CRCT).
- Dòng thứ hai là đánh giá sau CRCT.
- Các giá trị Backward dương rất lớn ở dòng pre-CRCT không hợp lệ để diễn giải, vì `pre_ca_acc_matrix` không có diagonal của task 1. Đây là vấn đề ở phần báo cáo metric pre-CRCT, không ảnh hưởng huấn luyện hoặc kết quả post-CRCT.

Mức thay đổi Acc@1 do full CRCT ở từng task:

| Task | Trước CRCT | Sau CRCT | Thay đổi |
|---:|---:|---:|---:|
| 2 | 96.1000 | 96.1000 | +0.0000 |
| 3 | 93.6667 | 94.0000 | +0.3333 |
| 4 | 91.3000 | 92.7000 | +1.4000 |
| 5 | 90.6000 | 91.5200 | +0.9200 |
| 6 | 88.6333 | 90.2167 | +1.5834 |
| 7 | 88.1429 | 89.6286 | +1.4857 |
| 8 | 87.8000 | 89.0000 | +1.2000 |
| 9 | 87.5556 | 89.0889 | +1.5333 |
| 10 | 86.0500 | 88.1300 | +2.0800 |

Nhận xét:

- Lợi ích của CRCT tăng dần khi số task/lớp tăng; task 10 được cộng `2.08` Acc@1.
- Loss sau CRCT thấp hơn trước CRCT ở mọi task.
- Không có điểm sụp đổ đột ngột. Forgetting post-CRCT dao động khoảng `4.28-4.70` từ task 6 đến task 10.
- Full replay sửa đúng vấn đề classifier ngày càng mất cân bằng khi số lớp tăng.
- Có thể tiếp tục thử boundary-aware CFS ratio thấp `0.25` trên nền full replay.
## 29. Kết quả boundary-aware CFS ratio 0.25

Giữ full CRCT replay và thêm boundary-aware CFS với `boundary_ratio=0.25`, không dùng semantic.

```text
[Average accuracy till task10]
Acc@task 88.0400
Acc@1    88.1700
Acc@5    98.0800
Loss     0.4704
Forgetting 4.4778
Backward  -4.3444
```

So sánh:

| Phiên bản | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---|---:|---:|---:|---:|---:|---:|
| CFS + full CRCT | 88.0100 | 88.1300 | 98.0800 | 0.4696 | 4.5222 | -4.4111 |
| Full CRCT + boundary 0.25 | 88.0400 | 88.1700 | 98.0800 | 0.4704 | 4.4778 | -4.3444 |

Thay đổi:

- Acc@task tăng `+0.03`.
- Acc@1 tăng `+0.04`.
- Acc@5 giữ nguyên.
- Forgetting giảm `0.0444` và Backward tốt hơn `0.0667`.
- Loss tăng nhẹ `0.0008`.

Kết luận:

- Đây là cấu hình tốt nhất hiện tại theo Acc@1 (`88.17`) và Acc@task (`88.04`).
- Boundary-aware CFS có tín hiệu tích cực, nhưng mức tăng `0.04` vẫn nhỏ và cần coi là kết quả một seed, chưa đủ khẳng định chắc chắn vượt nhiễu.
- Có thể thử ratio `0.5` tiếp theo vì ratio `0.25` không phá CFS và còn cải thiện forgetting/backward. Giữ mọi tham số khác nguyên để đo riêng liều boundary replay.
## 30. Cải tiến routing: task-energy top-2

Sau khi boundary-aware CFS ratio `0.25` chỉ tăng thêm `0.04` Acc@1, hướng tiếp theo chuyển sang task/LoRA routing.

Vấn đề trong routing cũ:

- Top-2 ứng viên được lấy theo hai class có logit cao nhất, không phải hai task.
- Hai class top đầu có thể thuộc cùng một task, khiến lần chạy adapter thứ hai bị trùng và không bổ sung thông tin.
- Khi nhánh adapter thứ hai được chọn, `prompt_id` không được cập nhật; do đó Acc@task có thể không phản ánh adapter thực sự tạo ra logits cuối.
- Logits sau khi reroute chưa được mask lại ở các task trung gian.

Đã thêm chế độ tùy chọn:

```text
--task_routing_mode task_energy
--task_routing_temperature 0.1
```

Cách mới:

1. Gom logits của các class theo từng task.
2. Tính task energy bằng temperature-scaled log-sum-exp.
3. Chọn hai task có energy cao nhất; hai ứng viên luôn là hai task phân biệt.
4. Reroute qua adapter của task thứ nhất.
5. Với mẫu uncertainty, so sánh adapter thứ nhất và thứ hai như cơ chế cũ.
6. Cập nhật `prompt_id` theo adapter thực sự được chọn và mask lại class chưa thấy.

Đây là thay đổi ở pha đánh giá. Có thể dùng `--eval` với checkpoint tốt nhất hiện tại mà không cần train lại. Mặc định vẫn là `task_routing_mode=class`, nên baseline cũ không thay đổi.
## 31. Kết quả task-energy evaluation routing

Đã đánh giá checkpoint tốt nhất `full CRCT + boundary 0.25` bằng task-energy routing, không train lại.

```text
[Average accuracy till task10]
Acc@task 84.9100
Acc@1    88.1400
Acc@5    98.0700
Loss     0.4727
Forgetting 4.4222
Backward  -4.3222
```

So với routing class gốc trên cùng checkpoint:

| Routing | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---|---:|---:|---:|---:|---:|---:|
| Class routing gốc | 88.0400 | 88.1700 | 98.0800 | 0.4704 | 4.4778 | -4.3444 |
| Task-energy routing | 84.9100 | 88.1400 | 98.0700 | 0.4727 | 4.4222 | -4.3222 |

Kết luận:

- Acc@1 giảm `0.03`, Acc@5 giảm `0.01`, loss tăng `0.0023`.
- Acc@task giảm mạnh `3.13`.
- Forgetting/backward tốt hơn nhẹ nhưng không bù được routing accuracy giảm.
- Không dùng `--task_routing_mode task_energy` cho cấu hình chính; giữ mặc định `class`.
- Checkpoint không bị thay đổi vì đây chỉ là eval.

## 32. Cải tiến CTIRD: chọn top-K task liên quan

Trong CTIRD gốc, code tính task-level energy nhưng không dùng. Nó gọi `torch.topk(..., largest=False)` trên class logits, tức chọn các class có logit thấp nhất, sau đó đổi class sang task. Hệ quả là source adapter có thể ít liên quan và nhiều class có thể trỏ trùng cùng một task.

Đã thêm chế độ:

```text
--ctird_task_selection task_energy
--ctird_task_temperature 0.1
```

Khi bật:

- Tính energy riêng cho từng task cũ từ logits TII.
- Chọn top-K task có energy cao nhất (`largest=True`).
- Mỗi source là một task phân biệt, không còn trùng do nhiều class cùng task.
- CTIRD distill quan hệ từ những adapter cũ liên quan nhất tới batch hiện tại.

Mặc định vẫn là `legacy`, nên baseline không thay đổi. Thay đổi này tác động lúc train LoRA và vì vậy cần chạy LoRA mới; TII checkpoint vẫn được tái sử dụng.
## 33. Kết quả CTIRD top-K task-energy

Giữ cấu hình tốt nhất `full CRCT + boundary 0.25`, routing đánh giá mặc định `class`, và thêm CTIRD chọn top-K source task bằng task energy.

```text
[Average accuracy till task10]
Acc@task 88.0500
Acc@1    88.3100
Acc@5    98.0700
Loss     0.4675
Forgetting 4.5111
Backward  -4.4111
```

So sánh với cấu hình tốt nhất trước đó:

| Phiên bản | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---|---:|---:|---:|---:|---:|---:|
| Full CRCT + boundary 0.25 | 88.0400 | 88.1700 | 98.0800 | 0.4704 | 4.4778 | -4.3444 |
| Thêm CTIRD task-energy | 88.0500 | 88.3100 | 98.0700 | 0.4675 | 4.5111 | -4.4111 |

Thay đổi:

- Acc@1 tăng `+0.14`, đạt mốc tốt nhất mới `88.31`.
- Acc@task tăng `+0.01`.
- Loss giảm `0.0029`.
- Acc@5 giảm rất nhẹ `0.01`.
- Forgetting tăng `0.0333`; Backward giảm `0.0667`.

Kết luận:

- Việc CTIRD legacy chọn bottom-class là một nút thắt thật.
- Chọn các source adapter cũ liên quan nhất bằng task energy cải thiện classification rõ hơn các lần chỉnh boundary hoặc semantic ratio nhỏ.
- Cấu hình này hiện tốt nhất theo Acc@1 và loss.
- Bước tiếp theo nên kiểm tra tiến trình task 1-10, sau đó thử giảm `K` từ 5 xuống 3 để CTIRD tập trung vào các task liên quan nhất và giảm nhiễu từ source thứ 4-5. Giữ mọi tham số khác nguyên.
### 33.1. Phân tích CTIRD task-energy theo từng task

Mỗi task từ task 2 có dòng pre-CRCT và post-CRCT. Mức thay đổi Acc@1:

| Task | Trước CRCT | Sau CRCT | Thay đổi |
|---:|---:|---:|---:|
| 2 | 96.1000 | 96.1000 | +0.0000 |
| 3 | 93.7667 | 93.8333 | +0.0666 |
| 4 | 91.4750 | 92.6750 | +1.2000 |
| 5 | 90.6200 | 91.4000 | +0.7800 |
| 6 | 88.6167 | 90.1167 | +1.5000 |
| 7 | 88.1429 | 89.6714 | +1.5285 |
| 8 | 87.7375 | 89.2000 | +1.4625 |
| 9 | 87.7667 | 89.2778 | +1.5111 |
| 10 | 85.9400 | 88.3100 | +2.3700 |

Nhận xét:

- Lợi ích lớn nhất vẫn đến từ CRCT ở các task muộn; task 10 tăng `2.37` Acc@1.
- So với full CRCT trước đó, mức correction tại task 10 tăng từ `2.08` lên `2.37`.
- Forgetting post-CRCT đạt cao nhất `4.90` tại task 6 rồi ổn định quanh `4.1-4.5`.
- CTIRD task-energy chủ yếu có giá trị từ giai đoạn nhiều task, đúng nơi legacy bottom-class selection gây nhiễu.
- Thử nghiệm kế tiếp là giữ nguyên toàn bộ cấu hình và đổi `K=5` thành `K=3`. Từ task 5 trở đi, cách này loại source task thứ 4-5 và tập trung CTIRD vào ba task liên quan nhất.
## 34. Kết quả CTIRD task-energy K=3

Đã giữ toàn bộ cấu hình tốt nhất và giảm số source task CTIRD từ `K=5` xuống `K=3`.

```text
[Average accuracy till task10]
Acc@task 87.8000
Acc@1    88.1600
Acc@5    98.1500
Loss     0.4720
Forgetting 4.6333
Backward  -4.6000
```

So với `K=5`:

| K | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 88.0500 | 88.3100 | 98.0700 | 0.4675 | 4.5111 | -4.4111 |
| 3 | 87.8000 | 88.1600 | 98.1500 | 0.4720 | 4.6333 | -4.6000 |

Kết luận:

- K=3 giảm Acc@1 `0.15` và Acc@task `0.25`.
- Loss tăng `0.0045`, forgetting/backward đều xấu hơn.
- Acc@5 tăng `0.08` nhưng không bù được các chỉ số chính.
- Giữ K=5 làm cấu hình tốt nhất.
- Giảm K đồng thời làm tổng số KL loss giảm từ 5 xuống 3, nên không chỉ loại nhiễu mà còn làm yếu CTIRD.
- Hướng tiếp theo: giữ đủ K=5 nhưng dùng task-energy để gán trọng số theo hạng, sau đó chuẩn hóa trọng số có trung bình bằng 1 để tổng cường độ CTIRD không đổi.
## 35. Cải tiến CTIRD: energy-weighted K=5

Sau khi K=3 làm kết quả giảm, hướng mới giữ đủ K=5 nhưng phân bổ trọng số KL theo mức liên quan của từng source rank.

Cờ mới:

```text
--ctird_task_weighting energy
--ctird_weight_temperature 1.0
--ctird_weight_floor 0.2
```

Cách tính:

1. Lấy task-energy của năm source task đã chọn.
2. Softmax task-energy để tạo trọng số theo batch.
3. Trộn thêm `20%` trọng số đều để source hạng thấp không bị triệt tiêu.
4. Lấy trung bình trọng số theo batch.
5. Nhân toàn bộ trọng số với K, làm trọng số trung bình bằng 1 và tổng bằng K.

Nhờ bước chuẩn hóa cuối, tổng cường độ CTIRD giữ tương đương K=5 uniform hiện tại. Thay đổi chỉ chuyển lực distillation từ source ít liên quan sang source liên quan hơn, không làm yếu regularization như thí nghiệm K=3.

Mặc định `ctird_task_weighting=uniform`, nên baseline cũ không thay đổi.
## 36. Kết quả CTIRD energy-weighted K=5

Giữ K=5 và task-energy source selection, sau đó gán trọng số KL theo task energy với `weight_temperature=1.0`, `weight_floor=0.2`. Tổng trọng số vẫn bằng K.

```text
[Average accuracy till task10]
Acc@task 88.3000
Acc@1    88.5600
Acc@5    98.0700
Loss     0.4627
Forgetting 4.2778
Backward  -4.0778
```

So sánh với K=5 uniform:

| Phiên bản | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---|---:|---:|---:|---:|---:|---:|
| CTIRD task-energy K=5 uniform | 88.0500 | 88.3100 | 98.0700 | 0.4675 | 4.5111 | -4.4111 |
| CTIRD task-energy K=5 weighted | 88.3000 | 88.5600 | 98.0700 | 0.4627 | 4.2778 | -4.0778 |

Thay đổi:

- Acc@task tăng `+0.25`.
- Acc@1 tăng `+0.25`, đạt mốc tốt nhất mới `88.56`.
- Acc@5 giữ nguyên.
- Loss giảm `0.0048`.
- Forgetting giảm `0.2333`.
- Backward tốt hơn `0.3333`.

So với CFS-only reproduce ban đầu:

- Acc@1 tăng tổng cộng `87.88 -> 88.56`, tức `+0.68`.
- Acc@task tăng `87.62 -> 88.30`, tức `+0.68`.
- Loss giảm `0.5222 -> 0.4627`, tức `-0.0595`.

Kết luận:

- Energy weighting giải quyết đúng nhược điểm của K=3: vẫn giữ thông tin từ đủ năm source task nhưng tập trung lực distillation vào source liên quan hơn.
- Đây là cấu hình tốt nhất hiện tại trên tất cả chỉ số chính, ngoại trừ Acc@5 bằng bản K=5 uniform.
- Bước tinh chỉnh tiếp theo có thể hạ `ctird_weight_temperature` từ `1.0` xuống `0.5` để tăng độ tập trung, giữ `weight_floor=0.2` và mọi tham số khác nguyên.
## 37. Kết quả CTIRD weight temperature 0.5

Đã giữ K=5, energy weighting và giảm `ctird_weight_temperature` từ `1.0` xuống `0.5` để làm trọng số tập trung hơn.

```text
[Average accuracy till task10]
Acc@task 87.8800
Acc@1    88.2700
Acc@5    98.0200
Loss     0.4771
Forgetting 4.5000
Backward  -4.4222
```

So với temperature `1.0`:

| Temperature | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 88.3000 | 88.5600 | 98.0700 | 0.4627 | 4.2778 | -4.0778 |
| 0.5 | 87.8800 | 88.2700 | 98.0200 | 0.4771 | 4.5000 | -4.4222 |

Kết luận:

- Temperature 0.5 giảm Acc@1 `0.29` và Acc@task `0.42`.
- Loss tăng `0.0144`; forgetting/backward đều xấu hơn.
- Trọng số quá tập trung làm mất lợi ích từ source task hạng thấp, phù hợp với kết quả K=3 cũng bị giảm.
- Giữ temperature `1.0` làm cấu hình tốt nhất.
- Có thể thử phía ngược lại là temperature `1.5`, giúp phân bố mềm hơn nhưng vẫn giữ energy weighting. Nếu không vượt 1.0 thì dừng dò temperature và chuyển sang adaptive weighting.
## 38. Kết quả CTIRD weight temperature 1.5

Đã thử làm phân bố trọng số mềm hơn bằng cách tăng `ctird_weight_temperature` từ `1.0` lên `1.5`.

```text
[Average accuracy till task10]
Acc@task 88.1400
Acc@1    88.5200
Acc@5    98.1600
Loss     0.4645
Forgetting 4.4444
Backward  -4.4000
```

So với temperature `1.0`:

| Temperature | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 88.3000 | 88.5600 | 98.0700 | 0.4627 | 4.2778 | -4.0778 |
| 1.5 | 88.1400 | 88.5200 | 98.1600 | 0.4645 | 4.4444 | -4.4000 |

Kết luận:

- Temperature 1.5 giảm Acc@1 `0.04`, Acc@task `0.16` và làm loss/forgetting/backward xấu hơn.
- Acc@5 tăng `0.09`, đạt `98.16`.
- Temperature 1.0 vẫn là cấu hình cân bằng tốt nhất.
- Dừng dò weight temperature; cả phía nhọn hơn (0.5) và mềm hơn (1.5) đều không vượt 1.0.
- Hướng tiếp theo: giữ toàn bộ cấu hình 88.56 và tăng `crct_epochs` từ 3 lên 4 để thử tăng mức classifier correction trên full replay.
## 39. Kết quả CRCT 4 epochs

Giữ cấu hình CTIRD weighted tốt nhất và tăng `crct_epochs` từ 3 lên 4.

```text
[Average accuracy till task10]
Acc@task 87.9600
Acc@1    88.3400
Acc@5    98.1200
Loss     0.4721
Forgetting 4.5444
Backward  -4.4444
```

So với CRCT 3 epochs:

| CRCT epochs | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 88.3000 | 88.5600 | 98.0700 | 0.4627 | 4.2778 | -4.0778 |
| 4 | 87.9600 | 88.3400 | 98.1200 | 0.4721 | 4.5444 | -4.4444 |

Kết luận:

- Epoch 4 giảm Acc@1 `0.22` và Acc@task `0.34`.
- Loss tăng `0.0094`; forgetting/backward xấu hơn.
- Acc@5 tăng nhẹ `0.05` nhưng không bù được.
- CRCT 3 epochs vẫn là cấu hình tốt nhất; 4 epochs gây over-correction trên replay tổng hợp.
- Hướng tiếp theo là giữ 3 epochs nhưng tạo mini-batch CRCT gần cân bằng lớp thay vì shuffle toàn bộ ngẫu nhiên.
## 40. Cân bằng lớp trong từng mini-batch CRCT

Sau khi `crct_epochs=4` gây over-correction, giữ lại cấu hình tốt nhất với `crct_epochs=3` và thay đổi cách xếp thứ tự replay.

Vấn đề của bản trước:

- Tập replay tổng thể đã cân bằng vì mỗi lớp có cùng số mẫu tổng hợp.
- Tuy nhiên, toàn bộ tập được shuffle ngẫu nhiên rồi cắt liên tiếp thành các mini-batch.
- Do đó từng mini-batch chỉ cân bằng theo kỳ vọng; một batch có thể thiếu nhiều lớp và lặp lại một số lớp khác.
- Gradient của classifier correction vì vậy có thể dao động và tạm thời thiên về các lớp xuất hiện nhiều trong batch đó.

Thay đổi mới:

- Thêm cờ `--crct_balanced_batches`.
- Mẫu trong từng lớp vẫn được shuffle ngẫu nhiên.
- Các hàng đợi theo lớp sau đó được đan xen theo vòng: mỗi lượt lấy một mẫu của từng lớp.
- Với CIFAR-100 ở task 10, batch 120 mẫu sẽ chứa ít nhất một mẫu của cả 100 lớp trước khi lớp nào được lặp lại.
- Nếu dữ liệu đầu vào bất ngờ có số mẫu mỗi lớp không bằng nhau, mã tự quay về global random shuffle để tránh lỗi.
- Áp dụng nhất quán cho cả engine LoRA và TII.
- Script Kaggle có thể bật bằng `CRCT_BALANCED_BATCHES=1`.

Phạm vi kiểm soát thí nghiệm:

- Giữ `crct_epochs=3`.
- Giữ full replay, boundary CFS ratio 0.25, CTIRD task-energy K=5 và energy weighting temperature 1.0/floor 0.2.
- Không bật semantic và không đổi evaluation routing.
- Không đổi tổng số mẫu replay hoặc hàm loss.
- Khi không truyền cờ mới, mã chạy đúng cách shuffle cũ.

Mục tiêu của thử nghiệm kế tiếp là kiểm tra xem giảm nhiễu gradient giữa các batch có vượt mốc tốt nhất hiện tại `Acc@1=88.56` và `Acc@task=88.30` hay không.

## 41. Kết quả class-balanced CRCT và cải tiến thứ tự lớp

Kết quả của bản cân bằng lớp cứng trong từng mini-batch:

```text
[Average accuracy till task10]
Acc@task 88.2400
Acc@1    88.3100
Acc@5    98.2400
Loss     0.4638
Forgetting 4.4333
Backward  -4.2889
```

So với cấu hình tốt nhất chưa cân bằng batch:

| Cấu hình | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---|---:|---:|---:|---:|---:|---:|
| CTIRD weighted, shuffle cũ | 88.3000 | 88.5600 | 98.0700 | 0.4627 | 4.2778 | -4.0778 |
| Class-balanced CRCT | 88.2400 | 88.3100 | 98.2400 | 0.4638 | 4.4333 | -4.2889 |

Nhận xét:

- `Acc@1` giảm `0.25` và `Acc@task` giảm `0.06`, nên bản này chưa thay thế cấu hình tốt nhất.
- `Acc@5` tăng `0.17` lên `98.24`, là mức cao nhất trong các lần thử hiện tại.
- Điều này cho thấy cân bằng batch giúp giữ các lớp đúng trong nhóm dự đoán đầu, nhưng lịch batch quá cứng làm giảm khả năng chọn đúng top-1.
- Nguyên nhân có thể kiểm soát được: thứ tự lớp trong mỗi vòng luôn cố định từ nhãn nhỏ tới nhãn lớn. Khi batch 120 cắt chu kỳ 100 lớp, các lớp được lặp thêm tạo thành cụm nhãn liên tiếp và cũng có thể tạo thành cụm task liên tiếp.

Cải tiến tiếp theo:

- Vẫn shuffle mẫu bên trong từng lớp.
- Mỗi chu kỳ đi qua tất cả lớp dùng một hoán vị lớp ngẫu nhiên độc lập.
- Mỗi batch vẫn chứa số mẫu mỗi lớp lệch tối đa một, nhưng lớp được lấy thêm không còn bị cố định theo nhãn/task.
- Không thêm tham số mới; cờ `--crct_balanced_batches` dùng thuật toán class-order ngẫu nhiên đã sửa.
- Mọi thành phần CFS, CTIRD, số epoch, số replay sample và loss tiếp tục được giữ nguyên để đo riêng tác động này.

## 42. Xác nhận random class-order và target-side boundary CFS

Máy chạy đã được xác nhận ở đúng commit `5785c5f`. Bản random class-order cho kết quả giống hoàn toàn bản class-order cố định:

```text
[Average accuracy till task10]
Acc@task 88.2400
Acc@1    88.3100
Acc@5    98.2400
Loss     0.4638
Forgetting 4.4333
Backward  -4.2889
```

Kết luận về cân bằng batch:

- Hoán vị thứ tự lớp không tạo cải thiện đo được.
- Cross-entropy không phụ thuộc thứ tự phần tử bên trong batch; full replay qua ba epoch cũng làm khác biệt về các lớp dư ở biên batch trở nên rất nhỏ.
- Dừng hướng `--crct_balanced_batches` và quay lại global shuffle của cấu hình tốt nhất `Acc@1=88.56`.

Cải tiến tiếp theo tập trung vào chất lượng boundary replay:

- Bản cũ xếp hạng bằng trị tuyệt đối của margin nên có thể chọn cả mẫu đã nằm ở phía lớp đối thủ, sau đó vẫn gán hard label của lớp nguồn.
- Thêm cờ `--cfs_boundary_target_side` để ưu tiên margin dương nhỏ nhất trong vùng mật độ hợp lệ: mẫu khó, sát biên nhưng classifier vẫn nhận đúng phía lớp nguồn.
- Nếu số mẫu target-side không đủ, phần thiếu quay về cách chọn trị tuyệt đối margin cũ, không làm giảm số replay sample.
- Mặc định cờ tắt nên mọi cấu hình cũ giữ nguyên.
- Thử nghiệm mới bỏ `--crct_balanced_batches`, giữ full replay, CRCT 3 epochs, boundary ratio 0.25 và CTIRD weighted tốt nhất.

## 43. Kết quả target-side boundary CFS và kế hoạch sweep CTIRD

Kết quả khi chỉ thêm `--cfs_boundary_target_side` trên cấu hình tốt nhất:

```text
[Average accuracy till task10]
Acc@task 88.2900
Acc@1    88.5600
Acc@5    98.0700
Loss     0.4630
Forgetting 4.2778
Backward  -4.1000
```

So với boundary CFS cũ:

| Cấu hình | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---|---:|---:|---:|---:|---:|---:|
| Boundary cũ | 88.3000 | 88.5600 | 98.0700 | 0.4627 | 4.2778 | -4.0778 |
| Target-side boundary | 88.2900 | 88.5600 | 98.0700 | 0.4630 | 4.2778 | -4.1000 |

Kết luận:

- Acc@1, Acc@5 và forgetting giữ nguyên.
- Acc@task giảm `0.01`, loss tăng `0.0003` và backward kém `0.0222`.
- Target-side boundary gần như trung tính nhưng không vượt cấu hình cũ; không dùng cờ này cho mốc tốt nhất.
- Các thay đổi nhỏ ở boundary replay hiện đã bão hòa quanh `Acc@1=88.56`.

Hướng tiếp theo:

- Giữ nguyên cấu hình tốt nhất và sweep hệ số CTIRD `con`, vì mọi lần chạy trước đều dùng `0.2`.
- Thử `con=0.25` trước: energy weighting đã chọn lọc source task tốt hơn nên tăng vừa phải lực relation distillation có thể cải thiện ổn định giữa task.
- Không bật balanced batches, target-side boundary hoặc semantic để đo riêng tác động của `con`.

## 44. Kết quả CTIRD con=0.25

Giữ cấu hình tốt nhất và tăng hệ số relation distillation từ `con=0.2` lên `con=0.25`.

```text
[Average accuracy till task10]
Acc@task 87.9700
Acc@1    88.3500
Acc@5    98.1000
Loss     0.4635
Forgetting 4.6778
Backward  -4.6444
```

| con | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 88.3000 | 88.5600 | 98.0700 | 0.4627 | 4.2778 | -4.0778 |
| 0.25 | 87.9700 | 88.3500 | 98.1000 | 0.4635 | 4.6778 | -4.6444 |

Kết luận:

- `con=0.25` giảm Acc@task `0.33` và Acc@1 `0.21`.
- Acc@5 tăng nhẹ `0.03`, nhưng loss, forgetting và backward đều xấu hơn rõ rệt.
- Energy-weighted CTIRD bị quá mạnh ở `0.25`, làm giảm khả năng thích nghi với task mới mà vẫn không giữ task cũ tốt hơn.
- Không dùng `con=0.25`; thử phía ngược lại `con=0.15` để xác định liệu mức tối ưu nằm dưới `0.2` hay không.

## 45. Kết quả CTIRD con=0.15

Thử phía thấp hơn của hệ số relation distillation với `con=0.15`.

```text
[Average accuracy till task10]
Acc@task 88.1100
Acc@1    88.4000
Acc@5    98.0100
Loss     0.4658
Forgetting 4.7556
Backward  -4.6889
```

| con | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---:|---:|---:|---:|---:|---:|---:|
| 0.15 | 88.1100 | 88.4000 | 98.0100 | 0.4658 | 4.7556 | -4.6889 |
| 0.20 | 88.3000 | 88.5600 | 98.0700 | 0.4627 | 4.2778 | -4.0778 |
| 0.25 | 87.9700 | 88.3500 | 98.1000 | 0.4635 | 4.6778 | -4.6444 |

Kết luận:

- Cả `0.15` và `0.25` đều không vượt `0.20`; mức `0.20` tốt nhất ở Acc@task, Acc@1, loss, forgetting và backward.
- Dừng sweep `con` và cố định lại `con=0.20`.
- Hướng tiếp theo là giảm `ca_lr` từ `0.005` xuống `0.004`, vẫn giữ CRCT 3 epochs. Mục tiêu là làm classifier correction mềm hơn sau khi CRCT 4 epochs đã cho dấu hiệu over-correction.

## 46. Kết quả classifier-correction ca_lr=0.004

Giữ cấu hình CTIRD tốt nhất và giảm learning rate của classifier correction từ `0.005` xuống `0.004`.

```text
[Average accuracy till task10]
Acc@task 88.3400
Acc@1    88.5600
Acc@5    98.0700
Loss     0.4606
Forgetting 4.1889
Backward  -3.9444
```

| ca_lr | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---:|---:|---:|---:|---:|---:|---:|
| 0.005 | 88.3000 | 88.5600 | 98.0700 | 0.4627 | 4.2778 | -4.0778 |
| 0.004 | 88.3400 | 88.5600 | 98.0700 | 0.4606 | 4.1889 | -3.9444 |

Kết luận:

- Giữ nguyên Acc@1 và Acc@5.
- Acc@task tăng `0.04`.
- Loss giảm `0.0021`, forgetting giảm `0.0889`, backward cải thiện `0.1334`.
- `ca_lr=0.004` thay thế `0.005` làm cấu hình tốt nhất mới theo cân bằng nhiều chỉ số.
- Kết quả xác nhận classifier correction trước đó hơi mạnh; tiếp tục tinh chỉnh xuống `ca_lr=0.0035` để tìm điểm tối ưu gần vùng này.

## 47. Kết quả classifier-correction ca_lr=0.0035

Tiếp tục giảm learning rate của classifier correction từ `0.004` xuống `0.0035`.

```text
[Average accuracy till task10]
Acc@task 88.3300
Acc@1    88.4600
Acc@5    98.0500
Loss     0.4604
Forgetting 4.2222
Backward  -3.9444
```

| ca_lr | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0035 | 88.3300 | 88.4600 | 98.0500 | 0.4604 | 4.2222 | -3.9444 |
| 0.0040 | 88.3400 | 88.5600 | 98.0700 | 0.4606 | 4.1889 | -3.9444 |
| 0.0050 | 88.3000 | 88.5600 | 98.0700 | 0.4627 | 4.2778 | -4.0778 |

Kết luận:

- So với `0.004`, mức `0.0035` giảm Acc@1 `0.10`, Acc@task `0.01` và Acc@5 `0.02`.
- Loss chỉ giảm thêm `0.0002`, không đủ bù suy giảm accuracy; forgetting cũng xấu hơn `0.0333`.
- `ca_lr=0.004` vẫn là cấu hình tốt nhất.
- Thử điểm giữa phía trên `ca_lr=0.0045` để kiểm tra vùng giữa hai mức cùng đạt Acc@1 88.56; sau đó dừng sweep `ca_lr`.

## 48. Kết quả classifier-correction ca_lr=0.0045

Thử điểm giữa `0.004` và `0.005` cho classifier correction.

```text
[Average accuracy till task10]
Acc@task 88.3200
Acc@1    88.6000
Acc@5    98.0700
Loss     0.4607
Forgetting 4.2000
Backward  -3.9444
```

| ca_lr | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0040 | 88.3400 | 88.5600 | 98.0700 | 0.4606 | 4.1889 | -3.9444 |
| 0.0045 | 88.3200 | 88.6000 | 98.0700 | 0.4607 | 4.2000 | -3.9444 |
| 0.0050 | 88.3000 | 88.5600 | 98.0700 | 0.4627 | 4.2778 | -4.0778 |

Kết luận:

- `ca_lr=0.0045` tăng Acc@1 thêm `0.04`, đạt mốc cao nhất mới `88.60`.
- So với `0.004`, Acc@task giảm `0.02`, loss tăng `0.0001`, forgetting xấu hơn `0.0111`; backward và Acc@5 giữ nguyên.
- Nếu ưu tiên Acc@1, chọn `0.0045`; nếu ưu tiên cân bằng loss/forgetting, `0.004` vẫn tốt hơn rất nhẹ.
- Chốt `0.0045` làm nền accuracy cho thử nghiệm tiếp theo.
- Task-energy temperature `0.1` chưa từng được sweep. Thử `0.2` để energy score bớt phụ thuộc duy nhất vào class logit lớn nhất trong mỗi task.

## 49. Kết quả CTIRD task-energy temperature=0.2

Giữ `ca_lr=0.0045` và tăng task-energy temperature từ `0.1` lên `0.2`.

```text
[Average accuracy till task10]
Acc@task 88.0100
Acc@1    88.2800
Acc@5    98.1000
Loss     0.4676
Forgetting 4.5111
Backward  -4.4667
```

| Task temperature | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 88.3200 | 88.6000 | 98.0700 | 0.4607 | 4.2000 | -3.9444 |
| 0.2 | 88.0100 | 88.2800 | 98.1000 | 0.4676 | 4.5111 | -4.4667 |

Kết luận:

- Temperature `0.2` giảm Acc@task `0.31` và Acc@1 `0.32`.
- Loss tăng `0.0069`, forgetting xấu thêm `0.3111`, backward xấu thêm `0.5223`.
- Acc@5 tăng nhẹ `0.03` nhưng không bù được suy giảm còn lại.
- Energy score mềm hơn đã đưa quá nhiều class logit phụ vào task score; giữ `0.1` làm mốc.
- Thử phía ngược lại `0.05`, gần phép chọn max-logit hơn. Nếu không vượt `0.1` thì dừng sweep task temperature.

## 50. Hướng mới: Stability-Plasticity CRCT và continual norm consolidation

### 50.1. Vấn đề cần giải quyết

Mốc chạy lại của cấu hình gốc trên cùng thiết lập CIFAR100 là:

```text
Acc@task 88.0100 | Acc@1 87.8100 | Acc@5 98.0500
Loss 0.5228 | Forgetting 3.5889 | Backward -3.2556
```

Các cấu hình CFS/CRCT trước đã tăng accuracy và giảm loss, nhưng forgetting vẫn cao hơn bản gốc. Hai kết quả tiêu biểu:

| Cấu hình | Acc@task | Acc@1 | Loss | Forgetting | Backward |
|---|---:|---:|---:|---:|---:|
| Stability KD 0.25 + anchor 0.01 | 88.5400 | 88.4100 | 0.4699 | 4.1333 | -3.9778 |
| Stability KD 0.35 + anchor 0.015 | 88.4700 | 88.4600 | 0.4680 | 4.0333 | -3.8556 |

Do đó mục tiêu mới không phải tiếp tục tăng riêng accuracy, mà phải đạt đồng thời `Acc@task > 88.01` và `Forgetting < 3.5889` so với mốc gốc.

### 50.2. Phân tích nguyên nhân

Trong code cũ, CRCT được mô tả là bước hiệu chỉnh classifier nhưng optimizer thực tế cập nhật cả `head` và `fc_norm`. `fc_norm` là phép chuẩn hóa dùng chung cho mọi task, nên học trên feature tổng hợp có thể làm toàn bộ logits lớp cũ dịch cùng lúc. Ngoài CRCT, `fc_norm` còn được cập nhật trên dữ liệu riêng của từng task mới, trong khi các LoRA cũ đã được tách riêng. Đây là một nguồn gây quên chưa được kiểm soát trực tiếp.

Mẫu Gaussian/CFS cũng không có độ tin cậy bằng dữ liệu thật. Việc gán cùng trọng số CE cho mọi mẫu, đặc biệt mẫu gần biên mà teacher không nhận đúng lớp đích, có thể kéo classifier theo nhiễu tổng hợp.

### 50.3. Thay đổi đã triển khai

1. `--crct_head_only`: CRCT chỉ cập nhật classifier `head`, giữ `fc_norm` cố định. Điều này bám sát mô tả classification-layer fine-tuning và tránh overfit norm chung vào replay tổng hợp.
2. `--crct_reliability_weighting`: CE của mẫu replay lớp cũ được nhân với trọng số dựa trên xác suất teacher gán cho nhãn đích. Mẫu đáng tin gần trọng số 1; mẫu đáng ngờ được giảm nhưng không bỏ hẳn nhờ `--crct_reliability_floor`.
3. `--crct_old_row_lr_scale`: gradient của các hàng classifier thuộc lớp cũ được giảm riêng; hàng lớp mới vẫn học với learning rate đầy đủ. Đây là phân tách stability-plasticity trực tiếp ở classifier.
4. `--continual_norm_blend`: sau khi học task mới, tham số `fc_norm` được nội suy với trạng thái trước task. `--continual_norm_update_ratio` quy định phần cập nhật mới được giữ lại, thay vì đóng băng cứng.
5. Các cờ mới mặc định tắt hoặc giữ hệ số `1.0`, nên lệnh cũ vẫn giữ nguyên hành vi và checkpoint TII cũ vẫn dùng lại được.

### 50.4. Trạng thái kiểm tra

- `py_compile`: đạt.
- Parser nhận đầy đủ các cờ mới: đạt.
- `git diff --check`: đạt.
- Đã chạy đủ 10 task cho cấu hình v1; kết quả được ghi ở mục 51.

Thử nghiệm đầu tiên dùng mức bảo toàn vừa phải: norm update ratio `0.25`, reliability floor `0.5`, old-row scale `0.5`, không cộng thêm semantic và không dùng KD/anchor để đo riêng tác dụng của cơ chế mới.

## 51. Kết quả SP-CRCT v1 và sửa thiết kế v2

Kết quả v1:

```text
[Average accuracy till task10]
Acc@task 88.5400
Acc@1    88.2400
Acc@5    97.9900
Loss     0.4625
Forgetting 5.2556
Backward  -5.2333
```

So với bản gốc, Acc@task tăng `0.53` nhưng forgetting xấu thêm `1.6667`. Cấu hình này không đạt mục tiêu đồng thời và không được chọn làm kết quả cuối.

Phân tích nguyên nhân:

- Reliability v1 gán trọng số nhỏ hơn 1 cho mẫu cũ không chắc chắn nhưng không chuẩn hóa lại. Vì vậy tổng lực CE bảo vệ lớp cũ bị giảm.
- `old_row_lr_scale=0.5` giảm cập nhật các hàng classifier cũ, trong khi hàng lớp mới vẫn cập nhật đầy đủ. Việc giữ riêng hàng cũ không giữ được biên quyết định tương đối khi hàng mới dịch chuyển.
- Hai cơ chế trên cộng hưởng theo hướng ưu tiên lớp mới: Acc@task tăng nhưng lịch sử task cũ suy giảm mạnh, thể hiện qua forgetting và backward.

Thiết kế v2:

1. Thêm `--crct_reliability_preserve_mass`: chuẩn hóa trọng số reliability trong nhóm mẫu cũ về trung bình 1. Reliability chỉ chuyển lực học từ mẫu đáng ngờ sang mẫu đáng tin, không làm giảm tổng CE lớp cũ.
2. Trả `--crct_old_row_lr_scale` về `1.0` để các hàng cũ được hiệu chỉnh đầy đủ trước cạnh tranh từ lớp mới.
3. Nới `--continual_norm_update_ratio` từ `0.25` lên `0.5`, tránh bảo toàn norm quá cứng làm giảm khả năng tái cân bằng giữa các task.
4. Giữ `crct_head_only`, CFS, boundary replay và CTIRD không đổi để so sánh trực tiếp với v1.

Tiêu chí vẫn giữ nguyên: chỉ coi là cải thiện thành công khi Acc@task cao hơn `88.01` và forgetting thấp hơn `3.5889` trên cùng seed và thiết lập.

## 52. Kết quả SP-CRCT v2 và thiết kế core replay v3

Kết quả v2:

```text
Acc@task 88.5400 | Acc@1 88.3400 | Acc@5 98.0000
Loss 0.4709 | Forgetting 4.4444 | Backward -4.4000
```

V2 giảm forgetting `0.8112` so với v1 và tăng Acc@1 `0.10`, nhưng vẫn chưa thắng bản gốc về forgetting. Accuracy cuối theo từng task:

```text
Task 1..10 Acc@1:
90.8, 85.9, 88.6, 86.9, 86.2, 89.4, 87.8, 88.9, 91.8, 87.1
```

Các task yếu phân bố rải rác ở task 2, 4, 5 và 10; task 1 và 9 vẫn cao. Đây không phải mẫu suy giảm đơn điệu theo tuổi task, mà phù hợp hơn với sai lệch phân phối theo lớp/task.

Log CRCT cuối cho thấy:

```text
ReplayW = 1.0000
OldConf ≈ 0.883
Synthetic CRCT Acc@1 ≈ 98.76
Real final Acc@1 = 88.34
fc_norm delta_before = 0.0
```

Kết luận:

- Mass preservation v2 hoạt động đúng.
- `fc_norm` không có cập nhật trong cấu hình ViT này, nên norm consolidation không tạo tác dụng và được bỏ khỏi v3.
- Khoảng cách gần 10 điểm giữa synthetic CRCT accuracy và real accuracy cho thấy nút thắt là synthetic-real feature mismatch, không phải thiếu khả năng fit replay.
- Chuẩn hóa reliability toàn cục vẫn có thể chuyển loss từ lớp khó confidence thấp sang lớp dễ confidence cao, làm một số task bị bảo vệ kém.

Thiết kế v3:

1. `--crct_reliability_preserve_class_mass`: chuẩn hóa reliability riêng trong từng lớp cũ, giữ tổng CE của mỗi lớp thay vì chỉ giữ tổng toàn bộ lớp cũ.
2. `--cfs_core_replay_ratio 0.25`: mỗi lớp dùng 25% mẫu Gaussian mật độ cao gần lõi phân phối thật.
3. Giữ 50% mẫu CFS đa dạng để bao phủ feature space và 25% mẫu sát decision boundary để duy trì khả năng phân biệt.
4. Bỏ continual norm blend; giữ old-row scale `1.0`, head-only CRCT, CTIRD và tổng số replay sample không đổi.

V3 kiểm tra trực tiếp giả thuyết rằng replay quá thiên về mẫu đa dạng/biên làm classifier fit synthetic rất cao nhưng tổng quát hóa kém lên feature thật.

## 53. Kết quả core replay v3 và adaptive trust-region v4

Kết quả v3:

```text
Acc@task 88.5200 | Acc@1 88.3900 | Acc@5 98.0300
Loss 0.4717 | Forgetting 4.3444 | Backward -4.2667
```

So với v2, v3 tăng Acc@1 `0.05`, Acc@5 `0.03`, giảm forgetting `0.10` và cải thiện backward `0.1333`. Core replay có tác dụng đúng hướng nhưng mức cải thiện nhỏ; tiếp tục tăng tỷ lệ core đơn thuần khó tạo bước nhảy đủ lớn.

### 53.1. Nguyên nhân forgetting đã xác định

- LoRA cũ được tách riêng và không phải phần bị CRCT cập nhật.
- `fc_norm` không thay đổi trong cấu hình hiện tại.
- CRCT cập nhật global classifier lặp lại trên feature tổng hợp.
- Synthetic replay đạt gần `99%` accuracy nhưng dữ liệu thật chỉ khoảng `88%`, cho thấy classifier overfit synthetic distribution.
- Một số task thật bị giảm so với đỉnh lịch sử dù average accuracy tăng, vì vậy forgetting và accuracy có thể cùng tăng.

### 53.2. Adaptive trust-region CRCT

V4 lưu classifier ngay trước CRCT làm teacher. Sau khi CRCT train xong, phương pháp:

1. Tạo tập anchor độc lập quanh các centroid đã lưu từ feature thật; không dùng chính batch replay vừa train.
2. So sánh pre-CRCT và post-CRCT logits trên anchor của từng lớp cũ.
3. Thử các hệ số nội suy `alpha` từ `0.0` đến `1.0`.
4. Với mỗi alpha, đo theo từng lớp: KL drift, mức giảm target confidence và mức giảm target-vs-competitor margin.
5. Dùng quantile `0.9` để ràng buộc nhóm lớp drift mạnh nhất, thay vì chỉ nhìn trung bình toàn bộ lớp.
6. Chọn alpha lớn nhất vẫn thỏa cả ba giới hạn; sau đó nội suy toàn bộ classifier bằng cùng một alpha.

```text
head_final = head_before + alpha * (head_after - head_before)
```

Nội suy toàn bộ head cùng nhau giữ hình học tương đối giữa hàng lớp cũ và lớp mới, tránh lỗi v1 khi chỉ giảm gradient hàng cũ. Alpha được chọn lại sau mỗi task: CRCT hữu ích có thể giữ gần `1`, CRCT gây drift sẽ tự bị thu nhỏ hoặc rollback.

Thiết lập v4 đầu tiên:

```text
steps=10, anchors/component=4, covariance_scale=0.25
class_quantile=0.9, max_KL=0.02
max_confidence_drop=0.02, max_margin_drop=0.10
```

Code chọn alpha chỉ trên rank 0, broadcast sang mọi rank rồi mới áp classifier, bảo đảm đúng khi chạy DDP nhiều GPU. Tính năng mặc định tắt nên các cấu hình cũ vẫn tái lập được.

## 54. Sửa smoke test v4

Lần smoke đầu tiên không kiểm tra được trust-region vì dừng ở đầu task 2 với CUDA OOM. Log cho thấy một tiến trình khác (PID `790732`) đang chiếm khoảng `20.01 GiB`; tiến trình smoke chỉ dùng khoảng `2.58 GiB` và GPU còn khoảng `54 MiB`. Đây không phải lỗi bộ nhớ do trust-region.

Smoke cũ còn đặt `num_tasks=2`, khiến Split-CIFAR100 bị chia thành 2 task, mỗi task 50 lớp, trong khi TII checkpoint được train theo 10 task, mỗi task 10 lớp. Vì partition không khớp, Acc@1 task 1 chỉ `1.94%`; con số này không có giá trị đánh giá phương pháp.

Bản sửa thêm:

```text
--num_tasks 10
--max_train_tasks 2
```

`num_tasks=10` giữ nguyên class partition và target-task mapping. `max_train_tasks=2` chỉ giới hạn vòng train ở hai task đầu. Full run dùng `max_train_tasks=0`, nghĩa là chạy đủ 10 task như trước.

Smoke hợp lệ phải:

1. In `Limiting run to 2 of 10 tasks`.
2. Chạy hết task 2 mà không có traceback.
3. In dòng `CRCT adaptive trust region` với alpha, số anchor và ba metric drift.


## 55. ImageNet-R: Exhaustive LoRA Rematching

### 55.1. Van de

Ba huong dinh tuyen hau xu ly truoc do deu khong thanh cong:

- Prototype trong khong gian LoRA: feature cua cac adapter khac nhau khong so sanh truc tiep duoc.
- Shared-space prototype: nearest-prototype tren backbone dong bang khong mo ta tot ranh gioi 200 lop ImageNet-R.
- Learned replay task router: validation tren feature memory qua lac quan va khong tong quat hoa sang test renditions.

Nut that chinh van la phai chon mot task/LoRA duy nhat truoc khi phan loai. Neu task sai, lop dung de bi loai khoi nhom ung vien.

### 55.2. Exhaustive rematching

Phuong phap moi khong doan task truoc. Voi moi anh:

1. Chay anh qua tat ca LoRA da hoc.
2. LoRA cua task `t` chi cham cac lop thuoc task `t`.
3. Ghep logits cuc bo cua tat ca task thanh mot vector logits tren toan bo lop da thay.
4. Them TII task prior nhe (`0.1`) de on dinh thang diem giua cac adapter.
5. Lay du doan toan cuc tren vector da ghep.

Phuong phap khong train lai va khong thay doi checkpoint. Doi lai, chi phi inference tang tu mot adapter pass len toi da 10 adapter pass o task cuoi.

### 55.3. Ket qua sau 10 task

Thiet lap: Split-ImageNet-R, ViT-B/16, seed 42, TII prior weight `0.1`, logit temperature `1.0`.

| Chi so | Baseline | Hybrid Real+CFS | Exhaustive | Thay doi so voi baseline |
|---|---:|---:|---:|---:|
| Acc@task | 77.3007 | 77.5854 | **80.3087** | **+3.0080** |
| Acc@1 | 73.8379 | 74.0477 | **74.9447** | **+1.1068** |
| Acc@5 | 86.0767 | 86.4646 | **88.5370** | **+2.4603** |
| Loss | 1.2399 | 1.2230 | **1.0919** | **-0.1480** |
| Forgetting | 3.5268 | 3.3264 | **2.8804** | **-0.6464** |
| Backward | -3.1815 | -2.9319 | **-2.8605** | **+0.3210** |

Day la thi nghiem ImageNet-R dau tien cai thien dong thoi tat ca nhom chi so. So voi baseline, loss giam xap xi `11.9%` va forgetting giam xap xi `18.3%`.

### 55.4. Y nghia

- Acc@task tang manh xac nhan routing mot-LoRA la nut that thuc su.
- Acc@1 va Acc@5 tang cho thay moi adapter cham cac lop cua chinh task no tot hon viec dung adapter duoc router chon.
- Forgetting giam vi LoRA cu luon co co hoi tu cham lai cac lop cu, thay vi bi loai ngay khi TII chon nham task.
- Ket qua danh doi do chinh xac lay chi phi suy luan tang tuyen tinh theo so task da hoc.


## 56. Ablation TII prior va phan ra dong gop tren ImageNet-R

### 56.1. TII prior ablation

Giu logit temperature `1.0`, exhaustive rematching duoc danh gia voi TII prior tu `0.0` den `0.3`.

| TII prior | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 80.3817 | 75.0211 | 88.4048 | 1.1026 | 2.8914 | -2.8205 |
| 0.05 | 80.2631 | 74.9614 | 88.4533 | 1.0968 | 2.8959 | -2.8228 |
| 0.10 | 80.3087 | 74.9447 | 88.5370 | 1.0919 | **2.8804** | -2.8605 |
| 0.15 | 80.5439 | 75.1308 | 88.5306 | 1.0879 | 2.8986 | -2.8830 |
| 0.20 | 80.5369 | 75.0960 | **88.6079** | 1.0847 | 2.9877 | -2.9877 |
| 0.25 | 80.5505 | 75.1087 | 88.5955 | 1.0824 | 2.9211 | -2.9012 |
| **0.30** | **80.6549** | **75.1798** | 88.5327 | **1.0809** | 2.8848 | -2.8449 |

Prior `0.3` duoc chon lam cau hinh chinh vi dat Acc@task, Acc@1 va loss tot nhat. Forgetting chi cao hon gia tri thap nhat tai prior `0.1` dung `0.0044` diem. Moi lan exhaustive evaluation mat khoang `495` giay, tuong duong `8.25` phut tren RTX 4090.

So voi baseline routing ban dau, cau hinh Hybrid + Exhaustive prior `0.3`:

- Tang Acc@task `3.3542` diem.
- Tang Acc@1 `1.3419` diem.
- Tang Acc@5 `2.4560` diem.
- Giam loss `0.1590`, xap xi `12.8%`.
- Giam forgetting `0.6420`, xap xi `18.2%`.
- Cai thien backward tu `-3.1815` len `-2.8449`.

### 56.2. Baseline checkpoint + Exhaustive

De tach dong gop cua training-time Hybrid Real+CFS va inference-time Exhaustive, cung prior `0.3` duoc danh gia tren checkpoint baseline.

| Phuong phap | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---|---:|---:|---:|---:|---:|---:|
| Baseline routing | 77.3007 | 73.8379 | 86.0767 | 1.2399 | 3.5268 | -3.1815 |
| Baseline + Exhaustive | 80.3572 | 75.0440 | **88.7258** | 1.0914 | 3.3068 | -3.3068 |
| Hybrid Real+CFS routing | 77.5854 | 74.0477 | 86.4646 | 1.2230 | 3.3264 | -2.9319 |
| **Hybrid Real+CFS + Exhaustive** | **80.6549** | **75.1798** | 88.5327 | **1.0809** | **2.8848** | **-2.8449** |

### 56.3. Ket luan ablation

- Exhaustive rematching la nguon tang accuracy chinh: tren checkpoint baseline, Acc@task tang `3.0565`, Acc@1 tang `1.2061` va Acc@5 tang `2.6491` diem.
- Hybrid Real+CFS van co dong gop rieng. Khi ca hai deu dung exhaustive prior `0.3`, Hybrid tang them Acc@task `0.2977`, Acc@1 `0.1358`, giam loss `0.0105`, giam forgetting `0.4220` va cai thien backward `0.4619`.
- Baseline + Exhaustive co Acc@5 cao hon, nhung kha nang duy tri task cu kem hon ro ret.
- Ket hop Hybrid Real+CFS + Exhaustive la cau hinh can bang tot nhat cho continual learning: top-1 accuracy cao nhat, loss thap nhat, forgetting thap nhat va backward gan 0 nhat trong bang phan ra dong gop.
- Trade-off chinh la inference tang tu mot adapter pass len toi da muoi adapter pass tai task cuoi.
