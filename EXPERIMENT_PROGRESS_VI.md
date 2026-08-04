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