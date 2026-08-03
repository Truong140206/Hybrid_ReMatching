# Ti?n d? thí nghi?m HRM-PET + CFS + Semantic

## 1. M?c tiêu

M?c tiêu hi?n t?i là c?i ti?n HRM-PET b?ng hai hu?ng l?y ı tu?ng t? paper PMI-CFS:

- CFS, vi?t t?t c?a Contrastive Feature Selection.
- Semantic-aware feature projection / semantic-aware relation distillation.

Trong HRM-PET, ph?n phù h?p nh?t d? áp d?ng CFS là CRCT feature replay. Ph?n phù h?p nh?t d? áp d?ng semantic là CTIRD, vì CTIRD dang distill quan h? gi?a các sample/class qua feature similarity.

## 2. Nh?ng gì dã làm

### 2.1. Thêm CFS vào CRCT

Ğã thêm CFS vào bu?c sinh feature gi? cho CRCT ? c? hai engine:

- TII / HidePrompt engine.
- LoRA HRM engine.

Logic m?i:

```text
feature th?t theo class
-> tính mean/covariance
-> train MLP contrastive nh? cho t?ng class
-> sample nhi?u Gaussian candidates
-> dua candidates qua MLP CFS
-> ch?n subset da d?ng hon
-> dùng subset dó cho CRCT
```

CFS có th? b?t/t?t b?ng:

```bash
--cfs_sampling
```

### 2.2. Thêm semantic-aware CTIRD b?n d?u

B?n d?u dùng tên class d? t?o semantic embedding b?ng hashing, sau dó nhân semantic similarity vào target relation c?a CTIRD.

Cách này ch?y du?c, nhung dua semantic vào khá m?nh, nh?t là khi `semantic_alpha=1.0`.

### 2.3. Thêm semantic top-k theo hu?ng g?n paper hon

Sau khi d?c paper, semantic du?c s?a theo hu?ng:

- ch? dùng top-5 class g?n nghia nh?t,
- gi?m `semantic_alpha` xu?ng `0.1`, gi?ng paper,
- thêm superclass CIFAR100 d? similarity có ı nghia hon.

Mode này là:

```bash
--semantic_mode topk_mix
--semantic_alpha 0.1
--semantic_top_k 5
```

### 2.4. C?i ti?n ti?p: semantic adaptive gate

K?t qu? top-k v?n chua t?t hon CFS-only. Vì v?y code ti?p t?c du?c s?a theo hu?ng an toàn hon: `adaptive_gate`.

İ tu?ng:

- Không t?o semantic target m?i thay th? CTIRD.
- Không ép model h?c theo semantic prior d?c l?p.
- Ch? tang nh? tr?ng s? c?a nh?ng quan h? mà CTIRD cu dã có, d?ng th?i hai class cung g?n nghia.

Công th?c ı tu?ng:

```text
gated_target = normalize(old_relation * (1 + alpha * semantic_weight_topk))
```

Mode m?i:

```bash
--semantic_mode adaptive_gate
--semantic_alpha 0.05
--semantic_top_k 5
```

Ğây là hu?ng ít phá k?t qu? CFS-only hon, vì semantic ch? dóng vai trò di?u ch?nh ph?.

## 3. K?t qu? dã ch?y

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

Nh?n xét: CFS-only là b?n t?t nh?t hi?n t?i.

### 3.3. CFS + semantic b?n d?u alpha 1.0

```text
Acc@task: 87.3000
Acc@1:    87.2100
Acc@5:    97.6800
Loss:     0.5519
Forgetting: 3.4222
Backward: -3.0222
```

Nh?n xét: t?t hon baseline m?t chút, nhung th?p hon CFS-only.

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

Nh?n xét: không c?i thi?n, g?n baseline và th?p hon CFS-only.

## 4. K?t lu?n t?m th?i

CFS dang là ph?n c?i thi?n chính. Vi?c ch?n feature replay t?t hon cho CRCT giúp classifier ?n d?nh hon và gi?m loss rõ ràng.

Semantic-aware CTIRD dã ch?y du?c nhung chua c?i thi?n trên CIFAR100. Kh? nang cao là vì semantic t? class name/superclass còn y?u so v?i CLIP text feature trong paper, và n?u dua semantic vào quá m?nh thì nó làm l?ch target relation c?a CTIRD.

Vì v?y hu?ng c?i ti?n ti?p theo là dùng semantic r?t nh?, nhu m?t gate h? tr? CTIRD thay vì t?o target m?i. Ğây là lı do thêm mode `adaptive_gate`.

## 5. L?nh ch?y ti?p nên th?

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

K? v?ng: n?u semantic có ích, mode này nên gi? du?c m?c g?n CFS-only và có th? c?i thi?n nh? loss/forgetting. N?u v?n th?p hon CFS-only, nên báo cáo r?ng semantic class-name chua d? m?nh và c?n CLIP text embedding th?t d? bám sát paper hon.
## 6. Semantic projection sát ı tu?ng paper hon

Sau khi th? `semantic_mode=topk_mix` và th?y k?t qu? không vu?t CFS-only, code du?c c?i ti?n thêm m?t hu?ng sát ı tu?ng g?c c?a paper hon: `--semantic_projection`.

İ tu?ng g?c trong paper:

```text
text feature class c -> text feature class d
-> t?o projection/rotation c sang d
-> apply projection dó lên image feature c?a class c
-> t?o pseudo feature cho class d
```

B?n áp d?ng vào HRM-PET hi?n t?i:

```text
semantic embedding class ngu?n c
semantic embedding class dích d
-> t?o phép xoay/reflection trong feature space
-> sample feature t? Gaussian/CFS c?a class c
-> xoay ph?n residual c?a feature c sang hu?ng semantic c?a class d
-> d?t quanh mean c?a class d
-> dùng pseudo feature này trong CRCT c?a class d
```

Công th?c ı tu?ng:

```text
x_c = sample_from_class_c
r_c = x_c - mean_c
R_cd = semantic_rotation(text_c -> text_d)
x_d_pseudo = mean_d + R_cd(r_c)
```

Trong code, phép `R_cd` du?c tri?n khai b?ng m?t phép bi?n d?i tr?c giao ki?u Householder d? không c?n t?o ma tr?n 768 x 768 d?y d?. Ğây là cách nh? hon nhung v?n gi? tinh th?n: dùng quan h? semantic gi?a class d? project feature t? class ngu?n sang class dích.

Tham s? m?i:

```bash
--semantic_projection
--semantic_projection_ratio 0.25
--semantic_projection_top_k 5
--semantic_projection_strength 1.0
```

İ nghia:

- `--semantic_projection`: b?t semantic feature projection trong CRCT.
- `--semantic_projection_ratio`: t? l? feature CRCT c?a m?i class du?c l?y t? semantic projection. Ví d? `0.25` nghia là 25% projected feature, 75% Gaussian/CFS feature g?c.
- `--semantic_projection_top_k`: s? class ngu?n g?n nghia nh?t dùng d? project sang class dích.
- `--semantic_projection_strength`: m?c áp d?ng phép projection. `1.0` là dùng projection d?y d?, th?p hon thì tr?n nh? hon.

Luu ı quan tr?ng:

- B?n này sát paper hon các b?n semantic CTIRD tru?c vì nó th?t s? có bu?c project feature t? class này sang class khác.
- Tuy nhiên v?n chua ph?i b?n paper g?c tuy?t d?i, vì paper dùng CLIP text feature th?t và model inversion sinh ?nh. B?n hi?n t?i dùng semantic embedding nh? t? class name/superclass và áp d?ng tr?c ti?p trong HRM feature space.
- Ğây là ablation dáng th? ti?p theo. N?u c?i thi?n, có th? trình bày là semantic-aware feature projection dã du?c chuy?n hóa thành semantic-projected CRCT replay trong HRM-PET.

## 7. K?t qu? th? semantic projection trên CIFAR100

L?nh ch?y trên máy Ubuntu/RTX 4090 dùng output:

```text
~/Documents/truongnguyen/hrm-pet-output/cifar100_lora_cfs_semantic_projection_seed42
```

Thi?t l?p chính:

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

So sánh nhanh:

| Phiên b?n | Acc@task | Acc@1 | Acc@5 | Loss | Nh?n xét |
|---|---:|---:|---:|---:|---|
| Baseline chua CFS | 87.0500 | 86.9000 | 97.5900 | 0.5860 | M?c g?c d? so |
| CFS-only | 88.0600 | 88.0000 | 97.9400 | 0.5232 | T?t nh?t hi?n t?i |
| CFS + semantic top-k alpha 0.1 | 87.1700 | 86.8900 | 97.5700 | 0.5559 | Semantic top-k chua c?i thi?n rõ |
| CFS + semantic projection | 87.5600 | 87.0100 | 97.5900 | 0.5508 | T?t hon semantic top-k và baseline nh?, nhung chua vu?t CFS-only |

K?t lu?n t?m th?i:

- Semantic projection sát ı tu?ng g?c c?a paper hon semantic top-k vì có bu?c project feature t? class ngu?n sang class dích.
- K?t qu? dã nhích lên so v?i semantic top-k và loss t?t hon baseline.
- Tuy nhiên CFS-only v?n là c?u hình m?nh nh?t trong các l?n th? hi?n t?i.
- Nguyên nhân h?p lı: semantic embedding hi?n t?i v?n là class-name/superclass embedding nh?, chua ph?i CLIP text embedding th?t nhu paper, nên tín hi?u semantic chua d? m?nh d? vu?t CFS-only.
- Hu?ng ti?p theo n?u mu?n bám paper hon n?a: thay semantic embedding hi?n t?i b?ng CLIP text embedding th?t c?a class name, r?i dùng embedding dó cho c? `semantic_distill` và `semantic_projection`.

## 8. B?n paper-style: áp d?ng sát paper PMI-CFS hon

Sau khi th? semantic projection ki?u nh?, code du?c b? sung thêm m?t ch? d? m?i d? bám sát paper hon.

Các di?m thay d?i chính:

1. Semantic embedding chuy?n sang CLIP text embedding th?t khi b?t:

```bash
--semantic_backend clip
```

Thay vì dùng vector hash t? tên class, code s? dùng `open_clip_torch` ho?c `clip` d? encode prompt d?ng:

```text
a photo of a {class name}.
```

Ği?u này sát paper hon vì paper do semantic similarity b?ng cosine similarity gi?a CLIP text features.

2. Semantic-aware feature projection có mode m?i:

```bash
--semantic_projection_mode paper
--semantic_projection_alpha 0.1
```

Mode này mô ph?ng Eq. 7-9 trong paper:

```text
Ft(td) = R_c,d Ft(tc)
o_L,d = R_c,d o_L,c
o'_L,d = normalize((1 - alpha) o_L,d + alpha Ft(td))
```

Trong code hi?n t?i, `R_c,d` du?c tri?n khai nhu m?t phép xoay tr?c giao t?i thi?u trong m?t ph?ng t?o b?i text feature class ngu?n và text feature class dích. Phép xoay này map hu?ng semantic c?a class ngu?n sang class dích.

3. CFS có mode paper-style:

```bash
--cfs_paper_style
--cfs_selection_ratio 0.5
--cfs_selection_steps 5
--cfs_epochs 200
--cfs_lr 0.01
--cfs_hidden_dim 512
```

Mode này g?n Algorithm 2 hon: kh?i t?o m?t t?p feature dã ch?n, sau dó nhi?u bu?c sample candidate t? Gaussian và ch?n candidate có cosine similarity trung bình th?p nh?t v?i t?p dã ch?n trong không gian CFS MLP.

Luu ı r?t quan tr?ng:

- Paper g?c dùng CLIP image encoder + CLIP text encoder, feature image và feature text cùng n?m trên unit hypersphere.
- HRM-PET hi?n t?i dùng ViT/timm feature, không ph?i pipeline CLIP inversion d?y d?.
- Vì v?y b?n này là b?n áp d?ng sát công th?c semantic/CFS c?a paper vào CRCT feature replay c?a HRM-PET, chua ph?i bê nguyên toàn b? PMI + full-model inversion sinh ?nh t? paper.
- N?u mu?n y h?t tuy?t d?i paper, ph?i tích h?p thêm c? pipeline model inversion PMI/full-model inversion d? sinh ?nh synthetic, vi?c này l?n hon nhi?u so v?i s?a CRCT feature replay.

B?n dáng ch?y ti?p theo trên CIFAR100:

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

- N?u CLIP text feature giúp ch?n class liên quan t?t hon hash/superclass, k?t qu? có th? t?t hon semantic projection cu.
- N?u Acc v?n th?p hon CFS-only, có th? k?t lu?n r?ng ph?n semantic c?a paper c?n dúng CLIP image feature/model inversion m?i phát huy d?y d?, còn trong HRM feature space thì CFS replay v?n là thành ph?n có l?i nh?t.

## 9. K?t qu? ch?y paper-style CLIP semantic projection

L?nh ch?y dùng c?u hình sát paper hon:

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

So sánh v?i các m?c tru?c:

| Phiên b?n | Acc@task | Acc@1 | Acc@5 | Loss | Nh?n xét |
|---|---:|---:|---:|---:|---|
| Baseline chua CFS | 87.0500 | 86.9000 | 97.5900 | 0.5860 | M?c g?c |
| CFS-only | 88.0600 | 88.0000 | 97.9400 | 0.5232 | T?t nh?t hi?n t?i |
| CFS + semantic projection cu | 87.5600 | 87.0100 | 97.5900 | 0.5508 | T?t hon paper-style |
| CFS + paper-style CLIP semantic | 87.1600 | 86.5500 | 97.3300 | 0.6142 | Ch?y dúng nhung chua c?i thi?n |

K?t lu?n:

- B?n paper-style dã ch?y du?c end-to-end, không còn l?i runtime.
- Tuy nhiên k?t qu? không t?t hon CFS-only, và cung th?p hon semantic projection cu.
- Nguyên nhân h?p lı: công th?c paper gi? d?nh CLIP image feature và CLIP text feature n?m chung không gian semantic dã normalize. HRM-PET hi?n t?i dùng ViT/timm feature trong CRCT, không ph?i CLIP image feature dúng nghia, nên vi?c xoay feature theo CLIP text có th? làm l?ch feature kh?i không gian mà classifier HRM dang dùng.
- `semantic_projection_ratio = 0.5` có th? quá m?nh: 50% feature CRCT b? thay b?ng projected feature, trong khi feature space chua th?t s? align v?i CLIP.
- Hu?ng th? ti?p h?p lı hon: gi? `semantic_backend=clip` nhung gi?m projection ratio v? `0.1` ho?c `0.25`, ho?c b?t CLIP semantic ch? cho ch?n top-k class liên quan, còn projection gi? ki?u mean_shift cu.

## 10. Hu?ng c?i ti?n th?c d?ng: semantic-safe CRCT

Sau khi b?n paper-style ch?y xong nhung không c?i thi?n, hu?ng ti?p theo không c? bê nguyên paper n?a. Lı do là công th?c paper gi? d?nh CLIP image feature và CLIP text feature n?m chung không gian, còn HRM-PET hi?n t?i dùng feature ViT/timm trong CRCT. Vì v?y n?u xoay feature quá m?nh theo CLIP text, feature có th? r?i kh?i phân ph?i mà classifier HRM dang h?c.

İ tu?ng m?i: semantic ch? dùng d? g?i ı class ngu?n liên quan, còn feature cu?i cùng ph?i du?c ki?m tra b?ng th?ng kê feature HRM.

Pipeline m?i:

```text
1. Dùng semantic embedding d? ch?n top-k class ngu?n g?n class dích.
2. Sample feature t? Gaussian/CFS c?a các class ngu?n dó.
3. Project feature ngu?n sang class dích b?ng mode mean_shift nh?.
4. Sinh nhi?u candidate hon s? c?n dùng.
5. L?c candidate b?ng kho?ng cách t?i phân ph?i feature th?t c?a class dích.
6. Ch? dua các projected feature g?n phân ph?i class dích nh?t vào CRCT.
```

Các tham s? m?i:

```bash
--semantic_projection_filter
--semantic_projection_filter_multiplier 3
--semantic_projection_filter_cosine_weight 0.1
```

İ nghia:

- `--semantic_projection_filter`: b?t l?c feature projected theo phân ph?i class dích.
- `--semantic_projection_filter_multiplier`: sinh nhi?u candidate hon r?i ch?n l?i. Ví d? `3` nghia là c?n 100 feature thì sinh 300 candidate r?i l?c l?y 100 t?t nh?t.
- `--semantic_projection_filter_cosine_weight`: thêm m?t ph?n nh? cosine distance t?i mean class dích khi x?p h?ng candidate.

Khác v?i paper-style:

- Không normalize feature v? unit vector nhu Eq. 9 n?a, vì HRM classifier không nh?t thi?t ho?t d?ng trong cùng CLIP unit hypersphere.
- Không dùng semantic d? ép toàn b? feature space.
- Semantic ch? là prior d? ch?n ngu?n và t?o candidate; phân ph?i feature th?t c?a HRM m?i là b? l?c cu?i.

C?u hình nên th? d?u tiên:

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

Luu ı: c?u hình này c? ı không b?t `--semantic_distill` ban d?u, vì các l?n tru?c semantic CTIRD làm k?t qu? gi?m. Tru?c m?t ch? thêm semantic vào CRCT replay m?t cách có ki?m soát. N?u k?t qu? t?t hon CFS-only, sau dó m?i th? b?t semantic CTIRD r?t nh?.

## 11. Káº¿t quáº£ semantic-safe CRCT trÃªn CIFAR100

Cáº¥u hÃ¬nh Ä‘Ã£ cháº¡y:

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

Káº¿t quáº£ cuá»‘i sau task 10:

```text
[Average accuracy till task10]
Acc@task 87.3600
Acc@1    86.9900
Acc@5    97.7400
Loss     0.5526
Forgetting 3.7333
Backward  -3.3222
```

So sÃ¡nh vá»›i cÃ¡c má»‘c chÃ­nh:

| PhiÃªn báº£n | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Nháº­n xÃ©t |
|---|---:|---:|---:|---:|---:|---|
| Baseline chÆ°a CFS | 87.0500 | 86.9000 | 97.5900 | 0.5860 | - | Má»‘c gá»‘c |
| CFS-only | 88.0600 | 88.0000 | 97.9400 | 0.5232 | 3.8889 | Tá»‘t nháº¥t hiá»‡n táº¡i |
| CFS + semantic projection cÅ© | 87.5600 | 87.0100 | 97.5900 | 0.5508 | 3.8111 | Tá»‘t hÆ¡n baseline, nhÆ°ng chÆ°a vÆ°á»£t CFS-only |
| CFS + paper-style CLIP semantic | 87.1600 | 86.5500 | 97.3300 | 0.6142 | 3.5222 | KhÃ´ng cáº£i thiá»‡n |
| CFS + semantic-safe CRCT | 87.3600 | 86.9900 | 97.7400 | 0.5526 | 3.7333 | Cáº£i thiá»‡n so vá»›i paper-style, nhÆ°ng váº«n chÆ°a vÆ°á»£t CFS-only |

Nháº­n xÃ©t:

- Semantic-safe CRCT Ä‘Ã£ sá»­a Ä‘Æ°á»£c váº¥n Ä‘á» lá»›n cá»§a paper-style: khÃ´ng cÃ²n Ã©p feature quÃ¡ máº¡nh theo CLIP text space.
- So vá»›i paper-style, káº¿t quáº£ tá»‘t hÆ¡n rÃµ á»Ÿ Acc@task, Acc@1, Acc@5 vÃ  loss.
- Tuy nhiÃªn so vá»›i CFS-only, káº¿t quáº£ váº«n tháº¥p hÆ¡n khoáº£ng `0.70` Acc@task vÃ  `1.01` Acc@1.
- Äiá»u nÃ y cho tháº¥y pháº§n cáº£i thiá»‡n cháº¯c cháº¯n nháº¥t hiá»‡n táº¡i váº«n lÃ  CFS replay. Semantic hiá»‡n táº¡i cÃ³ thá»ƒ giÃºp lÃ m giÃ u replay, nhÆ°ng tÃ­n hiá»‡u semantic chÆ°a Ä‘á»§ chuáº©n Ä‘á»ƒ vÆ°á»£t replay thá»‘ng kÃª thuáº§n.
- HÆ°á»›ng nÃªn thá»­ tiáº¿p theo lÃ  giáº£m áº£nh hÆ°á»Ÿng semantic hÆ¡n ná»¯a, vÃ­ dá»¥ `semantic_projection_ratio=0.05`, hoáº·c chá»‰ dÃ¹ng semantic Ä‘á»ƒ chá»n class nguá»“n nhÆ°ng khÃ´ng project máº¡nh. Má»™t hÆ°á»›ng khÃ¡c lÃ  cáº£i thiá»‡n CFS-only trÆ°á»›c, vÃ¬ Ä‘Ã¢y Ä‘ang lÃ  ná»n tá»‘t nháº¥t.

## 12. Cáº£i tiáº¿n tiáº¿p theo: CFS distribution filter

Sau khi cÃ¡c biáº¿n thá»ƒ semantic chÆ°a vÆ°á»£t Ä‘Æ°á»£c CFS-only, hÆ°á»›ng cáº£i tiáº¿n Ä‘Æ°á»£c chuyá»ƒn vá» pháº§n Ä‘ang cÃ³ hiá»‡u quáº£ nháº¥t: CFS replay.

Váº¥n Ä‘á» cá»§a CFS-only hiá»‡n táº¡i:

- CFS chá»n synthetic feature sao cho cÃ¡c Ä‘iá»ƒm Ä‘Æ°á»£c chá»n Ä‘a dáº¡ng trong embedding CFS.
- Tuy nhiÃªn candidate ban Ä‘áº§u váº«n Ä‘Æ°á»£c sample tá»« Gaussian/covariance cá»§a class.
- Náº¿u Gaussian sinh ra outlier, CFS cÃ³ thá»ƒ chá»n outlier Ä‘Ã³ vÃ¬ nÃ³ khÃ¡c cÃ¡c Ä‘iá»ƒm cÃ²n láº¡i, dÃ¹ outlier nÃ y khÃ´ng tháº­t sá»± náº±m gáº§n phÃ¢n phá»‘i feature cá»§a class.
- Outlier trong CRCT cÃ³ thá»ƒ lÃ m classifier bá»‹ kÃ©o lá»‡ch, nháº¥t lÃ  á»Ÿ cÃ¡c task sau khi sá»‘ class tÄƒng lÃªn.

Ã tÆ°á»Ÿng má»›i:

```text
1. Sinh nhiá»u Gaussian candidate hÆ¡n bÃ¬nh thÆ°á»ng.
2. TÃ­nh Ä‘iá»ƒm gáº§n phÃ¢n phá»‘i class báº±ng diagonal Mahalanobis distance tá»›i mean/cov cá»§a class.
3. CÃ³ thá»ƒ cá»™ng thÃªm cosine distance nhá» tá»›i class mean.
4. Giá»¯ láº¡i cÃ¡c candidate sáº¡ch nháº¥t.
5. Cháº¡y CFS diversity selection trÃªn candidate pool Ä‘Ã£ lá»c.
```

Tham sá»‘ má»›i:

```bash
--cfs_distribution_filter
--cfs_filter_multiplier 3
--cfs_filter_cosine_weight 0.0
```

Ã nghÄ©a:

- `--cfs_distribution_filter`: báº­t lá»c candidate trÆ°á»›c khi CFS chá»n diversity.
- `--cfs_filter_multiplier`: sá»‘ candidate thÃ´ sinh thÃªm trÆ°á»›c khi lá»c. VÃ­ dá»¥ CFS cáº§n 360 candidate, multiplier 3 sáº½ sinh 1080 candidate rá»“i lá»c cÃ²n 360.
- `--cfs_filter_cosine_weight`: trá»ng sá»‘ cosine distance tá»›i class mean. Máº·c Ä‘á»‹nh `0.0` Ä‘á»ƒ Æ°u tiÃªn Mahalanobis thuáº§n; cÃ³ thá»ƒ thá»­ `0.05` hoáº·c `0.1` náº¿u muá»‘n thÃªm rÃ ng buá»™c hÆ°á»›ng feature.

Äiá»ƒm khÃ¡c so vá»›i semantic-safe:

- KhÃ´ng dÃ¹ng semantic, khÃ´ng dÃ¹ng CLIP, khÃ´ng project feature.
- Chá»‰ lÃ m sáº¡ch candidate Gaussian trÆ°á»›c khi Ä‘Æ°a vÃ o CFS.
- VÃ¬ CFS-only Ä‘ang lÃ  báº£n tá»‘t nháº¥t, hÆ°á»›ng nÃ y Ã­t rá»§i ro hÆ¡n semantic vÃ  bÃ¡m trá»±c tiáº¿p vÃ o thÃ nh pháº§n Ä‘ang cÃ³ lá»£i.

Cáº¥u hÃ¬nh nÃªn cháº¡y thá»­ Ä‘áº§u tiÃªn:

```bash
--cfs_sampling
--cfs_epochs 50
--cfs_train_max_samples 1024
--cfs_candidate_multiplier 3
--cfs_distribution_filter
--cfs_filter_multiplier 3
--cfs_filter_cosine_weight 0.0
```

Náº¿u káº¿t quáº£ tá»‘t hÆ¡n CFS-only, cÃ³ thá»ƒ thá»­ tiáº¿p:

```bash
--cfs_filter_cosine_weight 0.05
```

hoáº·c tÄƒng nháº¹ candidate pool:

```bash
--cfs_candidate_multiplier 4
--cfs_filter_multiplier 3
```
## 13. Cáº£i tiáº¿n semantic má»›i: semantic feature adapter

CÃ¡c láº§n semantic trÆ°á»›c chÆ°a vÆ°á»£t CFS-only vÃ¬ semantic Ä‘Æ°á»£c dÃ¹ng trá»±c tiáº¿p tá»« text/CLIP. CÃ¡ch nÃ y cÃ³ má»™t lá»‡ch pha quan trá»ng: CLIP text embedding náº±m trong khÃ´ng gian ngá»¯ nghÄ©a cá»§a CLIP, cÃ²n HRM-PET dÃ¹ng feature `pre_logits` cá»§a ViT/timm Ä‘á»ƒ CRCT. VÃ¬ váº­y semantic cÃ³ thá»ƒ Ä‘Ãºng vá» nghÄ©a ngÃ´n ngá»¯ nhÆ°ng váº«n khÃ´ng khá»›p hÃ¬nh há»c feature mÃ  classifier Ä‘ang há»c.

HÆ°á»›ng má»›i lÃ  `semantic feature adapter`.

Ã tÆ°á»Ÿng:

```text
1. Láº¥y semantic embedding cá»§a toÃ n bá»™ class, vÃ­ dá»¥ CLIP text embedding.
2. Sau má»—i task, láº¥y mean feature tháº­t cá»§a cÃ¡c class Ä‘Ã£ tháº¥y trong HRM.
3. Há»c má»™t phÃ©p chiáº¿u ridge regression tá»« semantic embedding sang HRM class mean.
4. DÃ¹ng semantic embedding Ä‘Ã£ align nÃ y Ä‘á»ƒ chá»n source class liÃªn quan vÃ  semantic projection.
5. Váº«n giá»¯ semantic projection ratio nhá» vÃ  filter theo target distribution Ä‘á»ƒ trÃ¡nh phÃ¡ CFS replay.
```

KhÃ¡c vá»›i paper-style CLIP semantic:

- Paper-style dÃ¹ng CLIP text vector trá»±c tiáº¿p Ä‘á»ƒ xoay/project feature.
- Báº£n adapter há»c cÃ¡ch dá»‹ch CLIP/text semantic sang feature space cá»§a HRM trÆ°á»›c.
- VÃ¬ váº­y semantic khÃ´ng cÃ²n Ã©p replay Ä‘i theo khÃ´ng gian CLIP thuáº§n, mÃ  bÃ¡m vÃ o thá»‘ng kÃª class tháº­t Ä‘Ã£ há»c.

Tham sá»‘ má»›i:

```bash
--semantic_feature_adapter
--semantic_adapter_dim 512
--semantic_adapter_ridge 0.01
--semantic_adapter_blend 1.0
--semantic_adapter_min_classes 5
```

Ã nghÄ©a:

- `--semantic_feature_adapter`: báº­t adapter semantic -> HRM feature mean.
- `--semantic_adapter_dim`: sá»‘ chiá»u semantic embedding Ä‘áº§u vÃ o cho adapter.
- `--semantic_adapter_ridge`: há»‡ sá»‘ regularization khi há»c phÃ©p chiáº¿u; giÃºp trÃ¡nh overfit khi sá»‘ class Ä‘Ã£ tháº¥y cÃ²n Ã­t.
- `--semantic_adapter_blend`: tá»‰ lá»‡ dÃ¹ng embedding Ä‘Ã£ align. `1.0` nghÄ©a lÃ  dÃ¹ng hoÃ n toÃ n embedding Ä‘Ã£ align.
- `--semantic_adapter_min_classes`: cáº§n Ã­t nháº¥t bao nhiÃªu class Ä‘Ã£ tháº¥y má»›i báº­t adapter.

Cáº¥u hÃ¬nh nÃªn thá»­:

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

Ká»³ vá»ng:

- Náº¿u semantic tháº­t sá»± giÃºp, báº£n nÃ y cÃ³ cÆ¡ há»™i tá»‘t hÆ¡n cÃ¡c báº£n semantic cÅ© vÃ¬ semantic Ä‘Ã£ Ä‘Æ°á»£c align vá»›i HRM feature space.
- Váº«n chÆ°a thá»ƒ Ä‘áº£m báº£o vÆ°á»£t CFS-only trÆ°á»›c khi cháº¡y, nhÆ°ng Ä‘Ã¢y lÃ  hÆ°á»›ng semantic há»£p lÃ½ hÆ¡n so vá»›i dÃ¹ng CLIP/text trá»±c tiáº¿p.
- NÃªn cháº¡y vá»›i ratio nhá» `0.05` trÆ°á»›c Ä‘á»ƒ trÃ¡nh phÃ¡ baseline CFS-only.
## 14. Káº¿t quáº£ semantic feature adapter vÃ  hÆ°á»›ng covariance transfer

Káº¿t quáº£ cháº¡y `semantic feature adapter` trÃªn CIFAR100:

```text
[Average accuracy till task10]
Acc@task 87.4200
Acc@1    87.1500
Acc@5    97.8200
Loss     0.5464
Forgetting 3.6556
Backward  -3.1000
```

So vá»›i semantic-safe trÆ°á»›c Ä‘Ã³:

| PhiÃªn báº£n | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---|---:|---:|---:|---:|---:|---:|
| Semantic-safe CRCT | 87.3600 | 86.9900 | 97.7400 | 0.5526 | 3.7333 | -3.3222 |
| Semantic feature adapter | 87.4200 | 87.1500 | 97.8200 | 0.5464 | 3.6556 | -3.1000 |

Nháº­n xÃ©t:

- Adapter cÃ³ cáº£i thiá»‡n tháº­t so vá»›i semantic-safe: Acc@1 tÄƒng `0.16`, loss giáº£m `0.0062`.
- Äiá»u nÃ y cho tháº¥y hÆ°á»›ng align semantic sang HRM feature space há»£p lÃ½ hÆ¡n dÃ¹ng CLIP/text trá»±c tiáº¿p.
- Tuy nhiÃªn váº«n chÆ°a vÆ°á»£t CFS-only, nÃªn semantic váº«n cáº§n Ä‘Æ°á»£c Ä‘Æ°a vÃ o nháº¹ vÃ  cÃ³ kiá»ƒm soÃ¡t hÆ¡n.

Cáº£i tiáº¿n tiáº¿p theo: `semantic_projection_mode=covariance_transfer`.

Ã tÆ°á»Ÿng:

```text
1. Semantic adapter chá»‰ dÃ¹ng Ä‘á»ƒ chá»n source class gáº§n target class.
2. Sample feature tá»« source class.
3. Láº¥y residual: source_feature - source_mean.
4. Scale residual theo tá»‰ lá»‡ variance target/source.
5. Äáº·t residual Ä‘Ã£ scale quanh target_mean.
6. Lá»c láº¡i báº±ng target distribution.
```

KhÃ¡c vá»›i `mean_shift`:

- `mean_shift` váº«n xoay residual theo hÆ°á»›ng semantic vector.
- `covariance_transfer` khÃ´ng xoay theo semantic ná»¯a, chá»‰ mÆ°á»£n hÃ¬nh dáº¡ng biáº¿n thiÃªn tá»« source class rá»“i neo vÃ o target distribution.
- VÃ¬ váº­y semantic Ä‘Ã³ng vai trÃ² chá»n hÃ ng xÃ³m class, cÃ²n hÃ¬nh há»c feature váº«n do mean/cov tháº­t cá»§a HRM quyáº¿t Ä‘á»‹nh.

Tham sá»‘ má»›i:

```bash
--semantic_projection_mode covariance_transfer
--semantic_cov_transfer_min_scale 0.5
--semantic_cov_transfer_max_scale 2.0
```

Cáº¥u hÃ¬nh nÃªn thá»­:

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

Ká»³ vá»ng:

- Ãt phÃ¡ CFS-only hÆ¡n vÃ¬ projected features bÃ¡m target mean/cov cháº·t hÆ¡n.
- Náº¿u semantic há»¯u Ã­ch, báº£n nÃ y cÃ³ kháº£ nÄƒng cáº£i thiá»‡n hÆ¡n adapter `mean_shift` do trÃ¡nh xoay feature theo semantic vector.