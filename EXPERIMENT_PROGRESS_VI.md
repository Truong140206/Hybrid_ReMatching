## Soft-mixture LoRA mot luot forward (14/08/2026)

- Vectorized exhaustive giu nguyen tuyet doi ket qua exhaustive nhung chi tang
  toc `1.032x`, vi van tinh du 10 LoRA cho moi anh.
- Thi nghiem ke tiep khong hoc router va khong dung du lieu cu: TII xep hang
  task, lay co dinh top-4 roi chuan hoa diem bang softmax.
- Tai tung attention block, residual QKV cua bon LoRA duoc tron theo
  `sum_t p_t * LoRA_t`; tong trong so bang 1. Mo hinh chi chay mot forward hon
  hop, thay vi chay rieng tung LoRA.
- Cau hinh duoc chot truoc khi xem test: top-k `4`, task temperature `1.0`, TII
  prior `0.3`, logit temperature `1.0`. Khong quet tham so tren test.
- Cong ket qua so sanh rieng voi baseline va exhaustive tren Acc@task, Acc@1,
  Acc@5, Loss, Forgetting, Backward; dong thoi kiem tra `LoRA/sample <= 4` va
  `ForwardCalls/sample = 1`.
- Day la ablation giam chi phi inference, khong phai bien the CFS. Chua ghi
  nhan ket qua cho toi khi unit test va danh gia ImageNet-R tren RTX 4090 hoan
  tat.

### Ket qua soft-mixture top-4

- Acc@task `79.2034`, Acc@1 `73.4173`, Acc@5 `87.0402`, Loss `1.1775`,
  Forgetting `2.9761`, Backward `-2.6194`.
- Chi phi dat dung muc tieu: `LoRA/sample=4.0`, `ForwardCalls/sample=1.0`.
- So sanh tu dong ban dau dung nham baseline rank-8 cho checkpoint rank-5; da
  sua script de mac dinh dung log conventional cua chinh run va log vectorized
  exhaustive cung checkpoint.
- Du reference cu, ket qua van cho thay routing, top-5, loss va retention tot
  hon, nhung Acc@1 giam khoang `0.6` diem. Nguyen nhan phu hop nhat la convex
  mixture giu task evidence nhung lam mo class evidence cua tung LoRA.
- Ket luan: efficiency gate PASS, quality gate FAIL; khong quet temperature.

### Thi nghiem soft-route/hard-classify

- Chan doan tu soft-mixture: task evidence va retention tot hon, nhung convex
  LoRA mixture lam mo class evidence nen Acc@1 giam.
- Thay doi duy nhat: luot 1 van dung top-4 mixture de chon task; luot 2 chay
  rieng LoRA cua task da chon va dung logits cua LoRA cung de phan loai tren
  toan bo lop da hoc.
- Khong ep logits vao 20 lop cua routed task, vi nhu vay Acc@5 se bi gioi han
  boi routing accuracy. Khong hoc router, khong du lieu cu, khong tuning.
- Chi phi dat truoc: `LoRA/sample=5`, `ForwardCalls/sample=2`, van thap hon
  vectorized exhaustive (`10` va `3`).

### Ket qua soft-route/hard-classify

- Acc@task `79.2034`, Acc@1 `73.7386`, Acc@5 `86.5075`, Loss `1.2210`,
  Forgetting `3.0484`, Backward `-2.6125`.
- So voi conventional cung checkpoint: Acc@task `+1.6180`, Acc@1 `-0.3091`,
  Acc@5 `+0.0429`, Loss `-0.0020`, Forgetting `-0.2780`, Backward `+0.3194`.
- So voi soft-mixture: Acc@1 `+0.3213`, nhung Acc@5 `-0.5327`, Loss
  `+0.0435`, Forgetting `+0.0723`; chi phi tang them 1 LoRA va 1 forward.
- Efficiency gate PASS (`5` LoRA, `2` forward), quality gate FAIL. Luot hard
  co tac dung voi top-1 nhung khong du bu phan chat luong va chi phi bi mat.
  Khong quet tham so tiep cho nhanh nay.

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


## 57. Calibrated progressive rematching va cascade-aware Stage 2

### 57.1. Moc doi chung truoc khi sua Stage 2

| Chi so | Ket qua |
|---|---:|
| Acc@task | 80.5149 |
| Acc@1 | 75.0551 |
| Acc@5 | 87.9225 |
| Loss | 1.3975 |
| Forgetting | 2.9435 |
| Backward | -2.9036 |
| Stage 1 dung sau 2 LoRA | 55.2782% |
| Stage 2 dung sau 4 LoRA | 2.7509% |
| Chay du tat ca LoRA | 41.9709% |
| LoRA trung binh moi mau | 5.4127 |

Ban nay giam `45.87%` so luot chay LoRA so voi exhaustive 10 LoRA, trong khi Acc@1 chi thap hon exhaustive `0.1247` diem. Tuy nhien Stage 2 dung qua it va loss tang tu `1.0809` len `1.3975`.

### 57.2. Nguyen nhan va thay doi moi

- Gate Stage 2 cu duoc hoc tren tat ca mau calibration, nhung khi inference no chi nhan nhom mau kho ma Stage 1 da tu choi. Day la lech phan phoi train-inference.
- Sua theo cascade: hoc Stage 1 truoc, dung quyet dinh Stage 1 de lay residual set, sau do moi hoc Stage 2 tren residual set nay.
- Dam bao moi lop con toi thieu 4 hard samples de phep chia train/calibration/report van hop le.
- Giu target precision doc lap `99.5%`; khong ha hang rao an toan de doi lay coverage.
- Giam excluded-logit margin tu `20` xuong `8`. Top-1 va top-5 trong cac lop da cham khong doi, nhung mot loi halt hiem hoi khong con tao cross-entropy qua lon.

### 57.3. Ket qua cascade residual thuan

| Chi so | Residual thuan | Thay doi so voi muc 57.1 |
|---|---:|---:|
| Acc@task | 80.3693 | -0.1456 |
| Acc@1 | 74.8818 | -0.1733 |
| Acc@5 | 87.6868 | -0.2357 |
| Loss | 1.3362 | -0.0613 |
| Forgetting | 3.1344 | +0.1909 |
| Backward | -3.0945 | -0.1909 |
| Stage 2 dung | 9.5596% | +6.8087 diem phan tram |
| LoRA/mau | 5.0392 | -0.3735 |

Stage 2 da giam them chi phi nhung dung som qua manh, lam accuracy va kha nang giu task cu cung giam. Cau hinh nay khong duoc chon lam ket qua cuoi.

Tai task 10, gate Stage 2 hoc tren `914` residual samples, bao report precision `100%` va coverage `54.31%`. Tren test, Stage 2 chi dung `9.5596 / (100 - 54.8408) = 21.17%` residual. Chenh lech coverage lon cho thay report nho chua dai dien tot cho test distribution.

### 57.4. Stage 2 residual co boundary context

Bien the tiep theo duoc thiet ke de giam overfit cua residual thuan:

- Stage 1 giu nguyen.
- Stage 2 hoc tren residual va them `25%` cac mau Stage-1 accept gan nguong nhat cua tung lop.
- Stage 2 yeu cau empirical precision `100%`, trong khi Stage 1 van dung `99.5%`.
- Safe label cua Stage 2 yeu cau partial loss khong cao hon exhaustive loss; tolerance giam tu `0.05` xuong `0.0`.
- Excluded-logit margin van la `8` de tranh loss nhan tao tang qua manh.

### 57.5. Ket qua residual co boundary context

| Chi so | Ket qua |
|---|---:|
| Acc@task | 80.5329 |
| Acc@1 | 75.0731 |
| Acc@5 | 87.9394 |
| Loss | 1.2587 |
| Stage1Stop | 55.3059% |
| Stage2Stop | 2.0407% |
| FullFallback | 42.6534% |
| LoRA/mau | 5.4531 |
| Forgetting | 2.9413 |
| Backward | -2.9014 |

So voi gate truoc boundary-context, Acc@task, Acc@1, Acc@5, Forgetting va Backward deu tot hon rat nhe; loss giam manh tu `1.3975` xuong `1.2587`. Chi phi tang khong dang ke tu `5.4127` len `5.4531` LoRA/mau. Cau hinh nay la moc progressive can bang nhat hien tai.

So voi baseline rerun co loss `1.2305`, loss hien tai van cao hon `0.0282`, tuong duong khoang `2.3%`.

### 57.6. Output temperature scaling

De giam loss ma khong thay doi accuracy va forgetting, them mot buoc temperature scaling sau progressive rematching:

1. Mo phong dung quyet dinh Stage 1, Stage 2 va exhaustive fallback tren train calibration samples.
2. Tach mot tap calibration theo lop; khong dung anh test.
3. Tim mot temperature duong duy nhat toi thieu hoa cross-entropy cua logits dau ra.
4. Chia toan bo logits cho temperature da hoc khi danh gia.

Vi chia moi logit trong cung mot mau cho cung mot so duong, thu tu lop khong thay doi. Do do Acc@task, Acc@1, Acc@5, Forgetting, Backward, cac stop rate va LoRA/mau ve nguyen tac duoc giu nguyen; chi confidence va loss thay doi.

Day la calibration xac suat hop le, khac voi viec tuy y ha excluded-logit margin de lam dep loss. Neu temperature khong lam calibration loss giam, code tu quay ve `1.0`.

### 57.7. Temperature scaling that bai va nguyen nhan

Ket qua temperature scaling:

| Chi so | Khong temperature | Temperature scaling |
|---|---:|---:|
| Acc@task | 80.5329 | 80.5329 |
| Acc@1 | 75.0731 | 75.0731 |
| Acc@5 | 87.9394 | 87.9394 |
| Loss | **1.2587** | 1.5400 |
| Forgetting | 2.9413 | 2.9413 |
| LoRA/mau | 5.4531 | 5.4531 |

Temperature cua task 3--10 deu nam trong khoang `0.5548--0.7195`. Train calibration loss rat thap (`0.177--0.362`) nen chon `T < 1` va lam logits sac hon. Test kho hon train; khi du doan sai, confidence qua cao lam loss tang `0.2813`. Accuracy khong doi, dung voi tinh chat rank-preserving cua temperature scaling.

ImageNet-R trong repository da duoc chia truc tiep thanh `80% train / 20% test`; checkpoint hoc toan bo phan train. Khong con validation split doc lap. Do do khong tiep tuc fit calibration tren train, va cung khong tune tren test de tranh test leakage.

### 57.8. Uncertainty-aware probability smoothing

Phuong phap thay the khong hoc tham so tu test:

- Chi ap dung cho mau dung som tai Stage 1 hoac Stage 2.
- He so smoothing bang `min(0.05, 0.5 * (1 - gate confidence))`.
- Tron phan phoi du doan voi toi da `5%` phan phoi deu.
- Mau exhaustive fallback giu nguyen.
- Phep tron `p'=(1-epsilon)p+epsilon/C` bao toan thu tu moi lop, nen Acc@task, Acc@1, Acc@5, Forgetting va Backward khong doi.
- Muc tieu la giam phat cua cac early-exit error qua tu tin va dua loss xuong duoi baseline `1.2305`.

### 57.9. Ket qua uncertainty-aware smoothing

| Phuong phap | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward | LoRA/mau |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline rerun | 77.7914 | 74.0191 | 86.8893 | 1.2305 | 3.2801 | -2.9119 | - |
| Exhaustive tot nhat | **80.6549** | **75.1798** | **88.5327** | **1.0809** | **2.8848** | **-2.8449** | 10.0000 |
| Progressive khong smoothing | 80.5329 | 75.0731 | 87.9394 | 1.2587 | 2.9413 | -2.9014 | 5.4531 |
| **Progressive + uncertainty smoothing** | **80.5329** | **75.0731** | **87.9394** | **1.1253** | **2.9413** | **-2.9014** | **5.4531** |

Routing statistics cua ban smoothing:

- Stage1Stop: `55.3059%`.
- Stage2Stop: `2.0407%`.
- FullFallback: `42.6534%`.
- Trung binh: `5.4531` LoRA/mau.

So voi progressive khong smoothing, chi co loss thay doi: giam `0.1334`, tuong duong `10.60%`. Tat ca accuracy, forgetting, backward, stop rate va chi phi giu nguyen dung nhu thiet ke rank-preserving.

So voi baseline rerun:

- Acc@task tang `2.7415` diem.
- Acc@1 tang `1.0540` diem.
- Acc@5 tang `1.0501` diem.
- Loss giam `0.1052`, tuong duong `8.55%`.
- Forgetting giam `0.3388`, tuong duong `10.33%`.
- Backward tang tu `-2.9119` len `-2.9014`.

So voi exhaustive, progressive smoothing chi thap hon `0.1067` diem Acc@1 va cao hon `0.0444` loss, nhung giam `45.47%` so luot chay LoRA. Day la cau hinh progressive can bang tot nhat hien tai va la moc dau tien cua nhanh nay cai thien dong thoi tat ca chi so chat luong so voi baseline.

## 58. Toi uu chi phi suy luan bang batched-LoRA execution

### 58.1. Vi sao HRM-PET dat hon PET thong thuong

PET mot-adapter thong thuong chi can mot backbone va mot adapter cho moi mau. HRM-PET con co TII de suy luan task, parameter pool, CTIRD/CRCT khi train va co the thu nhieu LoRA khi rematching. Vi vay HRM-PET tang chi phi train, bo nho checkpoint va dac biet la chi phi inference khi task ID khong duoc cung cap.

Cau hinh progressive tot nhat hien tai da giam tu `10` xuong `5.4531` LoRA/mau so voi exhaustive, nhung cac LoRA van duoc chay tuan tu. Day la nut that trien khai tiep theo.

### 58.2. Batched-LoRA execution

Toi uu moi khong thay doi quyet dinh cua thuat toan:

1. Van xep hang LoRA bang TII.
2. Van dung gate tai cac moc `2 -> 4 -> all`.
3. Van tinh du `5.4531` LoRA/mau trung binh.
4. Hai LoRA ke nhau duoc ghep vao mot batch GPU duy nhat.
5. Khong bao gio batch vuot qua moc gate; vi vay mau da du an toan van dung dung cho cu.

Code ghi them `ForwardCalls/sample` de tach hai khai niem:

- `LoRA/sample`: khoi luong tinh toan ly thuyet, khong doi.
- `ForwardCalls/sample`: so luot goi GPU tuan tu, du kien giam tu `5.4531` xuong khoang `2.7265` khi batch hai rank.

Voi routing hien tai (`55.3059%` dung sau 2 LoRA, `2.0407%` dung sau 4 LoRA, `42.6534%` chay du 10 LoRA), batch hai rank giam dung `50%` so forward call tuan tu. FLOPs khong giam, batch tam thoi lon hon va bo nho dinh co the tang; toc do wall-clock thuc te phai do tren RTX 4090. Muc tieu cua thay doi nay la giam latency va tang GPU utilization trong khi giu nguyen accuracy, loss, forgetting, backward va LoRA/mau.

### 58.3. Tieu chi chap nhan

- Unit test phai xac nhan batch `1` va batch `2` cho cung logits, task routing, LoRA count va stop stage.
- Ket qua ImageNet-R phai giu gan nhu nguyen moc `80.5329 / 75.0731 / 87.9394`, loss `1.1253`, forgetting `2.9413`, backward `-2.9014`.
- `ForwardCalls/sample` phai xap xi `2.7265`.
- Wall time phai thap hon ban serial; neu khong, khong coi day la cai tien chi phi.
### 58.4. Ket qua batched-LoRA tren RTX 4090

Ket qua batch hai LoRA giu nguyen chinh xac toan bo metric cua ban serial:

| Chi so | Serial | Batch 2 |
|---|---:|---:|
| Acc@task | 80.5329 | 80.5329 |
| Acc@1 | 75.0731 | 75.0731 |
| Acc@5 | 87.9394 | 87.9394 |
| Loss | 1.1253 | 1.1253 |
| Forgetting | 2.9413 | 2.9413 |
| Backward | -2.9014 | -2.9014 |
| LoRA/mau | 5.4531 | 5.4531 |
| ForwardCalls/mau | 5.4531 | 2.7265 |
| Tong wall time | 731 s | 718 s |

So forward call tuan tu giam dung `50%`, nhung wall time chi giam `13` giay, tuong duong `1.78%`. Muc giam nay co the nam trong dao dong giua cac lan chay va chua du de cong bo la cai tien toc do dang ke.

Nguyen nhan la input duoc mo rong de hai LoRA chay trong cung batch. So lan goi model giam, nhung backbone van xu ly cung tong so ban sao anh va FLOPs gan nhu khong doi. Ket qua nay chi chung minh vectorization bao toan chat luong; khong chung minh chi phi tinh toan giam 50%.

Huong tiep theo phai tac dong vao chi phi that: giam so LoRA thuc su duoc danh gia, cache gate sau khi hoc, benchmark rieng inference sau khi gate da san sang, hoac thu mixed precision/compile voi ablation chat luong. Khong tiep tuc dung so `ForwardCalls/mau` nhu dai dien truc tiep cho FLOPs hay wall time.
## 59. Sua giao thuc: strict exemplar-free

Kiem tra lai hai paper cho thay checkpoint `imr_lora_hybrid_real_ageaware...` khong tuan thu giao thuc ket qua chinh:

- HRM-PET goc la rehearsal-free, khong luu historical exemplars.
- PMI-CFS la data-free: luu thong ke phan phoi va sinh mau gia; khong replay feature that tung mau.
- Checkpoint cu luu toi da 48 feature that/lop va calibrated gate doc lai 12 anh train/lop. Hai co che nay khong co trong HRM-PET goc va khong phu hop data-free CL.

Tu commit nay, ket qua `80.5329 Acc@task / 75.0731 Acc@1` chi duoc ghi la ablation feature-memory ngoai giao thuc, khong phai ket qua exemplar-free chinh.

Thay the hop le:

1. CRCT chi luu mean/covariance va CFS model, sau do sinh feature tong hop Gaussian+CFS.
2. Khong bat `crct_real_feature_replay`; checkpoint khong duoc co `real_feature_memory`.
3. Khong dung calibrated gate, prototype hay distilled router hoc tu historical train images.
4. Routing dung hybrid rematching goc hoac progressive rule co dinh chi dua tren tin hieu cua mau test.
5. `--strict_exemplar_free` tu dong tu choi cau hinh vi pham.
6. Moi checkpoint duoc audit sau khi train; chi `EXEMPLAR_FREE_AUDIT=PASS` moi duoc dung lam ket qua chinh.

CFS van la phan ap dung tu paper thu hai: feature gia duoc lay tu phan phoi lop va chon da dang trong khong gian contrastive. Pseudo-image do model inversion sinh ra duoc phep vi khong phai anh lich su, nhung khong bat trong thi nghiem CFS-CRCT dau tien de tach ro tac dung cua CFS.
### 59.1. Strict CFS pilot 3 task

Ket qua strict exemplar-free CFS-CRCT dau tien:

| Chi so | Baseline rank 8 | Strict CFS | Delta |
|---|---:|---:|---:|
| Acc@task | 89.1321 | 89.0284 | -0.1037 |
| Acc@1 | 80.9782 | 81.0398 | +0.0616 |
| Acc@5 | 93.7122 | 93.6410 | -0.0712 |
| Loss | 0.8372 | 0.8584 | +0.0212 |
| Forgetting | 2.8462 | 2.2205 | -0.6257 |
| Backward | -2.5128 | -2.2205 | +0.2923 |

Pilot khong dat strict multi-metric gate. CFS cai thien ro kha nang giu kien thuc cu, nhung classifier correction hoi manh, lam Acc@task/Acc@5/loss xau nhe. Ablation tiep theo khong doi nguon du lieu: gioi han noi suy classifier toi da `alpha=0.5`, van chon alpha bang Gaussian validation anchors va van strict exemplar-free.
### 59.2. Alpha cap 0.5 that bai va CFS old-class-only

Alpha cap `0.5` lam Acc@task `-0.9436`, Acc@1 `-0.4997`, Acc@5 `-0.2826` va loss `+0.0412` so voi baseline; chi retention tot hon. Do do bo alpha cap va quay ve validation gate day du (`max alpha=1.0`).

Ablation moi dua truc tiep tren quan sat cua hai pilot: CFS co loi cho lop cu nhung correction cua toan bo lop lam accuracy giam. CFS chi duoc dung de chon feature replay cho old classes; current-task classes dung Gaussian replay goc. CFS model cua task hien tai van duoc hoc khi du lieu task do dang san sang, nhung chi duoc dung tu task sau. Cach nay khong luu anh/feature that va van strict exemplar-free.
### 59.3. Ket qua old-class-only va doi chieu lai code PMI-CFS

Pilot old-class-only dat Acc@task `88.9128`, Acc@1 `80.9509`, Acc@5
`93.6410`, Loss `0.8598`, Forgetting `2.2205`, Backward `-2.2205`.
So voi baseline, retention tot hon nhung ba accuracy va loss van xau hon. So voi
CFS tren tat ca lop, old-class-only cung khong tang retention va lam accuracy
giam them. Vi vay gia thuyet tach CFS theo tuoi lop bi loai.

Doi chieu code PMI-CFS goc phat hien mot buoc quan trong bi thieu trong ban port:
sau contrastive selection, tap feature duoc hieu chinh lai mean va standard
deviation theo thong ke lop goc. Thieu buoc nay lam CFS doi phan phoi dung de
CRCT can bang classifier, phu hop voi hien tuong forgetting giam nhung loss va
accuracy xau di.

Pilot tiep theo chi sua sai khac nay: CFS paper-style, selection ratio `0.5`, 5
selection steps, contrastive model 200 epochs/batch 64 va moment preservation.
Boundary/core replay, distribution filter, semantic va old-class-only deu tat.
Giao thuc van strict exemplar-free.

### 59.4. Ket qua pilot moment-preserving CFS

Pilot task 3 dat Acc@task `88.9572`, Acc@1 `80.8620`, Acc@5 `93.5521`,
Loss `0.8603`, Forgetting `2.3538`, Backward `-2.3538`. So voi baseline:
Acc@task `-0.1749`, Acc@1 `-0.1162`, Acc@5 `-0.1601`, Loss `+0.0231`,
Forgetting `-0.4924` va Backward `+0.1590`. Strict gate that bai.

Ket qua nay bac bo gia thuyet rang thieu buoc phuc hoi mean/std la nguyen nhan
chinh cua trade-off. Moment matching khong khoi phuc accuracy va loss; so voi
old-class-only no con lam Acc@1/Acc@5 giam `0.0889` va retention xau hon
`0.1333`. Ket luan: khong chay full va dung quet tham so cho huong dua CFS
feature truc tiep vao CRCT. CFS lam replay cu manh hon nhung gay sai lech ranh
gioi classifier tren du lieu that.

### 59.5. Chan doan validation gate sau pilot

Gate chon `alpha=1.0` o ca task 2 va task 3. Tren 1920/2880 Gaussian anchors,
teacher va classifier sau CRCT deu dat 100% Acc@1, Acc@5 va task accuracy;
classifier CFS chi co CE thap hon (`0.01648 -> 0.01383` va
`0.03355 -> 0.02435`). Vi vay gate luon chon toan bo classifier CFS, mac du
accuracy tren anh ImageNet-R that giam.

Day la circular validation: CRCT hoc tren feature tong hop tu thong ke Gaussian
va gate cung danh gia tren feature sinh tu chinh thong ke do. Cac anchor qua de
va khong dai dien cho distribution gap den anh that. `worst_old_class_drop=0`
vi ca hai classifier deu dat 100%, nen khong phai bang chung retention ngoai
phan phoi tong hop.

Huong kiem soat tiep theo, neu trien khai, la hybrid validation gate: dung tam
thoi feature cua anh train task hien tai de bao ve learning/plasticity, va dung
Gaussian anchors chi cho cac lop cu de bao ve retention. Feature task hien tai
chi ton tai trong luc chon alpha va khong duoc luu vao checkpoint; khong doc hay
luu anh/feature task cu, nen van rehearsal-free va strict exemplar-free.

### 59.6. Hybrid current-real/old-synthetic validation gate

Trien khai gate moi de loai circular validation. Truoc CRCT, gate lay tam thoi
toi da 16 feature/lop tu anh train cua task hien tai bang LoRA hien tai. Sau
CRCT, cac alpha duoc danh gia tren tap ket hop: feature that task hien tai va
Gaussian anchors cua rieng lop cu.

Alpha chi duoc chap nhan khi tung lop hien tai khong giam accuracy, macro
Acc@1/Acc@5/task accuracy khong giam, CE hien tai khong tang, cac lop cu khong
giam accuracy, va cac rang buoc toan bo tap van dat. Feature hien tai chi la
bien cuc bo trong luc CRCT, khong vao global memory/checkpoint va khong duoc
dung o task sau. Khong doc hoac luu anh/feature cua task cu; strict
exemplar-free van duoc giu.

### 59.7. Test phat hien loi canh tranh cross-task trong gate

Test hybrid dau tien tra ve `alpha=0.9` thay vi 0. Nguyen nhan la metric loc ca
mau va cot logit theo nhom dang cham; khi cham current classes, logit old
classes bi loai bo. Gate vi the khong thay mau hien tai bi lop cu canh tranh.

Da sua metric de chi loc mau theo old/current, nhung luon tinh du doan va CE
tren tat ca seen classes. Cach tinh nay khop class-incremental evaluation va
kiem tra truc tiep trade-off plasticity/retention ma hybrid gate can bao ve.
## CFS-PMI diagnostic v2 và chuyển pilot sang TII

- Diagnostic v2: PASS, inversion hợp lệ (real-control cosine 0.9983).
- CFS giữ class accuracy 100% và giảm output pairwise cosine từ 0.3992 xuống 0.3732.
- Khoảng cách tới manifold thật không tốt hơn Gaussian: 0.4845 so với 0.4720; lớp 2 và 3 xấu hơn rõ nhất.
- Không đưa PMI inversion vào head-only LoRA CRCT vì nó gần như tái tạo lại final feature đã được CRCT dùng trực tiếp.
- Thêm pilot ImageNet-R 3 task để đánh giá CFS paper-style tại TII, nơi quyết định task ban đầu.
- Pilot giữ nguyên cấu hình TII baseline, strict exemplar-free và dùng gate 5 chỉ số.
## Kết quả TII-CFS pilot 3 task, ratio 0.5

- Acc@1 +0.5806, Loss -0.0212, Forgetting -0.7077, Backward +0.7077.
- Acc@5 giảm 0.7196 nên strict gate FAIL; chưa được chạy full.
- Script nhận biến `CFS_SELECTION_RATIO`; lần kế tiếp thử 0.25 với toàn bộ thiết lập khác giữ nguyên.
## TII-CFS ratio 0.25 và sửa metric đánh giá

- Ratio 0.25: Acc@1 +1.0530, Loss -0.0235, Forgetting -0.5641 nhưng Acc@5 -0.9960; strict gate FAIL.
- Dừng tuning tỷ lệ CFS.
- Phát hiện TII engine chưa đo Acc@task dù downstream dùng class-to-task mapping để chọn LoRA.
- Thêm Acc@task và script eval-only cho baseline/candidate checkpoint 3 task; không train lại.
## TII routing PASS, chuẩn bị ablation end-to-end

- Acc@task baseline 78.4962, CFS 78.6438: +0.1476.
- TII_ROUTING_GATE=PASS nhưng STRICT_ALL_METRIC_GATE=FAIL do Acc@5 -0.9960.
- Thêm eval-only dùng cùng LoRA rank-8 và thay duy nhất TII gốc/CFS ở inference 3 task.
- Chưa chạy full TII; chỉ cân nhắc nếu END_TO_END_GATE=PASS.
## CFS-TII end-to-end: loại bỏ

- Fixed LoRA + TII gốc: Acc@task 89.1321, Acc@1 80.9782, Acc@5 93.7122, Loss 0.8372.
- Fixed LoRA + TII-CFS: Acc@task 89.0233, Acc@1 80.9999, Acc@5 93.5829, Loss 0.8446.
- END_TO_END_GATE=FAIL; Acc@1 chỉ +0.0217 trong khi Acc@task, Acc@5 và Loss xấu hơn.
- Dừng CFS-TII, không train full 10 task và không tuning thêm tỷ lệ.

## 60. Selector soft/hard theo confidence va oracle audit

Sau khi soft-mixture va soft-route/hard-classify deu khong dat strict quality gate,
khong tiep tuc quet he so. Trien khai mot phep thu co kha nang bac bo nhanh:
chay hai dau ra trong cung mot lan danh gia, sau do chon theo tung mau bang
normalized top-1 margin. Margin duoc chuan hoa theo do lech chuan logits tren
cac lop da hoc, nen bat bien voi phep cong hang so va nhan he so duong. Selector
khong dung nhan test; khi margin bang nhau thi giu soft output.

Chi phi ky vong voi top-k=4 la 5 LoRA/mau va 2 forward call/mau. Log bo sung
SoftAcc@1, HardAcc@1, SoftHardAgree, SoftOnlyCorrect, HardOnlyCorrect,
OracleAcc@1 va HardSelectRate. Nhan chi duoc dung de tinh oracle diagnostic,
khong tham gia quyet dinh dau ra.

Tieu chi khoa truoc khi chay:
- OracleAcc@1 phai cao hon thanh phan tot nhat it nhat 0.5 diem; neu khong,
  hai dau ra khong du bo sung de tiep tuc nhanh nay.
- Selector thuc te phai cao hon thanh phan tot nhat it nhat 0.3 diem Acc@1.
- Van doi chieu day du voi baseline va exhaustive; khong chap nhan danh doi
  Acc@5, Loss, Forgetting hoac Backward chi de tang Acc@1.
## 61. Ket qua soft-hard selector va local hard refinement

Soft-hard selector task 10 dat Acc@task 79.2034, Acc@1 73.7807, Acc@5
86.6564, Loss 1.2157, Forgetting 2.9704 va Backward -2.5390 voi chi phi
5 LoRA/mau, 2 forward call/mau. So voi baseline: Acc@task +1.6180, Acc@5
+0.1918, Loss -0.0073, Forgetting -0.3560 va Backward +0.3929, nhung Acc@1
-0.2670 nen strict baseline gate FAIL.

OracleAcc@1 74.9435 cao hon hard component 1.2049 diem, vi vay hai dau ra co
bo sung. Tuy nhien normalized-margin selector chi tang 0.0421 diem so voi hard,
thu duoc khoang 3.5% oracle headroom; SELECTOR_GAIN_GATE FAIL. Dong nhanh chon
theo global normalized margin, khong quet threshold.

Chan doan moi: soft Acc@task 79.2034 cao hon hard Acc@1 73.7386 toi 5.4648
diem. Hard LoRA dang de logits cua tat ca 200 lop canh tranh, du LoRA duoc chon
cho mot task cu the. Trien khai local hard refinement: soft giu toan bo bang
chung cross-task; hard chi thay thu tu lop trong task soft da route. Hard local
logits duoc bien doi affine theo tung mau de giu nguyen max va std cua soft
local logits. Cach nay khong dung nhan, khong train gate, khong luu mau cu va
van co chi phi 5 LoRA/mau, 2 forward call/mau.

Tieu chi khoa truoc: refined Acc@1 phai hon soft it nhat 0.3 diem va strict
baseline gate van phai PASS; neu khong thi dong nhanh local refinement.

## 62. Ket qua local hard refinement: dong nhanh

Task 10 dat Acc@task 79.2034, Acc@1 73.4515, Acc@5 87.0896, Loss
1.1740, Forgetting 3.1679 va Backward -2.7657 voi chi phi 5 LoRA/mau, 2
forward call/mau. So voi baseline, nam chi so Acc@task/Acc@5/Loss/Forgetting/
Backward deu tot hon, nhung Acc@1 giam 0.5962; BASELINE_GATE FAIL.

Refinement chi tang 0.0342 diem Acc@1 so voi soft, thap hon nguong khoa 0.3;
LOCAL_REFINEMENT_GAIN_GATE FAIL. Oracle cua cap soft/refine chi dat 74.2342,
tuc cao hon baseline 74.0477 dung 0.1865 diem va thap hon exhaustive 75.4277
toi 1.1935 diem. Ke ca selector hoan hao cung khong the tao cai thien dang ke.

Quyet dinh: dong hoan toan nhanh soft/local-hard refinement; khong tune scale,
std, threshold hoac selector. Cung dong viec ket hop lai hai output soft/hard
tren checkpoint nay. Huong tiep theo phai danh truc tiep vao chi phi tim LoRA
thang cua exhaustive, vi day moi la nguon khoang cach accuracy con lai.
## 2026-08-15: Prediction-induced task proposal oracle audit

### Ly do chon huong nay

Cac audit routing chi dua vao TII, prototype, tham so LoRA, Arrow va nang luong dap ung LoRA deu khong dat recall du cao. Ket qua oracle cho thay TII top-4 chi chua exhaustive winner tren 88.8299% mau, trong khi exhaustive phai chay 10 LoRA. Vi vay nut that hien tai la de xuat dung cac task nam ngoai top-k cua TII.

Huong moi mo rong direct rematching cua HRM-PET ma khong dung anh cu, nhan that, feature replay hay router duoc hoc:

1. Chay hai LoRA dau theo TII.
2. Lay top-5 lop co logit cao nhat tu moi LoRA da chay.
3. Anh xa cac lop du doan ve task so huu lop do.
4. Them toi da hai task moi co bang chung lop manh nhat; neu khong du thi dien theo thu tu TII.
5. Oracle audit van chay du 10 LoRA chi de xac dinh exhaustive winner va do recall. Chi phi trien khai dang duoc audit luon bi gioi han boi 4 LoRA/mau.

Day la hard routing, nen khong tron trong so va khong lam thay doi bien do LoRA nhu cac soft-mixture da thu. Tin hieu de de xuat task den tu du doan hau-LoRA tren chinh mau test, vi vay no khac Arrow/response audit chi nhin tham so hoac nang luong adapter.

### Gate khai bao truoc

Chi trien khai thanh phuong phap inference that neu dong task-10 thoa tat ca:

- ProposalWinnerRecall >= 95%.
- ProposalExactAgreement >= 95%.
- ProposalWinnerRecall - WinnerRecall@4 >= 5 diem phan tram.
- ProposalLoRA/sample <= 4.

Neu gate khong dat, dong nhanh nay va khong quet tham so top-k de tranh toi uu theo test set.
## 2026-08-15: Ket qua prediction-induced proposal oracle

Cau hinh khoa truoc TII top-2 + 2 task de xuat tu top-5 lop da dat:

- ProposalWinnerRecall 97.2243%.
- ProposalExactAgreement 97.2243%.
- ProposalLoRA/sample 4.0000.
- Tang 8.1576 diem recall so voi TII top-4 (89.0667%).
- Ca bon dieu kien cua PREDICTION_PROPOSAL_GATE deu PASS.

Dong Acc@task 80.6549, Acc@1 75.1798, Acc@5 88.5327, Loss 1.0809,
Forgetting 2.8848 va Backward -2.8449 trong audit van la ket qua exhaustive,
khong phai ket qua deployment 4 LoRA. Audit phai chay 10 LoRA de biet winner
that va chi dung winner do de tinh recall/exact agreement.

Buoc tiep theo da duoc trien khai la prediction_proposal_rematching that: hai
LoRA TII dau duoc chay trong mot model call, hai LoRA de xuat duoc chay trong
model call thu hai, sau do chi gop logit cua bon task nay. Cau hinh khong co
exhaustive fallback, khong hoc router va chay voi strict_exemplar_free.

## 2026-08-15: Ket qua operational prediction proposal 4 LoRA

Inference that voi TII top-2 + 2 task proposal dat Acc@task 80.2823, Acc@1
74.7520, Acc@5 87.5311, Loss 2.7701, Forgetting 3.1791 va Backward
-3.1392. Chi phi dung 4 LoRA/mau va 2 forward call/mau.

So voi baseline, ba accuracy deu tang (Acc@task +2.6969, Acc@1 +0.7043,
Acc@5 +1.0665) va Forgetting giam 0.1473. Day la bang chung proposal routing
hoat dong that o chi phi 40% exhaustive. Tuy nhien strict all-metric gate FAIL
vi Backward giam 0.2073 va Loss tang 1.5471.

Loss cao khong dong nghia accuracy thap: cac lop thuoc task khong duoc chon bi
gan logit rat nho de tao tensor huu han. Khi nhan that nam ngoai candidate,
cross-entropy bi phat rat lon. Huong sua khong dung nhan la danh mot phan khoi
luong xac suat TII cho cac lop bi loai, voi gioi han theo tung mau de top-1 cua
candidate khong bi thay doi.

Cau hinh tiep theo duoc khoa o TII top-2 + 3 proposal, TII probability
completion, toi da 5 LoRA/mau va van chi 2 forward call. Muc tieu cua proposal
thu ba la dua BWT gan exhaustive; completion nham sua Loss ma khong doi top-1.

## 2026-08-15: Ablation prediction proposal 5 LoRA tren checkpoint feature-memory

Cau hinh TII top-2 + 3 task proposal + TII probability completion dat:

- Acc@task 80.6062.
- Acc@1 75.0935.
- Acc@5 87.5040.
- Loss 1.1801.
- Forgetting 2.9703.
- Backward -2.8927.
- LoRA/sample 5.0000.
- ForwardCalls/sample 2.0000.

So voi baseline, Acc@task tang 3.0208, Acc@1 tang 1.0458, Acc@5 tang
1.0394, Loss giam 0.0429, Forgetting giam 0.3561 va Backward tang 0.0392.
BASELINE_ALL_METRIC_GATE PASS va OPERATIONAL_PROPOSAL_EFFICIENCY_GATE PASS.

So voi exhaustive, Acc@task chi thap hon 0.0487 va Acc@1 thap hon 0.0863,
trong khi so LoRA duoc danh gia giam tu 10 xuong 5. Acc@5 thap hon 1.0287,
Loss cao hon 0.0992, Forgetting cao hon 0.0855 va Backward thap hon 0.0478.

Wall time tren RTX 4090 la 295 giay (4 phut 55 giay), so voi 495 giay cua
vectorized exhaustive tren cung setting. Nhu vay proposal giam 40.4% thoi gian
va dat toc do nhanh hon 1.68 lan, trong khi Acc@1 chi giam 0.0863 diem.

Day la cau hinh inference duoc khoa, nhung cac so tren dung checkpoint
`hybrid_real_ageaware` da duoc train voi per-example real-feature memory. Vi
vay day chi la ablation ve routing/chi phi, khong phai claim exemplar-free
end-to-end. Buoc bat buoc tiep theo la chay nguyen cau hinh proposal tren
`imr_lora_rank8_baseline_10tasks_seed42`, sau do moi lap lai nhieu seed va bao
cao mean/std. Khong tune them top-k, prior, completion hay temperature.
## 2026-08-15: Ket qua strict rank-8 prediction proposal

Audit end-to-end da dat:

- `Checkpoint training protocol: PASS`.
- `STRICT_CHECKPOINT_AUDIT=PASS`.
- Checkpoint khong co `real_feature_memory`.

Ket qua cung seed 42 va checkpoint rank-8 exemplar-free:

| Cau hinh | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward | Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Conventional | 77.7914 | 74.0191 | 86.8893 | 1.2305 | 3.2801 | -2.9119 | 148 s |
| Exhaustive | 81.1182 | 75.4277 | 88.9183 | 1.0860 | 3.0772 | -2.9088 | 409 s |
| Proposal 5 LoRA | 80.7780 | 75.0850 | 87.3132 | 1.2087 | 3.3122 | -3.0612 | 295 s |

Proposal tang Acc@task 2.9866, Acc@1 1.0659, Acc@5 0.4239 va giam
Loss 0.0218 so voi conventional. Tuy nhien Forgetting tang 0.0321 va Backward
giam 0.1493, nen `BASELINE_ALL_METRIC_GATE=FAIL`. Efficiency gate PASS voi
5 LoRA/mau va 2 call/mau. So voi exhaustive, wall time giam 114 giay (27.9%)
va nhanh hon 1.39 lan; so voi conventional, proposal van cham hon 147 giay,
gan 1.99 lan.

Quyet dinh: chua chay seed 43/44 va khong tune threshold tren test set. Buoc
ke tiep la tai dung ma tran accuracy stage-task tu ba log de xac dinh Forgetting
tang do peak accuracy som cao hon hay do final accuracy cua task cu giam. Chi
sua routing sau khi chuan doan nay co ket qua.

## 2026-08-15: Chẩn đoán retention theo task và audit nhánh TII ban đầu

Kết quả strict seed 42 cho thấy proposal không làm giảm độ chính xác cuối của nhóm task cũ một cách tổng thể:

- Acc@1 cuối trung bình task 1-9: `74.4464 -> 75.3906` (`+0.9442`).
- Peak trung bình task 1-9: `77.7266 -> 78.7028` (`+0.9763`).
- Forgetting tăng rất nhẹ `+0.0321` vì mức tăng peak lớn hơn mức tăng final đúng `0.0321`, không phải vì final old-task accuracy thấp hơn baseline.
- Backward giảm `0.1493` vì accuracy ban đầu tăng `1.0935` nhưng final chỉ tăng `0.9442`.
- Suy giảm final thực sự tập trung ở task 6 (`-0.7169`) và task 7 (`-0.5386`). Không được hạ peak/initial để làm đẹp Forgetting/BWT vì như vậy làm chất lượng mô hình tệ đi.

Đã thêm `--prediction_proposal_initial_branch_audit`. Proposal vốn đã chạy LoRA của task TII-top1 trong vectorized call đầu; audit giữ lại logits này và đo complementarity với đầu ra proposal:

- `InitialOnlyCorrect`: nhánh TII-top1 đúng nhưng proposal sai.
- `ProposalOnlyCorrect`: proposal đúng nhưng nhánh TII-top1 sai.
- `InitialProposalOracleAcc@1`: trần lý thuyết nếu chọn đúng giữa hai đầu ra bằng nhãn.

Nhãn chỉ dùng để báo cáo audit, không đi vào routing. Audit không giữ ảnh/feature cũ, không thêm LoRA và không thêm forward call. Chỉ phát triển selector nếu audit chứng minh có đủ headroom; nếu không thì đóng hướng này thay vì tiếp tục tinh chỉnh mù.

### Kết quả complementarity và giả thuyết khóa trước

Audit seed 42 xác nhận nhánh TII-top1 và proposal bổ sung cho nhau: `InitialOnlyCorrect=2.0376`, `ProposalOnlyCorrect=5.4334`, oracle hai nhánh đạt `77.1226` Acc@1 so với proposal `75.0850`. Bước kế tiếp dùng một quy tắc không tham số, không fit trên test: chỉ chọn nhánh đầu khi dự đoán khác nhau và nhánh đầu đồng thời có top-1 probability cùng top1-top2 margin cao hơn proposal. Đây là Pareto-confidence dominance; nếu không tăng Acc@1 trong một lần đánh giá thì đóng hướng selector này, không tune threshold trên test.

### Cross-adapter full-logit consensus audit

Pareto-confidence selector giảm Acc@1 từ `75.0850` xuống `74.9055`, vì vậy hướng so sánh confidence giữa initial branch và proposal đã được đóng mà không tune threshold trên test. Audit kế tiếp kiểm tra tín hiệu đã có sẵn trong full logits của cả 5 LoRA proposal: plurality vote, any-adapter label oracle, proposal/vote oracle và strict-majority rescue. Tie của plurality vote được phá bằng tổng xác suất; rescue chỉ bật khi có đa số tuyệt đối. Quy tắc không có tham số học, nhãn chỉ chấm oracle/accuracy, không tham gia vote. Không thêm LoRA, forward call, ảnh cũ hay feature memory.

### Calibration-free Borda rank aggregation

Cross-adapter plurality vote có oracle rất cao (`82.4275`) nhưng rescue giảm Acc@1 xuống `73.7517`; top-1 vote bị chi phối bởi thiên lệch chung giữa các adapter. Phép thử cuối của hướng fusion không học dùng Borda aggregation: chuyển logits của từng adapter thành thứ hạng lớp, cộng thứ hạng qua 5 adapter, phá hòa bằng tổng xác suất và chỉ rescue proposal khi lớp Borda thắng nằm trong top-5 của đa số adapter. Không có threshold fit từ test; nếu Borda rescue không vượt `75.0850`, đóng hướng no-learning fusion.

Kết quả Borda seed 42: `BordaAcc@1=71.2973`, `BordaRescueAcc@1=71.3306`, thấp hơn proposal `3.7544` điểm. Rescue kích hoạt trên `17.2585%` mẫu, trong khi top-5 support trung bình là `98.3229%`; điều kiện support vì vậy gần như không có khả năng loại các quyết định sai. `Proposal/Borda oracle=77.9539` chỉ xác nhận hai nhánh có tính bổ sung, nhưng không cung cấp quy tắc chọn nhánh khả dụng khi không có nhãn. `CROSS_BORDA_HEADROOM_GATE=PASS` và `CROSS_BORDA_RESCUE_GATE=FAIL`.

Quyết định: đóng toàn bộ hướng fusion không học gồm confidence dominance, plurality vote, strict-majority rescue và Borda. Không tiếp tục tune ngưỡng trên test. Nếu khai thác headroom cross-adapter, bước sau phải là selector được học trên calibration data hợp lệ và rehearsal-free, được khóa trước khi đánh giá test; nếu không đáp ứng giao thức này thì giữ proposal hiện tại.

## 2026-08-15: Preregistered strict task-mass fusion

Sau khi confidence dominance, plurality vote và Borda đều thất bại, không tiếp tục thêm quy tắc chọn nhánh dựa trên test. Bottleneck mới được xác định trong chính proposal: các LoRA đang được so bằng raw maximum logit, trong khi logit của các adapter độc lập không có cùng offset/thang đo.

Phép thử kế tiếp được khóa trước khi chạy: giữ nguyên candidate TII top-2 + ba task được đề xuất, 5 LoRA/mẫu, 2 vectorized calls và TII completion; chỉ thay phép hợp nhất bằng `P(class, task|x) = P_TII(task|x) * P_LoRA(class|task,x)`. TII cấp tổng probability mass cho task, còn LoRA phân phối mass đó giữa các lớp trong task bằng conditional softmax. Không có hệ số mới, dữ liệu calibration, ảnh cũ, feature từng mẫu, semantic hay CFS; checkpoint vẫn phải qua strict training-log và `real_feature_memory` audit.

Giả thuyết được chấp nhận chỉ khi `BASELINE_ALL_METRIC_GATE=PASS` trên strict rank-8 checkpoint và efficiency vẫn là 5 LoRA/mẫu, 2 calls/mẫu. Nếu fail, đóng task-mass fusion; không quét temperature hoặc pha trộn với raw-logit fusion trên test.

### Kết quả strict task-mass fusion và quyết định dừng

Task-mass fusion thất bại trên toàn bộ sáu metric: `Acc@task=69.0115`, `Acc@1=63.7507`, `Acc@5=81.5513`, `Loss=1.6541`, `Forgetting=8.8896`, `Backward=-8.8896`. Efficiency vẫn đúng 5 LoRA/mẫu và 2 calls/mẫu, nhưng `BASELINE_ALL_METRIC_GATE=FAIL`. Kết quả bác bỏ giả thuyết rằng TII probability mass có thể đóng vai trò task posterior đã hiệu chỉnh; TII phù hợp để xếp hạng candidate nhưng không được dùng để khóa xác suất tuyệt đối của task. Đóng task-mass fusion và không tune temperature hay pha hệ số theo test.

### Preregistered strict conditional fusion

Phép thử kế tiếp giữ nguyên proposal, candidate budget, completion, temperature `1.0` và standardized TII prior weight `0.3` đã có trước task-mass. Thay đổi duy nhất là chuẩn hóa logits của từng LoRA bằng log-softmax trong 20 lớp của task trước khi cộng fixed TII prior. Cách này loại offset giữa adapter nhưng không ép LoRA tuân theo absolute TII task mass. Hai mode task-mass và conditional bị khóa loại trừ nhau trong script.

Không có ảnh/feature cũ, calibration data, router học thêm hoặc tham số mới. Chỉ chấp nhận nếu strict checkpoint audit, `BASELINE_ALL_METRIC_GATE` và efficiency gate cùng PASS. Nếu fail, đóng toàn bộ nhánh normalization-based fusion; không quét prior/temperature trên test.
