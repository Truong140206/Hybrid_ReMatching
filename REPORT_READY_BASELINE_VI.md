# BASELINE CHỐT ĐỂ VIẾT BÁO CÁO (FALLBACK AN TOÀN)

**Trạng thái:** đây là cấu hình + kết quả đã hoàn tất, đủ vững để viết báo cáo.
Mọi thử nghiệm sau (router audit, tail-completion, dataset mới…) nếu FAIL thì
**quay về đúng điểm này** để viết. Không sửa/không tune gì trong file này.

**Commit đóng băng:** `2c3b5f9` (`2c3b5f94a5f6e6f3833dbc4b7ee56b001ec2012d`)
**Nhánh:** `main`. Quay về: `git checkout 2c3b5f9` (hoặc tạo tag để chắc:
`git tag report-baseline-4seed 2c3b5f9 && git push origin report-baseline-4seed`).

---

## 1. Phương pháp chốt

**closure + TII tail** (adaptive prediction closure, hoàn tất bằng TII probability
mass cho tail của các task chưa đánh giá, top-1-safe).

Cấu hình **khóa cứng, y hệt mọi seed, không tune**:

```text
initial_count = 2 (TII top-2)
top_classes   = 5
prior_weight  = 0.3
temperature   = 1.0
TII tail      = top-1-safe completion
LoRA rank     = 8 (strict), backbone ViT-B/16 đóng băng
dataset       = Split-ImageNet-R, 200 lớp, 10 task, 20 lớp/task
strict exemplar-free = PASS (không lưu ảnh/feature từng mẫu của task cũ)
```

## 2. Kết quả 4 seed (42/43/44/45)

### Bảng chính — Acc@1 (mean ± std)

| Phương pháp | Acc@1 | Acc@5 | Loss | LoRA/mẫu | Calls/mẫu |
|---|---:|---:|---:|---:|---:|
| Conventional | 73.94 ± 0.48 | 86.71 | 1.228 | 1.0 | 1.0 |
| **closure+tail** | **75.14 ± 0.51** | 88.15 | 1.137 | **5.53** | 2.58 |
| Exhaustive | 75.24 ± 0.34 | 89.04 | 1.080 | 10.0 | 3.0 |

closure+tail: **+1.20 Acc@1 so conventional**, **−0.10 so exhaustive**, ở **55.3%
số LoRA** (tiết kiệm 44.7%).

### Per-seed — closure+tail (chính xác)

| Seed | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward | LoRA | Calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 81.1152 | 75.4164 | 87.9218 | 1.1421 | 3.0898 | -2.8964 | 5.5593 | 2.5995 |
| 43 | 80.7403 | 75.6634 | 88.7109 | 1.1295 | 3.0446 | -3.0291 | 5.5277 | 2.5817 |
| 44 | 79.9995 | 74.5076 | 87.7681 | 1.1418 | 3.4665 | -3.4134 | 5.5172 | 2.5883 |
| 45 | 80.2905 | 74.9884 | 88.2097 | 1.1365 | 3.8304 | -3.8304 | 5.5137 | 2.5810 |

### Per-seed — Exhaustive (chính xác)

| Seed | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward | LoRA |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 81.1182 | 75.4277 | 88.9183 | 1.0860 | 3.0772 | -2.9088 | 10 |
| 43 | 80.6896 | 75.6188 | 89.5382 | 1.0710 | 3.1120 | -3.0964 | 10 |
| 44 | 80.3613 | 74.9037 | 88.7369 | 1.0810 | 3.2152 | -3.1111 | 10 |
| 45 | 80.3774 | 75.0261 | 88.9492 | 1.0826 | 3.6344 | -3.6188 | 10 |

### Per-seed — Conventional

| Seed | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 77.7914 | 74.0191 | 86.8893 | 1.2305 | 3.2801 | -2.9119 |
| 43 | (grep log) | 74.4077 | 86.5177 | 1.2213 | 2.5989 | -2.4084 |
| 44 | 77.3359 | 73.2768 | 86.5352 | 1.2343 | 3.8695 | -3.6567 |
| 45 | 77.9417 | 74.0719 | 86.8835 | 1.2262 | 3.6919 | -3.6468 |

## 3. Parity vs exhaustive (mean paired-delta, 4 seed)

| Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---:|---:|---:|---:|---:|---:|
| −0.10 | −0.10 | −0.88 | +0.057 | +0.098 | −0.108 |

Khớp exhaustive trong ~0.1 điểm ở Acc@task/Acc@1/Forgetting/Backward; gap nhất
quán duy nhất = **Acc@5 (−0.88) + Loss (+0.057) = phần TAIL** (class của task chưa
đánh giá).

## 4. All-six-vs-conventional (trung thực)

- **Thắng literal cả 6:** seed 42, seed 44.
- **5/6 (chỉ Forgetting/Backward fail):** seed 43, seed 45 — **đã chứng minh bằng
  taskwise là ARTIFACT peak/final**: final old-task Acc@1 cao hơn conventional
  (+1.08 seed 43, +0.79 seed 45), retention thực sự được bảo toàn. Forgetting
  "tệ" chỉ vì peak tăng nhiều hơn final (plasticity cao hơn).
- **closure+tail luôn thắng Acc@task/Acc@1/Acc@5/Loss trên CẢ 4 seed.**

## 5. Lệnh tái lập (đã dùng, không đổi)

```bash
# Mỗi seed: prepare (train TII+rank8+conventional) rồi eval
bash training_scripts/reproduce_imagenet_r_closure_seed_4090.sh prepare <SEED>
bash training_scripts/eval_imagenet_r_vectorized_exhaustive_4090.sh <RUN_DIR> 4 0.3 1.0
bash training_scripts/eval_imagenet_r_prediction_closure_tii_tail_4090.sh <RUN_DIR>
```

Log gates: `CLOSURE_TII_TAIL_ALL_METRIC_GATE`, `VECTORIZED_EQUIVALENCE_GATE`,
`CONVENTIONAL_REPRODUCTION_GATE`, `STRICT_CHECKPOINT_AUDIT`.

## 6. Khung viết báo cáo (theo SOHO-CL, đã cải tiến)

Giới thiệu → Thiết lập → Baseline (conventional/exhaustive, math) → Phương pháp
(closure+tail, math) → Đánh đổi cost↔quality → Kết quả (bảng mean±std + paired-
delta + bảng chi phí) → So sánh → Mục negative-result CFS (phân tích cơ chế) →
Hạn chế (Acc@5 gap, tail) → Hướng phát triển (router rẻ hơn, tail-completion,
dataset/task-scaling) → Kết luận. Headline: **strict exemplar-free** + **near-
exhaustive ở 44.7% ít LoRA hơn**.

## 7. Negative results phải ghi trung thực trong báo cáo

- **CFS/PMI (bài kia) không ghép được** — 3 lý do cơ chế: (i) PMI-CFS là data-free
  còn HRM-PET có data thật; (ii) PMI inversion cần backbone thay đổi, HRM-PET
  đóng băng; (iii) bottleneck là routing, CFS không đụng tới. 3 điểm cắm
  (CFS-CRCT, CFS-TII, PMI inversion) đều FAIL.
- **Routing-only budget <5 LoRA**: ~10 biến thể (owner-mass, consensus, majority,
  self-owner, beam…) đều mất chất lượng; owner-mass 3.05 LoRA nhưng Acc@5 tụt về
  ~conventional. closure+tail 5.53 là điểm Pareto tốt.
