# BÀN GIAO CÔNG VIỆC: CẢI TIẾN HRM-PET TRÊN IMAGENET-R

**Ngày cập nhật:** 15/08/2026
**Repository:** `Truong140206/Hybrid_ReMatching`
**Mốc code được bàn giao:** từ `9320266` (`record soft-hard result`)
**Máy chạy chính:** Ubuntu, NVIDIA RTX 4090 24 GB
**Dataset chính:** Split-ImageNet-R, 200 lớp, 10 task, 20 lớp/task

---

## 1. Mục tiêu của công việc

Mục tiêu ban đầu là cải tiến HRM-PET trong bài toán continual learning bằng ý
tưởng từ paper PMI-CFS và semantic-aware feature projection, đồng thời giữ giao
thức rehearsal-free/exemplar-free của HRM-PET.

Sau nhiều ablation, vấn đề quan trọng nhất trên ImageNet-R được xác định là
**rematching/routing**: mô hình không phải lúc nào cũng chọn đúng LoRA/task cho
ảnh kiểm thử. Exhaustive rematching cải thiện rõ kết quả nhưng phải chạy tất cả
LoRA, vì vậy hướng hiện tại là tìm cách giữ chất lượng của exhaustive với chi
phí thấp hơn.

Đây chưa phải kết quả cuối để công bố. Một số nhánh cho kết quả tốt nhưng không
đạt toàn bộ tiêu chí hoặc chưa chắc tuân thủ strict exemplar-free.

---

## 2. Môi trường và đường dẫn trên máy RTX 4090

```text
Repository:
/home/s24gbn1/Documents/truongnguyen/Hybrid_ReMatching

Python environment:
/home/s24gbn1/Documents/truongnguyen/Hybrid_ReMatching/.venv/bin/python

Dataset:
/home/s24gbn1/Documents/truongnguyen/datasets/imagenet-r

Output:
/home/s24gbn1/Documents/truongnguyen/hrm-pet-output

TII gốc:
/home/s24gbn1/Documents/truongnguyen/hrm-pet-output/imr_tii_original_10tasks_seed42

Checkpoint đang dùng cho nhóm thí nghiệm rematching mới:
/home/s24gbn1/Documents/truongnguyen/hrm-pet-output/imr_lora_hybrid_real_ageaware_crct30_old035_new010_seed42

Baseline rank-8 strict đã dùng trong các ablation CTIRD:
/home/s24gbn1/Documents/truongnguyen/hrm-pet-output/imr_lora_rank8_baseline_10tasks_seed42
```

Checkpoint `hybrid_real_ageaware` có LoRA rank 5. Script mới tự đọc rank từ
checkpoint; không được mặc định ép `LORA_RANK=8`.

---

## 3. Quy tắc thực nghiệm bắt buộc

Một phương pháp chỉ được xem là cải thiện toàn diện khi đồng thời thỏa:

- `Acc@task`, `Acc@1`, `Acc@5` cao hơn hoặc bằng reference;
- `Loss`, `Forgetting` thấp hơn hoặc bằng reference;
- `Backward` cao hơn, tức gần 0 hơn;
- cùng dataset split, seed, backbone, checkpoint và giao thức;
- không chọn tham số dựa trên test set;
- chi phí phải báo bằng cả `LoRA/sample`, `ForwardCalls/sample` và wall time.

### Strict exemplar-free

Kết quả dùng làm claim chính phải:

- không lưu ảnh task cũ;
- không lưu feature theo từng mẫu của task cũ;
- không dùng calibration memory lấy từ ảnh/feature cũ;
- có thể lưu thống kê phân phối theo lớp như mean/covariance/centroid;
- có thể sinh pseudo-feature từ thống kê đã lưu.

**Cảnh báo:** tên run `hybrid_real_ageaware` xuất phát từ nhánh thử nghiệm real
feature replay. Trước khi dùng checkpoint này cho claim strict exemplar-free,
phải kiểm tra Namespace/log huấn luyện để xác nhận có bật
`crct_real_feature_replay` hay không. Nếu đã bật, checkpoint này chỉ được dùng
cho ablation inference/cost, không được so trực tiếp như kết quả strict của
paper.

Giá trị `strict_exemplar_free=False` trong Namespace của một lần `--eval` chỉ
là mặc định parser khi đánh giá; nó không đủ để kết luận checkpoint đã được
train theo giao thức nào. Phải xem log train gốc.

---

## 4. Ý nghĩa các chỉ số

| Chỉ số | Ý nghĩa | Hướng tốt |
|---|---|---|
| Acc@task | Tỷ lệ chọn đúng task/LoRA | Cao hơn |
| Acc@1 | Nhãn đúng đứng đầu | Cao hơn |
| Acc@5 | Nhãn đúng nằm trong 5 dự đoán đầu | Cao hơn |
| Loss | Cross-entropy trung bình | Thấp hơn |
| Forgetting | Mức giảm của task cũ từ thời điểm tốt nhất | Thấp hơn |
| Backward | Ảnh hưởng của task mới lên task cũ | Cao hơn/gần 0 hơn |
| LoRA/sample | Số nhánh LoRA tính cho mỗi ảnh | Thấp hơn |
| ForwardCalls/sample | Số lượt gọi model rematching cho mỗi ảnh | Thấp hơn |

`Acc@1` là chỉ số phân loại chính. Không được tuyên bố tốt hơn chỉ vì
`Acc@task` tăng nếu `Acc@1` giảm.

---

## 5. Kết quả rematching quan trọng nhất hiện tại

Các dòng dưới đây dùng cùng checkpoint
`imr_lora_hybrid_real_ageaware_crct30_old035_new010_seed42`.

| Phương pháp | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward | LoRA/mẫu | Forward/mẫu |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Conventional cùng checkpoint | 77.5854 | 74.0477 | 86.4646 | 1.2230 | 3.3264 | -2.9319 | không ghi | không ghi |
| Soft-mixture top-4 | 79.2034 | 73.4173 | 87.0402 | 1.1775 | 2.9761 | -2.6194 | 4 | 1 |
| Soft-route/hard-classify | 79.2034 | 73.7386 | 86.5075 | 1.2210 | 3.0484 | -2.6125 | 5 | 2 |
| Exhaustive chính xác | 81.1182 | 75.4277 | 88.9183 | 1.0860 | 3.0772 | -2.9088 | 10 | 3 vectorized |

### Kết luận từ bảng

1. Exhaustive là phương pháp tốt nhất về accuracy và loss.
2. Soft-mixture giảm 60% số LoRA và chỉ cần một model call, đồng thời cải thiện
   routing/retention, nhưng Acc@1 giảm 0.6304 so với conventional.
3. Lượt hard lấy lại 0.3213 Acc@1 so với soft-mixture, nhưng làm Acc@5, Loss và
   Forgetting xấu hơn, đồng thời tăng chi phí lên 5 LoRA và 2 call.
4. Cả soft-mixture và soft-hard đều `quality gate = FAIL`.
5. Không quét temperature/top-k tiếp trên test set.

### So soft-hard với exhaustive

- Acc@task thấp hơn `1.9148`;
- Acc@1 thấp hơn `1.6891`;
- Acc@5 thấp hơn `2.4108`;
- Loss cao hơn `0.1350`;
- Forgetting tốt hơn `0.0288`;
- Backward tốt hơn `0.2963`;
- chi phí giảm từ 10 xuống 5 LoRA và từ 3 xuống 2 model call.

---

## 6. Exhaustive vectorized

Code vectorized exhaustive gom nhiều task LoRA vào một batch model call nhưng
vẫn tính đủ 10 LoRA.

Kết quả:

```text
Acc@task: 81.1182
Acc@1: 75.4277
Acc@5: 88.9183
Loss: 1.0860
Forgetting: 3.0772
Backward: -2.9088
VECTORIZED_EQUIVALENCE_GATE=PASS
```

Tất cả metric khớp tuyệt đối exhaustive tuần tự. Tuy nhiên wall-time chỉ nhanh
hơn `1.032x`, khoảng 3.2%, vì FLOPs của 10 LoRA vẫn còn nguyên. Đây là tối ưu
kỹ thuật đúng nhưng không phải đóng góp giảm chi phí đủ mạnh.

Tài liệu chi tiết: `VECTORIZED_EXHAUSTIVE_RESULT_VI.md`.

---

## 7. Các hướng đã thử và quyết định

### 7.1. CFS trong CRCT

CFS được chèn vào bước sinh/chọn pseudo-feature cho classifier correction.
Trên CIFAR100 từng có tín hiệu tăng accuracy, nhưng trên ImageNet-R các cấu
hình CFS thường tạo trade-off: retention có thể tốt hơn nhưng accuracy/loss
xấu đi, hoặc correction quá mạnh làm forgetting tăng.

Các biến thể đã thử:

- CFS paper-style;
- moment matching;
- boundary/core replay;
- distribution filter;
- old-class-only;
- validation gate synthetic;
- hybrid current-real/old-synthetic gate.

Kết luận: chưa có biến thể nào cải thiện đồng thời toàn bộ metric. Không tiếp
tục tuning CFS trực tiếp trong CRCT nếu không thay đổi nguyên lý chọn mẫu.

### 7.2. CFS tại TII

Pilot 3 task cho thấy Acc@1, Loss và Forgetting tốt hơn nhưng Acc@5 giảm gần
1 điểm. Khi thay TII-CFS vào cùng LoRA checkpoint, end-to-end gate vẫn fail:
Acc@1 chỉ tăng 0.0217 trong khi Acc@task, Acc@5 và Loss xấu hơn.

Kết luận: đóng nhánh CFS-TII; không train full 10 task.

### 7.3. Semantic

Đã thử semantic distillation/projection và semantic teacher selection. Các
thay đổi thường rất nhỏ hoặc làm Forgetting/Backward xấu đi. Sparse semantic
CTIRD có cơ sở hơn projection trực tiếp, nhưng chưa tạo kết quả full 10 task
cải thiện toàn diện.

Không bật semantic trong nhánh rematching hiện tại.

### 7.4. CTIRD online aligned

CTIRD được sửa để teacher dùng đúng batch student và chọn teacher theo task.
Pilot 3 task từng cải thiện mạnh, nhưng full 10 task với sum reduction làm
regularization tăng theo số teacher và khiến retention xấu đi.

Mean reduction sửa một phần vấn đề này. Pilot task 3/5 có tín hiệu Acc@1 tốt
hơn, nhưng chưa vượt baseline trên tất cả metric ở chuỗi dài.

### 7.5. Parameter-only routing

Hai audit đã loại hướng dùng tham số LoRA để chọn candidate:

- Arrow rank-1: `ArrowRecall@4 = 38.7908`, gate FAIL;
- full-rank response: `ResponseRecall@4 = 38.0790`, gate FAIL;
- TII `WinnerRecall@4 = 88.8299` tốt hơn rất nhiều.

Không thử thêm SVD, pooling, threshold hoặc union trên test.

### 7.6. Selective/progressive/distilled routing

Đã thử top-k cố định, adaptive top-k, cascade, distilled router, budgeted
fallback và calibrated progressive gate. Một số cấu hình giảm chi phí nhưng
accuracy/loss giảm đáng kể. Calibrated progressive tốt nhất giữ gần exhaustive
với khoảng 5.45 LoRA/mẫu, nhưng dùng calibration data và cần kiểm tra strict
protocol trước khi coi là kết quả chính.

---

## 8. Code mới quan trọng

| File | Vai trò |
|---|---|
| `engines/soft_mixture_rematching.py` | Soft-mixture và soft-route/hard-classify |
| `vits/hrm_lora_vision_transformer.py` | Nhận `ensemble_weights` và trộn residual QKV theo từng mẫu |
| `engines/vectorized_exhaustive_rematching.py` | Exhaustive chính xác theo task chunk |
| `engines/hrm_lora_wtp_and_tap_engine.py` | Nối các mode vào evaluation và ghi metric chi phí |
| `configs/imr_lora.py` | Các cờ cấu hình rematching |
| `tests/test_soft_mixture_rematching.py` | Test trọng số LoRA, một forward và hard second pass |
| `tests/test_vectorized_exhaustive_rematching.py` | Test tương đương exhaustive |
| `training_scripts/eval_imagenet_r_soft_mixture_4090.sh` | Chạy soft hoặc hard, tự nhận rank checkpoint |
| `training_scripts/eval_imagenet_r_vectorized_exhaustive_4090.sh` | Chạy exhaustive vectorized |

Các commit gần nhất:

```text
9320266 record soft-hard result
1ca4582 add soft-route hard-LoRA evaluation
58abe73 record soft mixture result and fix references
2b6a2e7 detect LoRA rank for soft mixture eval
9ec129f evaluate one-pass soft LoRA mixture
d0abf95 record vectorized exhaustive benchmark
043cde3 vectorize exact exhaustive rematching
a1b0061 audit full-rank LoRA response routing
```

---

## 9. Lệnh kiểm tra repository trên máy 4090

```bash
cd ~/Documents/truongnguyen/Hybrid_ReMatching
git pull --ff-only
git rev-parse --short HEAD
git status --short
```

HEAD tại thời điểm bàn giao phải ít nhất là `9320266`. `git status --short`
không nên in gì trước khi bắt đầu thay đổi mới.

Chạy unit test liên quan:

```bash
cd ~/Documents/truongnguyen/Hybrid_ReMatching

.venv/bin/python -m pytest -q \
  tests/test_soft_mixture_rematching.py \
  tests/test_vectorized_exhaustive_rematching.py
```

---

## 10. Lệnh tái hiện thí nghiệm mới nhất

### Soft-mixture top-4

```bash
cd ~/Documents/truongnguyen/Hybrid_ReMatching
git pull --ff-only
unset LORA_RANK

bash training_scripts/eval_imagenet_r_soft_mixture_4090.sh \
  ~/Documents/truongnguyen/hrm-pet-output/imr_lora_hybrid_real_ageaware_crct30_old035_new010_seed42 \
  4 \
  1.0 \
  0.3 \
  1.0 \
  soft
```

### Soft-route/hard-classify

```bash
cd ~/Documents/truongnguyen/Hybrid_ReMatching
git pull --ff-only
unset LORA_RANK

bash training_scripts/eval_imagenet_r_soft_mixture_4090.sh \
  ~/Documents/truongnguyen/hrm-pet-output/imr_lora_hybrid_real_ageaware_crct30_old035_new010_seed42 \
  4 \
  1.0 \
  0.3 \
  1.0 \
  hard
```

Script tự phát hiện checkpoint rank 5. Nếu log hoàn chỉnh đã tồn tại, script
sẽ từ chối ghi đè. Không xóa checkpoint để chạy eval.

---

## 11. Bước tiếp theo được khuyến nghị

Không nên thử thêm một fusion/temperature ngẫu nhiên. Bước tiếp theo phải là
**oracle complementarity audit** giữa output soft và hard:

1. Đo tỷ lệ hai output dự đoán giống nhau.
2. Đo số mẫu soft đúng/hard sai và hard đúng/soft sai.
3. Tính oracle Acc@1 nếu được chọn output đúng cho từng mẫu.
4. Thử đúng một selector không dùng nhãn: chọn theo normalized top-1 margin
   hoặc entropy.
5. Test label chỉ dùng để báo oracle/audit, không được dùng để fit selector.

Tiêu chí quyết định đặt trước:

- nếu oracle Acc@1 không cao hơn output tốt nhất ít nhất `0.5` điểm, đóng nhánh
  soft/hard vì hai output không đủ bổ sung cho nhau;
- nếu oracle có headroom nhưng selector không lấy lại ít nhất `0.3` điểm Acc@1
  mà vẫn giữ Acc@5/Loss/Forgetting, đóng selector;
- chỉ khi selector vượt conventional trên cả sáu metric mới cân nhắc chạy seed
  khác hoặc viết thành đóng góp.

Sau audit này, ưu tiên học thuật là thiết lập lại một baseline/checkpoint strict
exemplar-free hoàn toàn, rồi đánh giá exhaustive và phương pháp giảm chi phí
trên đúng checkpoint đó. Không trộn kết quả rank 5/rank 8 hoặc strict/non-strict.

---

## 12. Những điều không được làm tiếp

- Không quét top-k, temperature, prior hoặc fusion weight trên test set.
- Không tuyên bố CFS cải thiện HRM-PET chỉ từ diagnostic inversion.
- Không dùng checkpoint real-feature replay làm kết quả strict nếu chưa xác minh.
- Không so hai run khác rank/seed/checkpoint rồi gọi delta là cải tiến phương pháp.
- Không chỉ báo accuracy mà bỏ qua Loss, Forgetting và Backward.
- Không chạy lại training dài khi có thể dùng checkpoint và eval-only.
- Không xóa output/checkpoint cũ trước khi xác nhận đường dẫn và dung lượng.

---

## 13. Tài liệu liên quan

- `EXPERIMENT_PROGRESS_VI.md`: nhật ký đầy đủ theo thời gian.
- `IMAGENET_R_EXPERIMENT_PROGRESS_VI.md`: các thí nghiệm ImageNet-R ban đầu.
- `CFS_IMPLEMENTATION_NOTE_VI.md`: cách CFS được triển khai.
- `CFS_PMI_RESEARCH_RESET_VI.md`: diagnostic PMI-CFS và quyết định nghiên cứu.
- `SEMANTIC_TOPK_NOTE_VI.md`: semantic-aware experiments.
- `ARROW_ROUTING_AUDIT_RESULT_VI.md`: kết quả Arrow audit.
- `LORA_RESPONSE_AUDIT_RESULT_VI.md`: full-rank response audit.
- `VECTORIZED_EXHAUSTIVE_RESULT_VI.md`: exhaustive vectorized.

---

## 14. Trạng thái cuối khi bàn giao

- Code soft-mixture, soft-hard và vectorized exhaustive đã được push.
- Unit test soft-mixture/vectorized đã từng PASS trên máy 4090.
- Soft-mixture efficiency PASS nhưng quality FAIL.
- Soft-hard efficiency PASS nhưng quality FAIL.
- Exhaustive vẫn là upper-performance reference tốt nhất hiện tại.
- Chưa có phương pháp mới vừa vượt exhaustive/conventional về chất lượng vừa
  giảm mạnh chi phí.
- Bước hợp lý tiếp theo là audit complementarity, không phải tuning mù.

## 15. Thí nghiệm đang chờ chạy: soft-hard confidence selector

Mục tiêu của bước này là kiểm tra dứt điểm liệu hai đầu ra soft-mixture và
soft-route/hard-classify có sửa lỗi bổ sung cho nhau hay không. Mỗi mẫu được
chạy qua cả hai nhánh; bộ chọn không dùng nhãn so sánh top-1 margin đã chuẩn
hóa theo độ lệch chuẩn logits. Nhãn test chỉ được dùng sau quyết định để báo
cáo oracle headroom và không đi vào inference.

Chế độ mới: `selector` trong
`training_scripts/eval_imagenet_r_soft_mixture_4090.sh`.
Với top-k=4, chi phí dự kiến là 5 LoRA/mẫu và 2 forward call/mẫu. Dòng cuối sẽ
có thêm `SoftAcc@1`, `HardAcc@1`, `SoftHardAgree`, `SoftOnlyCorrect`,
`HardOnlyCorrect`, `OracleAcc@1`, `HardSelectRate`, cùng hai gate:
`SELECTOR_ORACLE_HEADROOM_GATE` và `SELECTOR_GAIN_GATE`.

Quyết định đã khóa: oracle headroom dưới 0.5 điểm thì đóng nhánh; selector tăng
dưới 0.3 điểm Acc@1 so với thành phần tốt nhất thì cũng đóng nhánh. Không quét
threshold hoặc hệ số sau khi thấy kết quả.

## 16. Kết quả selector và thí nghiệm local refinement kế tiếp

Selector theo normalized margin đã FAIL: Acc@1 73.7807, thấp hơn baseline
0.2670 điểm dù năm chỉ số còn lại đều tốt hơn. Oracle headroom đạt 1.2049 điểm
nhưng selector chỉ lấy được 0.0421 điểm, nên không được tune threshold tiếp.

Mode kế tiếp là `refine`. Soft-mixture giữ điểm giữa các task; hard LoRA chỉ
sắp xếp lại lớp thuộc task đã dự đoán. Phép chuẩn hóa giữ nguyên max và std của
soft logits trong task, do đó không phá task evidence, không dùng nhãn và không
huấn luyện/lưu replay. Chi phí dự kiến vẫn 5 LoRA/mẫu, 2 forward call/mẫu.
Gate khóa trước: `LOCAL_REFINEMENT_GAIN_GATE` yêu cầu ít nhất +0.3 Acc@1 so với
soft, đồng thời `BASELINE_GATE` phải PASS.
## 17. Local refinement đã bị loại

Kết quả cuối: Acc@1 73.4515, thấp hơn baseline 0.5962; gain so với soft chỉ
0.0342 nên cả BASELINE_GATE và LOCAL_REFINEMENT_GAIN_GATE đều FAIL. Oracle
soft/refine chỉ đạt 74.2342, cao hơn baseline 0.1865 và thấp hơn exhaustive
1.1935 điểm. Đây là trần toán học của cặp output này, vì vậy không được tune
thêm bất kỳ scale/threshold/selector nào cho nhánh soft-hard.

Hướng tiếp theo phải xử lý trực tiếp bài toán tìm LoRA thắng của exhaustive với
ít lần đánh giá hơn; không tiếp tục trộn hoặc chọn giữa các output yếu hơn.