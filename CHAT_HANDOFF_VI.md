# BÀN GIAO CHO CHAT MỚI: HRM-PET + PMI-CFS TRÊN IMAGENET-R

**Cập nhật:** 2026-08-16
**Repository:** `https://github.com/Truong140206/Hybrid_ReMatching`  
**Nhánh:** `main`  
**CRM đã đánh giá:** `c02c89c`; bản sửa unit test `f35bac7`

## 1. Việc đang làm

Mục tiêu là cải tiến HRM-PET trên Split ImageNet-R, đồng thời tìm một cách áp
dụng hợp lệ ý tưởng từ paper PMI-CFS. Phương pháp cuối cùng phải:

- Rehearsal-free/exemplar-free đúng giao thức của HRM-PET.
- Không lưu ảnh của task cũ.
- Không lưu feature theo từng mẫu của task cũ.
- Không dùng nhãn test để chọn route, hiệu chỉnh hoặc chọn siêu tham số.
- Cải thiện accuracy nhưng không đánh đổi rõ rệt Forgetting/Backward.
- Có chi phí inference thực tế hơn exhaustive rematching.
- Tách rõ đóng góp nhân quả của từng cơ chế; không gọi một biến thể là CFS nếu
  CFS không thực sự tạo ra cải thiện.

## 2. Máy chạy và đường dẫn

Máy Linux qua AnyDesk, một RTX 4090 24 GB:

```bash
cd ~/Documents/truongnguyen/Hybrid_ReMatching
source .venv/bin/activate
```

Các đường dẫn chính:

```text
Repo: /home/s24gbn1/Documents/truongnguyen/Hybrid_ReMatching
Dataset: /home/s24gbn1/Documents/truongnguyen/datasets/imagenet-r
Output: /home/s24gbn1/Documents/truongnguyen/hrm-pet-output
TII checkpoint: /home/s24gbn1/Documents/truongnguyen/hrm-pet-output/imr_tii_original_10tasks_seed42
Strict rank-8 LoRA: /home/s24gbn1/Documents/truongnguyen/hrm-pet-output/imr_lora_rank8_baseline_10tasks_seed42
```

Mỗi lệnh gửi cho người dùng phải có `cd` và `git pull --ff-only`. Sau khi sửa
code, tự commit và push lên `origin/main`.

## 3. Giao thức strict bắt buộc

Checkpoint dùng để claim chính thức phải qua:

```text
Checkpoint training protocol: PASS
STRICT_CHECKPOINT_AUDIT=PASS
```

Được phép giữ LoRA/parameter pool, classifier, TII model và thống kê gộp theo
lớp như mean/covariance. Không được phép dùng:

- `real_feature_memory`.
- Historical image replay.
- Historical per-example feature replay.
- Router/gate học từ ảnh hoặc feature từng mẫu của task cũ.
- Calibration set được dựng bằng cách đọc lại dữ liệu train lịch sử.
- Chọn cấu hình tốt nhất bằng cách quét trên test seed 42.

Checkpoint `hybrid_real_ageaware...` từng cho kết quả tốt nhưng có
`real_feature_memory`; chỉ là ablation, không phải kết quả strict.

## 4. Hai paper và vị trí liên hệ

### HRM-PET

- TII dự đoán task/LoRA ban đầu.
- DRM dùng lớp dự đoán từ LoRA đầu tiên để rematch trực tiếp.
- CRM dùng GEN để phát hiện prediction độ tin cậy thấp trước rematching; bước
  sau chọn candidate output bằng energy, không xem GEN là task posterior.
- CTIRD dùng ảnh task hiện tại qua nhiều LoRA để học quan hệ instance xuyên
  task; không cần lưu ảnh cũ.

### PMI-CFS

- CFS học contrastive MLP trên feature thật của lớp hiện tại.
- Candidate feature được lấy từ Gaussian theo lớp.
- CFS chọn greedily các feature ít giống tập đã chọn trong không gian
  contrastive để phủ phân phối đa dạng hơn.
- Feature được chọn là target cho PMI/full-model inversion.
- Synthetic image sau inversion mới được replay trong pipeline gốc.

Phải nói trung thực: các thử nghiệm CFS trong repo chưa chứng minh được đóng góp
dương end-to-end trên HRM-PET. CRM confidence mới nhất là mở rộng HRM-PET, chưa
phải bằng chứng tích hợp thành công PMI-CFS.

## 5. Các mốc kết quả strict đáng tin

ImageNet-R, 10 task, ViT-B/16, seed 42, strict rank-8 checkpoint:

| Phương pháp | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward | LoRA/mẫu | Calls/mẫu | Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Conventional | 77.7914 | 74.0191 | 86.8893 | 1.2305 | 3.2801 | -2.9119 | - | - | 148 s |
| Exhaustive | 81.1182 | 75.4277 | 88.9183 | 1.0860 | 3.0772 | -2.9088 | 10 | 3 | 409 s |
| Proposal 5 LoRA | 80.7780 | 75.0850 | 87.3132 | 1.2087 | 3.3122 | -3.0612 | 5 | 2 | 295 s |
| Proposal 6 LoRA | 81.0030 | 75.2882 | 88.0714 | 1.1622 | 3.1699 | -2.9588 | 6 | 2 | 333 s |
| Iterative 6 LoRA | 81.0181 | 75.3193 | 87.8836 | 1.1573 | 3.1319 | -2.9634 | 6 | 3 | 329 s |

Nhận xét:

- Exhaustive tốt nhất nhưng chạy cả 10 LoRA/mẫu.
- Proposal 5 LoRA nhanh hơn exhaustive khoảng 1.39 lần, nhưng Forgetting tăng
  `0.0321` và Backward giảm `0.1493` so với conventional; strict gate FAIL.
- Proposal 6 LoRA gần exhaustive hơn. So với conventional chỉ Backward kém
  khoảng `0.0469`; strict gate vẫn FAIL.
- Iterative thêm một forward call nhưng không tạo lợi ích đủ rõ, nên đã đóng.

Proposal 5 LoRA tăng final old-task Acc@1 trung bình `+0.9442`, nhưng task 6
giảm `0.7169` và task 7 giảm `0.5386`. Backward giảm một phần vì accuracy ban
đầu tăng nhiều hơn final accuracy, dù final old-task accuracy trung bình tăng.

## 6. Exhaustive và prediction proposal

Exhaustive chạy mọi LoRA đã học cho mỗi ảnh rồi chọn kết quả tốt nhất. Chất lượng
cao nhưng chi phí tăng tuyến tính theo số task.

Prediction proposal:

1. TII chọn hai LoRA đầu.
2. Chạy hai LoRA trong một vectorized forward.
3. Dự đoán lớp để đề xuất thêm task chưa chạy.
4. Chạy ba hoặc bốn LoRA đề xuất trong forward thứ hai.
5. Ghép kết quả và dùng TII completion cho lớp ngoài candidate.

Oracle audit: proposal 4 task đạt winner recall `97.2243%`, cao hơn TII top-4
`8.1576` điểm. Bottleneck không còn chỉ là thiếu task thắng; cách so và ghép
logit giữa các LoRA độc lập cũng là vấn đề.

## 7. Các hướng đã thử và không được lặp lại

### CFS/PMI

- CFS trực tiếp trong CRCT thường làm accuracy/loss xấu đi dù đôi lúc giảm
  Forgetting.
- CFS-only và distribution filter không vượt baseline ổn định.
- CFS-TII cải thiện vài metric nhưng Acc@5 giảm; end-to-end gate FAIL.
- PMI diagnostic xác nhận CFS target reachable và đa dạng hơn Gaussian, nhưng
  inversion về final feature cho fixed head không tạo đủ thông tin để bù chi phí.
- CFS task-logit calibration: task 2 pass synthetic gate nhưng không chuyển
  thành lợi ích test; task 3 trả về identity. So cùng checkpoint, accuracy/loss/
  Forgetting không đổi và Backward kém `0.0666`.
- `CFS_STRICT_GAIN=FAIL`, `CFS_CAUSAL_GATE=FAIL`.

Không quét learning rate, scale/bias, số synthetic feature hoặc nới CFS gate
trên test để ép kết quả pass.

### Semantic và routing khác

Đã thử semantic mean shift/top-k/covariance transfer/feature adapter/local
prototype; task-mass, conditional fusion; selective/distilled/cascade/budgeted;
Arrow/LoRA response; soft mixture/soft-hard/local refinement; plurality/Borda/
dominance. Tất cả đã fail hoặc không đáng chi phí.

Calibrated progressive dùng historical calibration/per-example feature có kết
quả tốt nhưng vi phạm strict protocol, không được dùng làm claim.

## 8. Thay đổi mới nhất: CRM GEN confidence fusion

Commit: `c02c89c`.

Lý do thử nghiệm: raw proposal so max-logit thô giữa các LoRA độc lập, có thể
lệch scale/offset. Tuy nhiên CRM gốc chỉ dùng GEN làm low-confidence gate trước
rematching và chọn candidate output bằng energy, không xem GEN là task posterior.

Cơ chế mới:

1. Lấy logits của 20 lớp thuộc task ứng với từng candidate LoRA.
2. Tạo `P(class | task, x)` bằng softmax nội task.
3. Tính GEN confidence với `gamma=0.1`, `M=20` như HRM-PET.
4. Kết hợp GEN với TII prior để tạo `P(task | x)`.
5. Ghép thành `P(task | x) * P(class | task, x)` trên 200 lớp.
6. Task route lấy từ task chứa joint class score lớn nhất để `Acc@task` khớp
   với lớp thắng.

Tính chất:

- Không học thêm tham số, không đọc/lưu ảnh cũ hay feature từng mẫu.
- Không dùng synthetic calibration.
- Bất biến với offset logit riêng của từng adapter.
- Vẫn 5 LoRA/mẫu và 2 forward calls.
- Guard không cho bật chồng task-mass, conditional hoặc CFS calibration.
- Script so nhân quả với raw proposal cùng checkpoint và cùng chi phí.

File thay đổi:

```text
engines/prediction_proposal_rematching.py
configs/imr_lora.py
tests/test_prediction_proposal_audit.py
training_scripts/eval_imagenet_r_prediction_proposal5_completion_4090.sh
EXPERIMENT_PROGRESS_VI.md
```

Đã pass `py_compile`, `bash -n`, `git diff --check`. Máy Windows local không có
PyTorch; script 4090 tự chạy unit test trước và tự dừng nếu test fail.

## 9. Cấu hình CRM đã chạy, không chạy lại

Cấu hình dưới đây đã được khóa trước và đã chạy; giữ để truy vết, không chạy lại:

```bash
cd ~/Documents/truongnguyen/Hybrid_ReMatching
git pull --ff-only

CRM_CONFIDENCE_FUSION=1 \
CRM_CONFIDENCE_TEMPERATURE=0.1 \
PROPOSAL_COUNT=3 \
TASK_MASS_FUSION=0 \
CONDITIONAL_FUSION=0 \
CFS_TASK_CALIBRATION=0 \
ITERATIVE_PROPOSAL=0 \
bash training_scripts/eval_imagenet_r_prediction_proposal5_completion_4090.sh \
  ~/Documents/truongnguyen/hrm-pet-output/imr_lora_rank8_baseline_10tasks_seed42
```

Log dự kiến:

```text
~/Documents/truongnguyen/hrm-pet-output/imr_lora_rank8_baseline_10tasks_seed42_eval_prediction_proposal_i2_p3_c5_tiicomplete_strict_crmgen_t0p1.log
```

Lấy kết quả:

```bash
cd ~/Documents/truongnguyen/Hybrid_ReMatching

LOG=~/Documents/truongnguyen/hrm-pet-output/imr_lora_rank8_baseline_10tasks_seed42_eval_prediction_proposal_i2_p3_c5_tiicomplete_strict_crmgen_t0p1.log

grep -E "Average accuracy till task10|wall time seconds|BASELINE_ALL_METRIC_GATE|CRM_CAUSAL_ALL_METRIC_GATE|OPERATIONAL_PROPOSAL_EFFICIENCY_GATE" "$LOG"
```

## 10. Quy tắc quyết định đã áp dụng

Chỉ coi CRM fusion là cải tiến khi cả ba đều pass:

```text
BASELINE_ALL_METRIC_GATE=PASS
CRM_CAUSAL_ALL_METRIC_GATE=PASS
OPERATIONAL_PROPOSAL_EFFICIENCY_GATE=PASS
```

`CRM_CAUSAL_ALL_METRIC_GATE` so với raw proposal 5 LoRA cùng checkpoint, cùng
candidate budget và cùng hai forward. Đây là gate quan trọng nhất.

Nếu pass: ghi metrics/delta vào `EXPERIMENT_PROGRESS_VI.md`, không tune thêm
seed 42, khóa cấu hình rồi mới chạy seed bổ sung. Báo cáo đây là cải tiến CRM,
chưa gọi là đóng góp CFS.

Nếu fail: ghi kết quả âm, không sweep confidence temperature trên test, đóng
nhánh GEN soft fusion. Ý tưởng tiếp theo phải bắt đầu bằng audit trên output sẵn
có, không train full theo giả định.

## 11. Cách làm việc với người dùng

- Chủ động tiếp tục, không dừng ở đề xuất chung chung.
- Trước thay đổi lớn phải giải thích sửa gì và vì sao.
- Không chạy hai experiment cùng lúc trên RTX 4090.
- Lệnh phải hiện log trực tiếp và tự kết thúc; không `nohup` nếu chưa được yêu cầu.
- Sau khi code ổn: tự commit/push; lệnh 4090 luôn kèm `git pull --ff-only`.
- Không hy sinh rehearsal-free để lấy con số đẹp.

## 12. File nên đọc ở chat mới

1. `CHAT_HANDOFF_VI.md` - file này.
2. `EXPERIMENT_PROGRESS_VI.md` - nhật ký đầy đủ.
3. `CFS_PMI_RESEARCH_RESET_VI.md` - chẩn đoán CFS/PMI.
4. `engines/prediction_proposal_rematching.py`.
5. `training_scripts/eval_imagenet_r_prediction_proposal5_completion_4090.sh`.
6. `protocols.py` và `tests/test_exemplar_free_protocol.py`.

## 13. Câu mở đầu cho chat mới

```text
Đọc CHAT_HANDOFF_VI.md trong repo. Tiếp tục đúng trạng thái bàn giao, không lặp
lại các nhánh đã fail và không vi phạm strict rehearsal-free. CRM GEN fusion đã
fail cả baseline và causal gate; không sweep tham số. Bước đang chờ chỉ là audit
taskwise trên log CRM và raw proposal cùng budget trước khi thiết kế ý tưởng mới.
```

## 14. Kết quả CRM GEN confidence fusion và trạng thái mới

Cấu hình khóa trước đạt Acc@task `79.5195`, Acc@1 `74.2104`, Acc@5 `81.4432`,
Loss `2.8989`, Forgetting `3.5364` và Backward `-3.4868`, với 5 LoRA/mẫu và
2 forward calls/mẫu.

So với conventional, chỉ Acc@task (`+1.7281`) và Acc@1 (`+0.1913`) tốt hơn;
Acc@5 giảm `5.4461`, Loss tăng `1.6684`, Forgetting tăng `0.2563` và Backward
giảm `0.5749`. So với raw proposal cùng budget, CRM kém hơn cả sáu metric:
Acc@task `-1.2585`, Acc@1 `-0.8746`, Acc@5 `-5.8700`, Loss `+1.6902`,
Forgetting `+0.2242`, Backward `-0.4256`.

```text
BASELINE_ALL_METRIC_GATE=FAIL
CRM_CAUSAL_ALL_METRIC_GATE=FAIL
OPERATIONAL_PROPOSAL_EFFICIENCY_GATE=PASS
```

Quyết định: đóng nhánh CRM GEN soft fusion; không sweep temperature, prior,
gamma hoặc top-M trên test seed 42. Cả audit taskwise lẫn stagewise đã hoàn tất;
không còn run hoặc audit CRM nào đang chờ.

## 15. Kết quả taskwise

So với raw proposal cùng budget, CRM làm initial Acc@1 trung bình task 1-9 giảm
`0.4181`, peak giảm `0.6195`, final giảm `0.8437`, Forgetting tăng
`0.2242` và Backward giảm `0.4256`.

Task 8 và 7 giảm final mạnh nhất (`-3.1461`, `-2.5135`) và đã có peak thấp
hơn từ trước; task 3 giảm `-1.6393` chủ yếu vì Forgetting tăng `+1.4051`.
CRM chỉ tăng final ở task 1-2, giảm bảy task cũ còn lại và giảm task 10
`-1.1527`. Đây là lỗi rộng trên cả initial classification và retention, không
phải một lỗi muộn cô lập; không tạo task-specific selector từ kết quả test.

Script taskwise đã báo stagewise Acc@1, Acc@5, Loss và các ô suy giảm mạnh nhất.
Kết quả cuối nằm ở mục 16; không chạy lại. Chẩn đoán CRM đã đóng và chỉ giữ raw
proposal budget-5/budget-6 như các Pareto reference.

## 16. Kết quả stagewise cuối và đóng CRM

Stage 1 giống raw proposal. Ngay stage 2, khi cả hai task đều được đánh giá đầy
đủ, Acc@1 tăng `0.5128` nhưng Acc@5 giảm `4.1282` và Loss tăng `0.6995`.
Do đó lỗi calibration xuất hiện từ multi-task stage đầu tiên, không phụ thuộc
việc proposal có bỏ sót task hay budget đã đạt 5 LoRA.

Acc@5 giảm ở mọi stage 2-10; Loss tăng ở mọi stage và delta tăng đến `+1.6903`.
Acc@1 chỉ tăng nhẹ ở stage 2-4, bắt đầu giảm từ stage 5 và đạt `-0.8746` tại
stage 10. Tệ nhất là stage 10/task 10 với Acc@5 `-10.2306`, Loss `+2.5069`;
task 6 cũng liên tục mất hơn 8 điểm Acc@5 tại nhiều stage.

Đọc lại implementation cho thấy CRM gốc dùng GEN làm low-confidence gate, rồi
chọn candidate output bằng energy. GEN không phải task posterior. CRM fusion đã
xóa absolute task-fit của raw logits khi chuẩn hóa trong từng task, khiến LoRA
sai có thể trở nên tự tin giả tạo.

Quyết định cuối: giữ CRM fusion như negative ablation và không chạy thêm bất kỳ
audit, temperature, prior, gamma, top-M, selector hay task-specific fix nào.
Không có thí nghiệm CRM đang chờ. Raw proposal budget-5 và budget-6 là các
Pareto reference strict hiện tại.
## 17. Huong dang cho: prediction-closure audit

CRM da dong hoan toan. Huong moi da preregister nhung chua co ket qua GPU:
prediction-closure bat dau TII top-2, sau moi wave them tat ca task so huu top-5
class predictions cua cac LoRA moi, va dung khi tap ung vien khong doi. No khac
iterative 2->2->2 da fail o cho khong co budget/wave co dinh va dung bang dieu
kien diem co dinh roi rac.

Cau hinh khoa: strict rank-8 seed 42, initial 2, top-class 5, prior 0.3,
temperature 1.0. Gate: winner recall va exact agreement >=99.5%, top-5 coverage
>=99%, LoRA/sample <=7, calls/sample <=3, full-scan rate <=20%. Khong sweep neu
fail. Chua trien khai operational router truoc khi gate audit pass.

Lenh can chay tren RTX 4090 sau khi pull:

~~~bash
cd ~/Documents/truongnguyen/Hybrid_ReMatching
git pull --ff-only
bash training_scripts/eval_imagenet_r_prediction_closure_audit_4090.sh \
  ~/Documents/truongnguyen/hrm-pet-output/imr_lora_rank8_baseline_10tasks_seed42
~~~

Khong chay song song voi job GPU khac. Gui lai dong task10,
PREDICTION_CLOSURE_GATE va wall time.
## 18. Closure da fail; dang cho closure + TII tail

Ket qua prediction-closure: winner/exact 99.7366%, top-5 coverage 85.3245%,
full-scan 0.8615%, LoRA/sample 5.5593, calls/sample 2.5995, wall time 408 giay.
PREDICTION_CLOSURE_GATE=FAIL duy nhat vi top-5 coverage. Closure thuan da dong;
khong sweep.

Nhanh moi da preregister: giu nguyen closure i2/c5 va dung TII probability de
dien tail cho class thuoc task chua evaluate. Phep nay top-1 safe, khong them
LoRA/call va khong co hyperparameter moi. Gate: ca sau quality metric phai hon
conventional strict; winner/exact >=99.5%; LoRA <=7; calls <=3. Acc@5 output
duoc do truc tiep, khong dung closure Top5Coverage lam gate cho nhanh moi.

Lenh RTX 4090:

~~~bash
cd ~/Documents/truongnguyen/Hybrid_ReMatching
git pull --ff-only
bash training_scripts/eval_imagenet_r_prediction_closure_tii_tail_4090.sh \
  ~/Documents/truongnguyen/hrm-pet-output/imr_lora_rank8_baseline_10tasks_seed42
~~~

Gui lai dong task10, bang comparison, CLOSURE_TII_TAIL_ALL_METRIC_GATE va wall
time. Khong chay song song voi job GPU khac.
