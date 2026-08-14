# Tiến độ thực nghiệm ImageNet-R

## 1. Thiết lập chung

- Dataset: Split-ImageNet-R, 200 lớp, 10 task, 20 lớp/task.
- Backbone: ViT-B/16.
- TII dùng chung: `imr_tii_original_10tasks_seed42`.
- Baseline và cải tiến trong mỗi lần chạy dùng cùng TII và cùng split dữ liệu.
- Cấu hình cải tiến được giữ lại:
  - legacy CTIRD;
  - CFS sampling;
  - full replay;
  - boundary replay ratio `0.10`;
  - CRCT `4` epoch;
  - CRCT learning rate `0.005`;
  - task-energy và semantic đều tắt.

Lưu ý: seed 0 hiện chỉ thay seed của giai đoạn LoRA/CRCT, còn TII vẫn cố định ở seed 42. Vì vậy đây là lặp lại giai đoạn 2 dưới TII cố định, chưa phải hai seed độc lập hoàn toàn của toàn pipeline.

## 2. Kết quả từng seed

| Seed LoRA/CRCT | Phương pháp | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | HRM-PET baseline | 77.3007 | 73.8379 | 86.0767 | 1.2399 | 3.5268 | -3.1815 |
| 42 | CFS + full replay + boundary 0.10 + CRCT4 | 77.7378 | 73.8498 | 86.2814 | 1.2488 | 3.3421 | -3.0883 |
| 0 | HRM-PET baseline | 78.3613 | 73.6062 | 86.4495 | 1.2267 | 3.4859 | -3.3020 |
| 0 | CFS + full replay + boundary 0.10 + CRCT4 | 77.9987 | 74.1049 | 86.3983 | 1.2489 | 3.3086 | -3.1867 |

## 3. Chênh lệch cải tiến trừ baseline

| Seed | Delta Acc@task | Delta Acc@1 | Delta Acc@5 | Delta Loss | Delta Forgetting | Delta Backward |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | +0.4371 | +0.0119 | +0.2047 | +0.0089 | -0.1847 | +0.0932 |
| 0 | -0.3626 | +0.4987 | -0.0512 | +0.0222 | -0.1773 | +0.1153 |

Trong cả hai seed, Acc@1 tăng, Forgetting giảm và Backward gần 0 hơn. Acc@task và Acc@5 dao động theo seed. Loss tăng nhẹ.

## 4. Trung bình hai seed

| Phương pháp | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---|---:|---:|---:|---:|---:|---:|
| HRM-PET baseline | 77.8310 | 73.7221 | 86.2631 | 1.2333 | 3.5064 | -3.2418 |
| Bản cải tiến | 77.8683 | 73.9774 | 86.3399 | 1.2489 | 3.3254 | -3.1375 |
| Cải tiến - baseline | +0.0373 | +0.2553 | +0.0768 | +0.0156 | -0.1810 | +0.1043 |

Kết luận tạm thời trên hai lần chạy giai đoạn 2:

- Acc@1 trung bình tăng `0.2553` điểm.
- Acc@task trung bình tăng `0.0373` điểm.
- Acc@5 trung bình tăng `0.0768` điểm.
- Forgetting trung bình giảm `0.1810` điểm, khoảng `5.16%` so với baseline.
- Backward tăng `0.1043` điểm, tức gần 0 hơn.
- Loss tăng `0.0156`; đây là chỉ số duy nhất đi theo hướng chưa tốt.

## 5. Các ablation quan trọng đã thực hiện

| Cấu hình | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward | Kết luận |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline seed42 | 77.3007 | 73.8379 | 86.0767 | 1.2399 | 3.5268 | -3.1815 | Mốc chính |
| CFS + task-energy + full replay CRCT30 | 77.3976 | 73.7220 | 86.2722 | 1.3896 | 4.2201 | -4.2201 | Correction quá mạnh; forgetting tăng |
| CFS + legacy CTIRD + full replay CRCT30 | 76.9255 | 73.1216 | 85.9195 | 1.4265 | 4.6460 | -4.5961 | Full replay 30 epoch không phù hợp |
| CFS, ngân sách batch như baseline | 77.0826 | 73.5323 | 85.8652 | 1.2502 | 3.3270 | -3.1276 | Retention tốt hơn, accuracy giảm nhẹ |
| CFS + full replay + boundary 0.10 + CRCT3 | 77.3035 | 73.7452 | 86.1310 | 1.2604 | 3.2234 | -3.0588 | Cân bằng tốt, Acc@1 còn thấp hơn baseline |
| CFS + full replay + boundary 0.10 + CRCT4 | 77.7378 | 73.8498 | 86.2814 | 1.2488 | 3.3421 | -3.0883 | Cấu hình tốt nhất seed42 |

## 6. Giải thích thay đổi CRCT

Với batch size 24, code sinh `24 x 5 = 120` pseudo-feature cho mỗi centroid. ImageNet-R dùng multi-centroid với tối đa 10 centroid/lớp. Khi có 200 lớp, full replay có thể tạo khoảng 2.000 batch correction mỗi epoch, trong khi baseline task 10 chỉ dùng khoảng 180 batch. Vì vậy full replay với 30 CRCT epoch tạo quá nhiều bước cập nhật và làm classifier bị over-correction.

Giảm CRCT xuống 3-4 epoch đưa tổng số update về gần ngân sách baseline. CRCT4 lấy lại accuracy tốt hơn CRCT3, trong khi Forgetting vẫn thấp hơn baseline.

## 7. Việc cần làm tiếp

1. Chạy thêm seed LoRA/CRCT dưới cùng TII để kiểm tra độ ổn định.
2. Nếu cần kết quả chuẩn publication, huấn luyện lại toàn bộ TII + LoRA/CRCT cho từng seed độc lập.
3. Báo cáo mean và độ lệch chuẩn khi có ít nhất 3 seed.
4. Không tiếp tục tối ưu riêng seed 42 để tránh chọn siêu tham số theo một lần chạy.

## 8. Kết quả CTIRD online căn chỉnh đúng batch

Mục tiêu của ablation này là sửa hai điểm của CTIRD cũ:

1. Feature của teacher LoRA cũ được tính trực tiếp trên đúng batch ảnh đang dùng cho student, thay vì lấy relation theo chỉ số batch từ cache.
2. Các nguồn teacher được chọn theo tổng xác suất của từng task cũ, tránh chọn trùng nhiều lớp thuộc cùng một task.

CFS, semantic, prototype routing và exhaustive routing đều tắt để chỉ đo tác động của CTIRD.

| Cấu hình rank 8, seed 42 | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 77.7914 | 74.0191 | 86.8893 | 1.2305 | 3.2801 | -2.9119 |
| CTIRD online aligned, cộng K loss | 78.0159 | 74.2252 | 86.6528 | 1.2367 | 3.6520 | -3.4288 |
| Chênh lệch | +0.2245 | +0.2061 | -0.2365 | +0.0062 | +0.3719 | -0.5169 |

Kết luận:

- Căn chỉnh đúng batch và chọn đúng task có tín hiệu tích cực cho Acc@task và Acc@1.
- Tuy nhiên, cấu hình cộng K loss làm Acc@5, Loss, Forgetting và Backward xấu đi.
- Pilot 3 task từng cải thiện cả sáu chỉ số, nhưng kết quả full 10 task không giữ được xu hướng đó.
- Quan sát theo từng task cho thấy vấn đề xuất hiện rõ từ khoảng task 5, đúng lúc số teacher cũ được chọn tăng lên.

Nguyên nhân trong triển khai cũ: mỗi batch chỉ chạy một teacher để tiết kiệm chi phí, sau đó nhân loss với số task cũ được chọn. Vì vậy tổng lực CTIRD tăng dần từ 1 lên tối đa 5 khi quá trình continual learning tiến về các task cuối. Điều này giữ relation quá mạnh và cản trở học feature mới, làm forgetting toàn chuỗi tăng.

## 9. Ablation tiếp theo: CTIRD mean reduction

Bản mới giữ nguyên căn chỉnh batch và cách chọn teacher, nhưng thay cách tổng hợp:

- sum: tổng loss của K teacher, tương đương hành vi vừa chạy;
- mean: lấy trung bình loss của K teacher, giữ tổng cường độ CTIRD không đổi khi K tăng.

Với một rank teacher được chạy luân phiên mỗi batch:

- chế độ sum dùng trọng số K;
- chế độ mean dùng trọng số 1.

Nếu chạy R rank trên mỗi batch:

- chế độ sum dùng trọng số K/R cho mỗi rank;
- chế độ mean dùng trọng số 1/R cho mỗi rank.

Đây là ablation một biến: CFS và semantic vẫn tắt. Mục tiêu là giữ phần tăng Acc@1 do alignment mang lại nhưng loại bỏ hiện tượng regularization mạnh dần ở các task cuối. Cấu hình mới được chạy bằng tùy chọn ctird_online_reduction=mean.

## 10. Pilot CTIRD mean và phép thử ghép CFS

Kết quả sau 3 task:

| Cấu hình | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---|---:|---:|---:|---:|---:|---:|
| Baseline rank 8 | 89.1321 | 80.9782 | 93.7122 | 0.8372 | 2.8462 | -2.5128 |
| CTIRD aligned mean | 89.2370 | 81.6272 | 93.7971 | 0.8184 | 2.6923 | -2.6923 |
| Chênh lệch | +0.1049 | +0.6490 | +0.0849 | -0.0188 | -0.1539 | -0.1795 |

CTIRD mean cải thiện 5/6 chỉ số. Backward giảm 0.1795 điểm nên ảnh hưởng trung bình lên task cũ vẫn xấu hơn baseline, dù chỉ số Forgetting tính theo độ giảm từ đỉnh tốt nhất lại thấp hơn. Hai chỉ số dùng mốc tham chiếu khác nhau nên không bắt buộc biến thiên ngược dấu tuyệt đối.

Phép thử tiếp theo ghép CFS ở đúng giai đoạn classifier correction:

- giữ CTIRD aligned mean trong giai đoạn học LoRA;
- bật CFS paper-style để chọn pseudo-feature đa dạng;
- giữ nguyên số mẫu và số bước CRCT như cấu hình mean-only;
- không bật boundary replay, full replay, semantic, prototype hay exhaustive routing;
- so sánh trực tiếp với log CTIRD mean cùng 3 task, không so với một cấu hình khác ngân sách.

Các cờ CFS được dùng:

- cfs_sampling;
- cfs_epochs=20;
- cfs_train_max_samples=1024;
- cfs_candidate_multiplier=3;
- cfs_paper_style;
- cfs_selection_ratio=0.5;
- cfs_selection_steps=5.

## 11. Kết quả pilot CTIRD mean + CFS

Kết quả sau 3 task:

| Cấu hình | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---|---:|---:|---:|---:|---:|---:|
| CTIRD aligned mean | 89.2370 | 81.6272 | 93.7971 | 0.8184 | 2.6923 | -2.6923 |
| CTIRD aligned mean + CFS | 89.0005 | 80.8871 | 93.7612 | 0.8346 | 3.0821 | -3.0821 |
| CFS trừ mean-only | -0.2365 | -0.7401 | -0.0359 | +0.0162 | +0.3898 | -0.3898 |

CFS làm xấu cả sáu chỉ số so với CTIRD mean-only. So với baseline rank 8, bản ghép chỉ tăng rất nhẹ Acc@5 0.0490 và giảm Loss 0.0026, nhưng:

- Acc@task giảm 0.1316;
- Acc@1 giảm 0.0911;
- Forgetting tăng 0.2359;
- Backward giảm 0.5693.

Kết luận:

1. Không chạy full cấu hình CTIRD mean + CFS này.
2. CFS không khắc phục phần Backward còn yếu của CTIRD mean trong setting hiện tại.
3. Đây chưa chứng minh CFS và CTIRD xung đột về lý thuyết. Nó chứng minh cách ghép CFS paper-style trực tiếp vào CRCT hiện tại không phù hợp trên ImageNet-R.
4. CFS tối ưu độ đa dạng trong embedding của CFS nhưng không buộc candidate phải được classifier nhận đúng lớp đích. Candidate đa dạng có thể nằm ở đuôi Gaussian hoặc phía sai của decision boundary; CRCT sau đó học chúng với nhãn lớp đích và làm classifier lệch.
5. Không thêm distribution filter hoặc boundary/full replay ngay: các ablation trước đã cho thấy những cơ chế này tạo trade-off và không giải quyết đồng thời mọi chỉ số.

Quyết định thực nghiệm:

- giữ CTIRD aligned mean là ứng viên hiện tại;
- coi CFS paper-style là ablation âm trên ImageNet-R;
- chỉ thử lại CFS nếu có thay đổi về nguyên lý chọn mẫu, ví dụ ràng buộc class-consistency bằng classifier hoặc dùng feature thật để xác thực, thay vì chỉ đổi hệ số CFS.

## 12. Kết quả CTIRD mean task 5 và sparse semantic CTIRD

Kết quả CTIRD aligned mean sau 5 task:

| Cấu hình | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---|---:|---:|---:|---:|---:|---:|
| Baseline rank 8 | 83.9900 | 78.3623 | 91.4095 | 0.9677 | 2.4002 | -2.2335 |
| CTIRD aligned mean K=5 | 83.9156 | 78.7027 | 90.5810 | 0.9708 | 2.4026 | -2.4026 |
| Mean K=5 trừ baseline | -0.0744 | +0.3404 | -0.8285 | +0.0031 | +0.0024 | -0.1691 |

So với CTIRD aligned sum K=5 ở task 5, mean reduction cải thiện rõ khả năng giữ kiến thức:

- Forgetting: 3.0679 xuống 2.4026;
- Backward: -3.0346 lên -2.4026;
- Acc@1: 78.3450 lên 78.7027;
- Loss: 0.9765 xuống 0.9708.

Như vậy mean reduction đã sửa hiện tượng tổng lực CTIRD tăng theo K. Tuy nhiên, Acc@5 vẫn thấp hơn baseline 0.8285 điểm. Vấn đề còn lại là chất lượng teacher/relation chứ không chỉ cường độ loss.

Một phát hiện quan trọng: ở task 5 có 4 task cũ, trong khi K=5, nên CTIRD chọn toàn bộ 4 teacher. Xếp hạng semantic với K=5 không thể loại teacher nhiễu. Ablation tiếp theo vì vậy dùng sparse semantic CTIRD:

1. Giảm K từ 5 xuống 2.
2. TII vẫn là tín hiệu chọn teacher chính.
3. Dùng CLIP text embedding với ánh xạ synset ImageNet-R sang tên lớp thật.
4. Semantic chỉ tham gia khi chênh lệch xác suất giữa hai task TII đứng đầu nhỏ hơn margin 0.15.
5. Trọng số semantic tối đa là 0.10.
6. Chỉ semantic teacher selection được bật; semantic relation loss, semantic projection, CFS và các router đều tắt.
7. Giữ mean reduction để tổng cường độ CTIRD không tăng theo số task.

Log bổ sung hai chỉ số:

- SemWeight: trọng số semantic trung bình thực tế sau confidence gate;
- SemChange: tỷ lệ sample/batch mà semantic làm thay đổi top-K teacher so với TII thuần.

Mục tiêu là giữ mức tăng Acc@1 của CTIRD, loại các teacher ít liên quan để phục hồi Acc@5 và Backward. Đây là áp dụng ý tưởng semantic-aware vào đúng nơi phát sinh nhiễu, không dịch chuyển feature và không tăng chi phí suy luận.

## 13. Ablation chi phí: prediction-induced proposal rematching

Sau các nhánh CFS, semantic, prototype, soft-mixture và learned routing không
đạt gate, nút thắt được xác định là TII top-k bỏ sót LoRA thắng của exhaustive.
Phương pháp chốt xử lý trực tiếp nút thắt này:

1. TII chọn hai LoRA đầu tiên.
2. Hai LoRA được chạy trong một model call.
3. Top-5 lớp dự đoán của chúng được ánh xạ về task để đề xuất ba LoRA khác.
4. Ba LoRA đề xuất được chạy chung trong model call thứ hai.
5. Chỉ logits cục bộ của năm task được ghép để phân loại.
6. TII probability completion cấp xác suất cho lớp ngoài candidate và giới hạn
   khối lượng theo từng mẫu để không đảo top-1 của nhánh LoRA.

Bản thân thuật toán inference không đọc ảnh cũ, feature cũ, nhãn test hay
router học từ dữ liệu lịch sử. Tuy nhiên các số dưới đây được đo trên checkpoint
`hybrid_real_ageaware`, vốn đã dùng per-example real-feature memory khi train.
Cờ `strict_exemplar_free` lúc eval chỉ kiểm tra cơ chế inference, không thể làm
checkpoint huấn luyện đó trở thành exemplar-free. Vì vậy bảng này chỉ là
ablation routing/chi phí, chưa phải kết quả chính end-to-end.

| Cấu hình | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward | LoRA/mẫu | Calls/mẫu | Thời gian |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Feature-memory baseline | 77.5854 | 74.0477 | 86.4646 | 1.2230 | 3.3264 | -2.9319 | -- | -- | chưa đo |
| Exhaustive cùng checkpoint | 80.6549 | 75.1798 | 88.5327 | 1.0809 | 2.8848 | -2.8449 | 10 | 3* | 495 giây |
| Proposal cùng checkpoint | 80.6062 | 75.0935 | 87.5040 | 1.1801 | 2.9703 | -2.8927 | 5 | 2 | 295 giây |

`*` Ba model call là cấu hình vectorized exhaustive với task chunk size 4.

So với baseline, cả sáu chỉ số chất lượng đều tốt hơn; số LoRA được đánh giá
chỉ bằng 50% exhaustive. So với exhaustive, Acc@task giữ lại trong 0.0487 điểm
và Acc@1 trong 0.0863 điểm. Thời gian đánh giá giảm từ 495 xuống 295 giây,
tức giảm 200 giây (40.4%) và nhanh hơn 1.68 lần trên RTX 4090. Đây là bằng
chứng hiệu quả inference trên cùng checkpoint, không phải claim strict chính.

Quyết định: khóa cấu hình. Trước khi chạy nhiều seed, phải kiểm chứng trên
baseline rank-8 không lưu feature cũ; evaluator mới sẽ từ chối log huấn luyện
vi phạm giao thức. Lệnh chuẩn là
`training_scripts/validate_imagenet_r_strict_proposal_4090.sh run`; nó tạo log
strict riêng và báo cáo conventional/exhaustive/proposal cùng seed. Chỉ khi
kiểm chứng này đạt mới chuyển sang mean/std.
