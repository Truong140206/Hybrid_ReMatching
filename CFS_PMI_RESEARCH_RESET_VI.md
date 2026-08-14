# Reset nghiên cứu CFS-PMI cho HRM-PET

## 1. Vì sao phải dừng cách thử cũ

Các pilot gần nhất cho thấy CFS đưa trực tiếp vào CRCT làm giảm forgetting nhưng đồng thời làm giảm rõ accuracy và tăng loss. Ví dụ task 3:

| Chỉ số | Baseline | CFS + hybrid gate | Thay đổi |
|---|---:|---:|---:|
| Acc@task | 89.1321 | 87.8586 | -1.2735 |
| Acc@1 | 80.9782 | 79.9697 | -1.0085 |
| Acc@5 | 93.7122 | 93.0928 | -0.6194 |
| Loss | 0.8372 | 0.8802 | +0.0430 |
| Forgetting | 2.8462 | 2.2872 | -0.5590 |
| Backward | -2.5128 | -2.2872 | +0.2256 |

Đây không phải cải tiến đa mục tiêu. Nó chỉ đổi accuracy lấy forgetting.

## 2. Sai lệch đã xác định

Trong PMI-CFS gốc, CFS không chọn feature Gaussian rồi đưa thẳng vào classifier bằng hard cross-entropy.

Pipeline gốc là:

1. CFS chọn các feature mục tiêu đa dạng.
2. Partial model inversion tìm input hoặc biểu diễn trung gian mà mô hình cũ thật sự ánh xạ tới các feature đó.
3. Mô hình cũ làm teacher trên dữ liệu tổng hợp.
4. Mô hình mới học bằng knowledge distillation.

Bản tích hợp trước đã bỏ qua bước 2 và 3, rồi dùng feature Gaussian/CFS làm dữ liệu hard-label cho CRCT. Vì vậy classifier có thể bị kéo bởi các điểm không nằm trên manifold mà backbone/LoRA thực sự tạo ra.

## 3. Quyết định về semantic projection

Semantic projection của paper PMI giải quyết trường hợp không có dữ liệu thật của lớp mới. HRM-PET trên ImageNet-R vẫn có dữ liệu thật của task hiện tại.

Do đó:

- không tiếp tục tuning semantic projection trong pipeline chính;
- không tuyên bố semantic là phần đóng góp nếu không có ablation độc lập chứng minh lợi ích;
- ưu tiên current-task real CE và old-task synthetic KD, đúng vai trò dữ liệu của hai nhánh.

## 4. Giả thuyết còn đáng kiểm chứng

CFS chỉ có thể hữu ích cho HRM-PET nếu feature do CFS chọn:

1. tái tạo được qua LoRA cũ từ biên patch-token;
2. giữ đúng lớp và confidence;
3. không xa manifold feature thật hơn Gaussian thường;
4. đa dạng hơn Gaussian thường sau khi đi qua chính mô hình cũ.

Nếu không đồng thời đạt bốn điều trên, dừng CFS cho HRM-PET thay vì tiếp tục đổi hyperparameter.

## 5. Chẩn đoán mới

File `engines/cfs_pmi_diagnostic.py` thực hiện so sánh có đối chứng trên checkpoint task 1:

- cùng lớp, cùng số target và cùng ngân sách inversion;
- nhánh A dùng Gaussian target;
- nhánh B dùng paper-style CFS target;
- cả hai được partial-invert ở biên patch-token;
- chỉ tensor tổng hợp được tối ưu, toàn bộ model bị đóng băng;
- không train classifier, không thay checkpoint và không lưu ảnh cũ.

Các chỉ số:

- `target_cosine`: feature tái tạo có đạt đúng target không;
- `class_accuracy`, `class_confidence`: giữ đúng lớp không;
- `nearest_real_cosine_distance`: khoảng cách tới manifold feature thật;
- `output_pairwise_cosine`: độ giống nhau giữa các mẫu, thấp hơn nghĩa là đa dạng hơn.

Điều kiện PASS mặc định:

- target cosine của CFS ít nhất 0.90;
- accuracy và confidence không kém Gaussian quá 0.02;
- khoảng cách manifold không xấu hơn Gaussian quá 0.02;
- pairwise cosine thấp hơn Gaussian ít nhất 0.005.

## 6. Nếu chẩn đoán PASS

Chỉ khi PASS mới triển khai pilot huấn luyện 3 task:

- current-task ảnh thật: cross-entropy;
- old-task patch-token tổng hợp: soft KD từ old LoRA/old shared head;
- shared classifier là phần được bảo vệ chính;
- Gaussian-PMI và CFS-PMI phải chạy cùng budget để đo đóng góp riêng của CFS;
- strict gate vẫn yêu cầu Acc@task, Acc@1, Acc@5, Loss, Forgetting và Backward đều không xấu hơn baseline.

## 7. Nếu chẩn đoán FAIL

Dừng nhánh CFS-PMI cho HRM-PET. Không chạy full 10 task và không tăng số epoch để che lỗi cơ chế.

Khi đó hướng nghiên cứu chính phải quay về đúng bottleneck của HRM-PET: task matching/routing, với đánh giá đồng thời accuracy, forgetting và chi phí số LoRA forward trên mỗi mẫu.

## 8. Tính đúng giao thức

Chẩn đoán dùng dữ liệu task 1 ngay tại thời điểm đánh giá checkpoint task 1 và không lưu dữ liệu đó vào checkpoint.

Nếu bước huấn luyện synthetic KD được triển khai sau này, cần mô tả chính xác là exemplar-free/data-free synthetic replay. Không được gọi là strict replay-free nếu giao thức hoặc bài báo mục tiêu định nghĩa mọi dạng synthetic replay là replay.


## 9. Nguồn đối chiếu

- HRM-PET chính thức, commit `f26a844`: https://github.com/wei-cheng777/HRM-PET
- PMI-CFS-DFCL chính thức, commit `9716272`: https://github.com/RuilinTong/PMI-CFS-DFCL
- Cấu hình ImageNet-R của PMI-CFS dùng `memory_per_class: 5`, `start_block: 1` và KD.
- Code inversion chính thức tối ưu target feature theo từng block rồi full-tune; code huấn luyện dùng output của old model làm teacher.


## 10. Kết quả diagnostic v1 và sửa thiết kế

Diagnostic v1 chạy trong 8 giây và trả FAIL:

- Gaussian target cosine: 0.6618;
- CFS target cosine: 0.6400;
- Gaussian/CFS đều đạt class accuracy 1.0;
- CFS output pairwise cosine giảm từ 0.7501 xuống 0.7248;
- khoảng cách output tới real manifold lần lượt là 0.3097 và 0.3209.

Không được diễn giải kết quả này là CFS thất bại. Cả Gaussian lẫn CFS đều không đạt ngưỡng reachability 0.90, nên năng lực inversion chưa được xác nhận. Diagnostic v1 thiếu positive control và ngân sách 20 + 40 bước quá thấp so với code nguồn.

Diagnostic v2 sửa như sau:

- thêm `real_control`: feature thật phải được inversion lại thành công trước;
- nếu real control không đạt thì trạng thái là `INCONCLUSIVE`;
- chỉ so sánh CFS với Gaussian sau khi inversion được xác nhận hợp lệ;
- thêm khoảng cách từ chính target tới real manifold;
- dùng split block 1 theo cấu hình ImageNet-R của PMI-CFS;
- tăng mặc định lên 200 epoch CFS, 100 bước layer-wise và 300 bước full;
- tăng screening lên 4 lớp và 5 target mỗi lớp.

Vì vậy kết quả v1 chỉ là bằng chứng rằng diagnostic cũ thiếu năng lực hội tụ, không phải bằng chứng bác bỏ CFS.
