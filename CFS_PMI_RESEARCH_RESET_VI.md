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
## 11. Kết quả diagnostic v2

Diagnostic v2 trên checkpoint task 1 trả về `PASS` và có positive control hợp lệ:

| Chỉ số | Real control | Gaussian | CFS |
|---|---:|---:|---:|
| Target cosine | 0.9983 | 0.9938 | 0.9930 |
| Class accuracy | 1.0000 | 1.0000 | 1.0000 |
| Class confidence | 0.9895 | 0.9953 | 0.9936 |
| Output pairwise cosine | 0.3155 | 0.3992 | 0.3732 |
| Nearest-real cosine distance | 0.0017 | 0.4720 | 0.4845 |

Kết luận đúng phạm vi:

- inversion đã hội tụ và tái tạo được cả feature thật lẫn feature tổng hợp;
- CFS giữ đúng lớp và làm các mẫu tổng hợp đa dạng hơn Gaussian trung bình khoảng 0.026 pairwise cosine;
- chưa có bằng chứng CFS cải thiện continual learning;
- CFS không gần manifold thật hơn Gaussian ở mức tổng hợp. Riêng lớp 2 và 3, khoảng cách output tăng lần lượt khoảng 0.041 và 0.028. Vì vậy không được chạy full chỉ dựa trên diagnostic này.

## 12. Vì sao chưa đưa PMI inversion vào CRCT LoRA

CRCT hiện tại huấn luyện classifier trực tiếp trên final feature; backbone và các LoRA cũ không thay đổi. Diagnostic cho thấy PMI inversion ánh xạ patch-token trở lại gần như đúng final feature mục tiêu với cosine khoảng 0.993. Sau đó đưa feature đó vào cùng classifier gần như lặp lại dữ liệu final-feature mà CRCT đã có, nhưng tốn thêm hàng trăm bước tối ưu.

PMI inversion có ý nghĩa hơn khi student backbone thay đổi và cần teacher truyền tri thức qua input tổng hợp. Điều kiện đó không đúng với head-only CRCT hiện tại. Vì vậy chưa triển khai synthetic KD theo hướng này để tránh thêm chi phí mà không có cơ chế tạo lợi ích rõ ràng.

## 13. Pilot tiếp theo: CFS cho TII

Exhaustive rematching cho thấy lỗi chọn task/LoRA là nút thắt lớn trên ImageNet-R. Trong khi đó các thí nghiệm ImageNet-R trước đều dùng lại TII gốc; CFS chưa được đánh giá trực tiếp tại bộ suy luận task.

Pilot mới chèn CFS vào classifier correction của TII:

1. giữ nguyên ViT-B/16, seed, 20 epoch mỗi task, covariance đầy đủ, learning rate và 30 epoch correction của baseline;
2. chỉ bật CFS paper-style với đúng feature của từng lớp;
3. không semantic, không boundary replay, không ảnh cũ và không per-example feature memory;
4. dừng sau 3 task nhưng vẫn giữ partition 10 task;
5. chỉ cho phép chạy full khi Acc@1, Acc@5, Loss, Forgetting và Backward đều không xấu hơn baseline task 3.

Đây là ablation của thành phần CFS, không phải triển khai toàn bộ PMI-CFS. Nếu gate thất bại, phải kết luận CFS chưa cải thiện TII trên ImageNet-R và dừng nhánh này.
## 14. Pilot TII-CFS paper-style tỷ lệ 0.5

Kết quả sau 3 task:

| Chỉ số | TII gốc | TII + CFS | Thay đổi | Đánh giá |
|---|---:|---:|---:|---|
| Acc@1 | 61.8101 | 62.3907 | +0.5806 | tốt hơn |
| Acc@5 | 84.2751 | 83.5555 | -0.7196 | xấu hơn |
| Loss | 1.5935 | 1.5723 | -0.0212 | tốt hơn |
| Forgetting | 7.4051 | 6.6974 | -0.7077 | tốt hơn |
| Backward | -7.4051 | -6.6974 | +0.7077 | tốt hơn |

Strict gate thất bại vì Acc@5 giảm. Không chạy full 10 task với tỷ lệ 0.5.

CFS đã cải thiện top-1, độ chắc chắn trung bình và khả năng giữ task cũ, nên cơ chế có tín hiệu tích cực tại TII. Tuy nhiên việc chọn 50% replay theo diversity làm giảm độ phủ các lớp cạnh tranh trong top 5. Ablation kế tiếp chỉ giảm `cfs_selection_ratio` từ 0.5 xuống 0.25, nghĩa là giữ 75% mẫu Gaussian và dùng 25% mẫu CFS. Mọi tham số khác giữ nguyên. Mục tiêu là giữ lợi ích Acc@1/forgetting nhưng phục hồi Acc@5; đây là kiểm tra nội suy có cơ sở, không phải thêm cơ chế mới.
## 15. Pilot TII-CFS tỷ lệ 0.25

Giảm CFS từ 50% xuống 25% không phục hồi Acc@5:

| Chỉ số | TII gốc | CFS 0.25 | Thay đổi |
|---|---:|---:|---:|
| Acc@1 | 61.8101 | 62.8631 | +1.0530 |
| Acc@5 | 84.2751 | 83.2791 | -0.9960 |
| Loss | 1.5935 | 1.5700 | -0.0235 |
| Forgetting | 7.4051 | 6.8410 | -0.5641 |
| Backward | -7.4051 | -6.8410 | +0.5641 |

Kết quả vẫn FAIL strict gate. Không tiếp tục tuning tỷ lệ vì xu hướng Acc@5 giảm đã lặp lại ở cả 0.5 và 0.25.

## 16. Sửa thiếu sót đánh giá TII

TII được dùng để chọn task/LoRA bằng quy trình: lấy lớp top-1 của TII rồi ánh xạ lớp đó sang task. Tuy nhiên engine TII trước đây nhận `target_task_map` nhưng không đo `Acc@task`; các pilot chỉ báo class Acc@1/Acc@5.

Đã bổ sung eval-only trên checkpoint có sẵn:

- `Acc@task` được tính đúng bằng cùng ánh xạ class-to-task mà LoRA engine dùng;
- baseline và candidate đều được đánh giá lại từ checkpoint task 1 đến task 3;
- không train lại, không thay checkpoint và không lưu dữ liệu;
- `TII_ROUTING_GATE` đo đúng mục tiêu vận hành của TII;
- `STRICT_ALL_METRIC_GATE` vẫn bao gồm Acc@5 để công khai hạn chế.

Nếu Acc@task không tốt hơn baseline thì dừng CFS tại TII. Nếu Acc@task tốt hơn nhưng strict gate vẫn fail, chỉ được xem là tín hiệu routing và phải kiểm chứng downstream; không được tuyên bố cải thiện toàn diện.
