# BÁO CÁO TÓM TẮT CẢI TIẾN HRM-PET

## Ứng dụng CFS và task-aware CTIRD trên Split-CIFAR100

**Trạng thái:** Tạm dừng cải tiến để tổng hợp kết quả  
**Thời điểm tổng hợp:** 06/08/2026  
**Mốc tốt nhất hiện tại:** Acc@1 = **88.60%**

## 1. Tóm tắt kết quả chính

Mục tiêu của quá trình này là cải thiện HRM-PET trong bài toán học liên tục, tập trung vào hai thành phần:

1. Nâng chất lượng feature replay dùng cho classifier correction (CRCT) bằng ý tưởng CFS.
2. Làm CTIRD chọn và sử dụng các task cũ liên quan hơn thay vì distill từ nguồn ít liên quan.

Các thay đổi có tác động rõ nhất là:

- Triển khai CFS trong feature replay của cả TII và LoRA.
- Cho CRCT sử dụng toàn bộ feature replay đã sinh thay vì chỉ dùng một phần.
- Bổ sung boundary-aware CFS để thêm một tỷ lệ nhỏ mẫu gần biên quyết định.
- Thay cách chọn nguồn CTIRD từ class logit thấp sang top-K task có task energy cao nhất.
- Gán trọng số CTIRD theo task energy nhưng vẫn giữ tổng cường độ distillation không đổi.
- Tinh chỉnh learning rate của classifier correction từ `0.005` xuống `0.0045`.

Kết quả tốt nhất trên Split-CIFAR100, 10 task, seed 42:

| Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---:|---:|---:|---:|---:|---:|
| 88.32 | **88.60** | 98.07 | **0.4607** | 4.20 | -3.9444 |

So với CFS-only được tái lập trên cùng máy và cùng checkpoint TII:

- Acc@task tăng từ `87.62` lên `88.32`: **+0.70 điểm phần trăm**.
- Acc@1 tăng từ `87.88` lên `88.60`: **+0.72 điểm phần trăm**.
- Loss giảm từ `0.5222` xuống `0.4607`: **-0.0615**.
- Acc@5 gần như giữ nguyên: `98.08` xuống `98.07`.
- Forgetting tăng từ `3.7222` lên `4.20`; đây là trade-off cần tiếp tục xử lý.

## 2. Thiết lập thí nghiệm

| Thành phần | Thiết lập |
|---|---|
| Dataset | Split-CIFAR100 |
| Số task | 10 |
| Số lớp mỗi task | 10 |
| Backbone | ViT-Base Patch16 224 |
| Phương pháp PET | HRM-PET với LoRA/Hide |
| Seed chính | 42 |
| TII epochs | 5 trong checkpoint TII đang tái sử dụng |
| LoRA epochs | 10 |
| CRCT epochs tốt nhất | 3 |
| CFS epochs | 20 |
| GPU chạy chính | NVIDIA RTX 4090 24 GB |

Các thí nghiệm sau giai đoạn tái lập đều dùng cùng checkpoint TII:

```text
cifar100_tii_cfs_10tasks_seed42
```

Việc giữ nguyên checkpoint này giúp so sánh các thay đổi ở LoRA, CTIRD và CRCT công bằng hơn.

## 3. HRM-PET ban đầu và các điểm còn hạn chế

Pipeline liên quan có thể tóm tắt như sau:

```text
Dữ liệu task hiện tại
-> huấn luyện adapter/LoRA
-> lưu thống kê feature theo lớp
-> sinh feature giả từ mean/covariance
-> classifier correction (CRCT)
-> đánh giá học liên tục
```

Ba điểm hạn chế được phát hiện:

### 3.1. Feature Gaussian chưa được chọn lọc

Bản gốc lấy mẫu từ phân phối Gaussian của từng lớp. Các mẫu hợp lệ về mặt thống kê nhưng chưa chắc đa dạng hoặc hữu ích cho classifier. Nhiều mẫu có thể gần nhau và đóng góp gradient tương tự.

### 3.2. CRCT sinh nhiều replay nhưng không dùng hết

Số vòng correction cũ dựa trên số lớp cũ (`crct_num`), trong khi số feature đã sinh lớn hơn. Vì vậy một phần replay bị bỏ qua sau khi shuffle.

### 3.3. CTIRD chọn nguồn chưa hợp lý

Logic legacy chọn các class có logit thấp rồi ánh xạ class sang task. Cách này có hai vấn đề:

- Class có logit thấp không nhất thiết thuộc task liên quan đến batch hiện tại.
- Nhiều class có thể ánh xạ về cùng một task, gây trùng nguồn distillation.

## 4. Cải tiến 1: CFS cho feature replay

### 4.1. Đã làm gì

CFS được đưa vào giai đoạn sinh feature cho CRCT ở cả hai engine:

- `engines/hide_tii_engine.py`
- `engines/hrm_lora_wtp_and_tap_engine.py`

Luồng mới:

```text
Feature thật của từng lớp
-> tính mean/covariance
-> huấn luyện MLP CFS nhỏ cho lớp đó
-> sinh nhiều Gaussian candidate
-> ánh xạ candidate qua CFS MLP
-> chọn tập candidate ít tương đồng, đa dạng hơn
-> đưa feature đã chọn vào CRCT
```

### 4.2. Khác gì so với bản gốc

| Bản gốc | Sau cải tiến CFS |
|---|---|
| Lấy ngẫu nhiên Gaussian feature | Sinh nhiều candidate rồi chọn lọc |
| Không tối ưu độ đa dạng | Ưu tiên feature ít tương đồng trong embedding CFS |
| Replay phụ thuộc hoàn toàn mean/covariance | Replay kết hợp thống kê lớp và contrastive selection |

### 4.3. Mức độ bám ý tưởng paper

Phần cốt lõi được giữ: dùng một không gian contrastive để chọn feature đại diện và đa dạng hơn.

Điểm thích nghi với HRM-PET: paper nguồn gắn CFS với model inversion, còn triển khai này áp dụng CFS trực tiếp vào feature replay của CRCT. Vì vậy đây là **chuyển giao ý tưởng**, không tuyên bố là sao chép nguyên bản toàn bộ pipeline paper.

## 5. Cải tiến 2: Full CRCT replay

### 5.1. Đã làm gì

Thêm cờ:

```text
--crct_use_all_samples
```

Khi bật, số vòng correction được tính theo tổng số feature thực tế:

```text
ceil(tổng số replay feature / batch replay)
```

thay vì chỉ dùng số lớp cũ.

### 5.2. Kết quả

| Cấu hình | Acc@task | Acc@1 | Loss | Forgetting |
|---|---:|---:|---:|---:|
| CFS-only reproduce | 87.62 | 87.88 | 0.5222 | 3.7222 |
| CFS + full CRCT | 88.01 | 88.13 | 0.4696 | 4.5222 |

Tác động:

- Acc@1 tăng `0.25`.
- Acc@task tăng `0.39`.
- Loss giảm mạnh `0.0526`.
- Lợi ích CRCT tăng rõ ở task muộn; riêng task 10, Acc@1 sau correction tăng `2.08` so với trước correction.
- Forgetting tăng, cho thấy classifier cuối tốt hơn nhưng độ giữ thành tích lịch sử chưa tối ưu.

Đây là thay đổi tạo bước nhảy lớn đầu tiên và được giữ trong cấu hình cuối.

## 6. Cải tiến 3: Boundary-aware CFS

### 6.1. Đã làm gì

Ngoài feature CFS đa dạng, một phần nhỏ replay được chọn gần biên quyết định hiện tại:

1. Sinh candidate trong phân phối lớp.
2. Loại phần Gaussian tail quá xa tâm lớp.
3. Tính margin giữa logit lớp đúng và lớp cạnh tranh mạnh nhất.
4. Chọn mẫu có trị tuyệt đối margin nhỏ, tức nằm gần decision boundary.
5. Trộn với feature CFS đa dạng.

Thiết lập tốt nhất dùng:

```text
boundary_ratio = 0.25
boundary_multiplier = 3
density_quantile = 0.9
```

### 6.2. Kết quả

| Cấu hình | Acc@task | Acc@1 | Loss | Forgetting | Backward |
|---|---:|---:|---:|---:|---:|
| Full CRCT | 88.01 | 88.13 | 0.4696 | 4.5222 | -4.4111 |
| Full CRCT + boundary 0.25 | 88.04 | 88.17 | 0.4704 | 4.4778 | -4.3444 |

Boundary-aware CFS chỉ tăng nhẹ Acc@1 `0.04`, nhưng đồng thời cải thiện forgetting và backward. Vì không phá kết quả và có cơ sở trực tiếp theo decision boundary, nó được giữ lại.

Biến thể chỉ chọn mẫu ở đúng phía biên (`target-side`) cũng đã được thử. Acc@1 giữ `88.56` nhưng không vượt bản boundary cũ, nên không dùng trong cấu hình tốt nhất.

## 7. Cải tiến 4: CTIRD chọn top-K task bằng task energy

### 7.1. Đã làm gì

Thay logic chọn nguồn CTIRD legacy bằng task-level energy:

```text
E_t(x) = T * logsumexp(logits_t(x) / T)
```

Sau đó:

- Gom logit theo từng task cũ.
- Tính energy cho mỗi task.
- Chọn K task có energy cao nhất.
- Mỗi source là một task riêng, không bị trùng do nhiều class cùng task.

Thiết lập tốt nhất:

```text
K = 5
task_temperature = 0.1
```

### 7.2. Kết quả

| Cấu hình | Acc@task | Acc@1 | Loss |
|---|---:|---:|---:|
| Full CRCT + boundary | 88.04 | 88.17 | 0.4704 |
| Thêm CTIRD task-energy K=5 | 88.05 | 88.31 | 0.4675 |

Acc@1 tăng `0.14`, lớn hơn mức tăng của các tinh chỉnh semantic hoặc boundary nhỏ. Kết quả này cho thấy chọn đúng source task là một điểm quan trọng của CTIRD.

Thử `K=3` làm Acc@1 giảm còn `88.16`; vì vậy giữ `K=5`.

## 8. Cải tiến 5: CTIRD energy weighting

### 8.1. Đã làm gì

Giữ đủ 5 source task nhưng phân bổ trọng số KL theo mức liên quan:

1. Softmax task energy của 5 source.
2. Trộn `20%` trọng số đều để không loại hẳn source hạng thấp.
3. Lấy trọng số trung bình theo batch.
4. Chuẩn hóa để tổng trọng số vẫn bằng K.

Thiết lập:

```text
weight_temperature = 1.0
weight_floor = 0.2
```

Điểm quan trọng là tổng cường độ CTIRD không tăng. Ta chỉ chuyển trọng số từ task ít liên quan sang task liên quan hơn.

### 8.2. Kết quả

| Cấu hình | Acc@task | Acc@1 | Loss | Forgetting | Backward |
|---|---:|---:|---:|---:|---:|
| Task-energy K=5 uniform | 88.05 | 88.31 | 0.4675 | 4.5111 | -4.4111 |
| Task-energy K=5 weighted | 88.30 | 88.56 | 0.4627 | 4.2778 | -4.0778 |

Đây là cải tiến hiệu quả nhất sau full CRCT:

- Acc@1 tăng `0.25`.
- Acc@task tăng `0.25`.
- Loss giảm `0.0048`.
- Forgetting giảm `0.2333`.
- Backward tốt hơn `0.3333`.

## 9. Semantic-aware feature projection

### 9.1. Các hướng đã thử

Đã triển khai và kiểm tra nhiều cách đưa semantic vào HRM-PET:

- Semantic top-k mix.
- Adaptive semantic gate cho CTIRD.
- Semantic feature projection.
- Bản paper-style dùng CLIP text embedding.
- Semantic feature adapter để căn semantic space với HRM feature space.
- Covariance transfer và các ratio nhỏ `0.03`, `0.02`, `0.015`.
- Semantic-safe filtering.

### 9.2. Kết quả và quyết định

Bản semantic cân bằng nhất đạt xấp xỉ:

| Cấu hình | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting |
|---|---:|---:|---:|---:|---:|
| CFS-only reproduce | 87.62 | 87.88 | 98.08 | 0.5222 | 3.7222 |
| Semantic covariance transfer ratio 0.02 | 87.76 | 87.86 | 97.99 | 0.5184 | 3.6667 |

Semantic giúp nhẹ ở Acc@task, loss và forgetting nhưng giảm nhẹ Acc@1/Acc@5. Các biến thể bám paper hơn cũng không vượt CFS-only ổn định.

Nguyên nhân có thể:

- Text semantic và HRM visual feature không nằm sẵn trong cùng không gian.
- CIFAR100 có nhiều lớp gần nhau về tên nhưng khác nhau về đặc trưng thị giác.
- Project feature giữa lớp dễ làm lệch covariance thật.
- Semantic prior quá mạnh sẽ làm sai relation target của CTIRD.

Quyết định cuối: **không bật semantic trong cấu hình tốt nhất**, nhưng giữ code như một ablation và hướng nghiên cứu tiếp theo. Đây là kết luận thực nghiệm, không phải semantic hoàn toàn vô ích.

## 10. Tinh chỉnh classifier correction

Sau khi cấu trúc CFS và CTIRD ổn định, learning rate của CRCT được sweep:

| ca_lr | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0035 | 88.33 | 88.46 | 98.05 | 0.4604 | 4.2222 | -3.9444 |
| 0.0040 | **88.34** | 88.56 | 98.07 | **0.4606** | **4.1889** | -3.9444 |
| 0.0045 | 88.32 | **88.60** | 98.07 | 0.4607 | 4.2000 | -3.9444 |
| 0.0050 | 88.30 | 88.56 | 98.07 | 0.4627 | 4.2778 | -4.0778 |

Có hai cấu hình có thể báo cáo:

- **Ưu tiên Acc@1:** `ca_lr=0.0045`, Acc@1 = `88.60`.
- **Ưu tiên cân bằng:** `ca_lr=0.004`, Acc@task/loss/forgetting tốt hơn rất nhẹ.

Báo cáo này chọn `0.0045` làm mốc chính vì mục tiêu cải thiện accuracy.

## 11. Tổng hợp tiến trình kết quả

| Giai đoạn | Acc@task | Acc@1 | Acc@5 | Loss |
|---|---:|---:|---:|---:|
| Baseline tham chiếu không CFS | 87.05 | 86.90 | 97.59 | 0.5860 |
| CFS-only reproduce cùng môi trường | 87.62 | 87.88 | 98.08 | 0.5222 |
| + Full CRCT replay | 88.01 | 88.13 | 98.08 | 0.4696 |
| + Boundary-aware CFS | 88.04 | 88.17 | 98.08 | 0.4704 |
| + CTIRD task-energy | 88.05 | 88.31 | 98.07 | 0.4675 |
| + CTIRD energy weighting | 88.30 | 88.56 | 98.07 | 0.4627 |
| + Tuning ca_lr=0.0045 | **88.32** | **88.60** | **98.07** | **0.4607** |

Lưu ý khi trình bày:

- So sánh đáng tin cậy nhất là từ `CFS-only reproduce` trở đi vì cùng máy, cùng checkpoint TII và cùng thiết lập chính.
- Baseline không CFS là mốc tham chiếu từ log giai đoạn đầu; không nên dùng riêng nó để tuyên bố ý nghĩa thống kê.

## 12. Các thử nghiệm không được giữ lại

| Thử nghiệm | Kết luận |
|---|---|
| Semantic top-k/gate/projection | Không vượt ổn định CFS-only |
| CFS distribution filter | Giảm accuracy |
| CFS mean initialization | Không cải thiện |
| Task-energy evaluation routing | Acc@task giảm mạnh xuống 84.91 |
| CTIRD K=3 | Yếu hơn K=5 |
| CTIRD weight temperature 0.5/1.5 | Đều kém temperature 1.0 |
| CRCT 4 epochs | Over-correction, kém 3 epochs |
| Class-balanced CRCT batches | Acc@5 tăng nhưng Acc@1 giảm |
| Target-side boundary | Gần như trung tính |
| CTIRD con=0.15/0.25 | Đều kém con=0.20 |
| Task-energy temperature=0.2 | Acc@1 giảm còn 88.28 |

Các kết quả âm này vẫn có giá trị vì giúp xác định thành phần nào thực sự đóng góp và tránh chỉ báo cáo các lần chạy thành công.

## 13. Cấu hình tốt nhất hiện tại

```text
Dataset: Split-CIFAR100
Tasks: 10
Model: vit_base_patch16_224
LoRA epochs: 10
LoRA lr: 0.03
LoRA rank: 5
LoRA momentum: 0.4
CTIRD con: 0.2
CTIRD K: 5
CTIRD source selection: task_energy
CTIRD task temperature: 0.1
CTIRD weighting: energy
CTIRD weight temperature: 1.0
CTIRD weight floor: 0.2
CFS epochs: 20
CFS candidate multiplier: 2
Full CRCT replay: bật
Boundary replay ratio: 0.25
CRCT epochs: 3
CRCT ca_lr: 0.0045
Semantic: tắt
Balanced CRCT batches: tắt
Seed: 42
```

## 14. Điểm mới so với HRM-PET ban đầu

Có thể tóm tắt đóng góp thành ba ý chính:

1. **Replay selection:** Gaussian replay được nâng thành contrastive-selected và boundary-aware replay.
2. **Replay utilization:** CRCT dùng đủ lượng replay đã sinh, đặc biệt có ích khi số task tăng.
3. **Task-aware distillation:** CTIRD chọn và gán trọng số cho source task theo task energy, thay cho lựa chọn class-level ít liên quan.

Semantic-aware projection là nhánh nghiên cứu bổ sung đã được triển khai và ablation đầy đủ, nhưng không được đưa vào cấu hình cuối vì chưa chứng minh được lợi ích về Acc@1.

## 15. Hạn chế và việc cần làm tiếp

### 15.1. Hạn chế

- Kết quả chính hiện mới có một seed (`42`), chưa đủ để khẳng định ý nghĩa thống kê.
- Thử nghiệm tập trung vào Split-CIFAR100, chưa kiểm chứng trên ImageNet-R hoặc five-datasets.
- Forgetting của cấu hình cuối vẫn cao hơn CFS-only dù final accuracy tốt hơn.
- Chưa đo riêng chi phí thời gian và bộ nhớ của từng thành phần bằng profiling thống nhất.
- Semantic feature chưa được căn chỉnh đủ tốt với visual feature space.

### 15.2. Việc nên làm khi tiếp tục

1. Chạy ít nhất 3 seed và báo cáo mean ± standard deviation.
2. Xác nhận cấu hình trên một benchmark thứ hai.
3. Tập trung giảm forgetting mà không làm mất mốc Acc@1 88.60.
4. Đo thời gian, VRAM và số tham số tăng thêm của CFS.
5. Nếu tiếp tục semantic, học alignment bằng dữ liệu train thay vì dựa chủ yếu vào class name.

## 16. Bài nói ngắn để trình bày

> Ban đầu HRM-PET sinh feature replay từ mean và covariance của từng lớp rồi dùng các feature này để correction classifier. Em nhận thấy hai vấn đề chính: feature Gaussian chưa được chọn lọc và CRCT không sử dụng hết lượng replay đã sinh. Ngoài ra, CTIRD cũ chọn source dựa trên class logit thấp nên source task có thể không liên quan hoặc bị trùng.
>
> Em đã áp dụng ý tưởng CFS vào feature replay. Với mỗi lớp, em huấn luyện một MLP contrastive nhỏ, sinh nhiều Gaussian candidate và chọn tập feature đa dạng hơn trong không gian CFS. Sau đó em sửa CRCT để dùng toàn bộ replay và bổ sung 25% mẫu gần decision boundary.
>
> Ở CTIRD, em thay cách chọn nguồn bằng task energy. Năm task cũ liên quan nhất được chọn trực tiếp, sau đó KL loss của từng source được gán trọng số theo energy nhưng tổng trọng số vẫn giữ nguyên. Đây là phần cải thiện rõ nhất sau full replay.
>
> Trên Split-CIFAR100 với 10 task và seed 42, CFS-only tái lập đạt Acc@1 87.88. Cấu hình cuối đạt 88.60, tăng 0.72 điểm phần trăm và giảm loss từ 0.5222 xuống 0.4607. Semantic-aware projection cũng đã được triển khai theo nhiều biến thể nhưng chưa vượt CFS-only ổn định nên em không đưa semantic vào cấu hình cuối.
>
> Hạn chế hiện tại là mới chạy một seed và forgetting vẫn cao hơn CFS-only. Bước tiếp theo cần chạy nhiều seed, benchmark thứ hai và tối ưu trade-off giữa final accuracy với forgetting.

## 17. Câu hỏi có thể được hỏi

### Vì sao áp dụng CFS vào CRCT?

CRCT đã làm việc trực tiếp với feature replay, nên đây là vị trí tự nhiên nhất để dùng CFS chọn feature đa dạng mà không phải thay đổi backbone hoặc sinh lại ảnh.

### Đây có phải triển khai y hệt paper CFS không?

Không. Phần contrastive feature selection được giữ, nhưng nó được thích nghi vào feature replay của HRM-PET thay vì model inversion. Báo cáo nên gọi đây là áp dụng/chuyển giao ý tưởng CFS.

### Vì sao semantic không có trong cấu hình cuối?

Vì ablation cho thấy semantic cải thiện nhẹ loss hoặc forgetting ở một số cấu hình nhưng chưa tăng Acc@1 ổn định. Loại semantic khỏi cấu hình cuối là quyết định dựa trên thực nghiệm.

### Cải tiến nào quan trọng nhất?

Full CRCT replay tạo bước nhảy lớn đầu tiên. Sau đó task-energy selection và energy weighting của CTIRD tạo thêm mức tăng rõ nhất và đồng thời cải thiện loss.

### Có thể khẳng định phương pháp mới tốt hơn về mặt thống kê chưa?

Chưa. Hiện mới có seed 42. Có thể nói kết quả bước đầu tốt hơn trên cấu hình đã kiểm tra, nhưng cần nhiều seed để kết luận thống kê.

### Tại sao final accuracy tăng nhưng forgetting cũng tăng so với CFS-only?

Full classifier correction cải thiện khả năng phân tách toàn bộ lớp ở cuối quá trình, nhưng chênh lệch giữa accuracy lúc task vừa học và accuracy cuối vẫn lớn hơn. Đây là trade-off giữa final accuracy và retention, cần được xử lý ở bước tiếp theo.
