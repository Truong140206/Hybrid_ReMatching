# Ghi chú triển khai CFS cho HRM-PET

## 1. Mục tiêu

Mục tiêu của phần chỉnh sửa này là áp dụng ý tưởng CFS, viết tắt của Contrastive Feature Selection, vào code HRM-PET hiện tại để cải thiện chất lượng đặc trưng giả được dùng trong bước replay/CRCT.

Trong bản gốc, HRM-PET không lưu ảnh cũ. Sau mỗi task, mô hình chỉ lưu thống kê đặc trưng của từng lớp, ví dụ mean và covariance. Khi cần cân chỉnh lại classifier ở giai đoạn CRCT, code sẽ sinh đặc trưng giả bằng cách lấy mẫu ngẫu nhiên từ phân phối Gaussian của từng lớp.

Vấn đề là lấy mẫu Gaussian ngẫu nhiên có thể tạo ra nhiều đặc trưng bị trùng, kém đa dạng hoặc nằm ở vùng không đại diện tốt cho lớp đó. Vì vậy, CFS được thêm vào để chọn ra các đặc trưng giả đa dạng hơn trước khi đưa vào CRCT.

## 2. HRM-PET gốc đang làm gì

Luồng chính của HRM-PET gồm các phần sau:

1. Train mô hình theo từng task trong continual learning.
2. Sau mỗi task, trích xuất feature của dữ liệu thật.
3. Với mỗi class, tính thống kê feature:
   - `cls_mean`: vector trung bình của feature.
   - `cls_cov`: ma trận covariance hoặc variance.
   - Với một số chế độ, có thể dùng nhiều centroid cho mỗi class.
4. Ở giai đoạn CRCT, code sinh lại feature giả từ thống kê đã lưu.
5. Classifier được cân chỉnh lại bằng các feature giả này để giảm quên lãng.

Nói ngắn gọn, bản gốc dùng replay ở feature space, không replay ảnh thật.

## 3. CFS thêm gì vào

CFS không thay đổi backbone, không thay đổi dữ liệu ảnh đầu vào và không thay đổi cách train task chính. CFS chỉ can thiệp vào bước sinh feature giả cho CRCT.

Thay vì lấy mẫu Gaussian rồi dùng luôn, CFS làm thêm các bước:

1. Với mỗi class, lấy feature thật đã trích xuất được.
2. Train một MLP nhỏ theo kiểu contrastive để học không gian so sánh feature.
3. Khi cần sinh feature giả:
   - Sinh nhiều candidate feature từ Gaussian.
   - Đưa candidate qua MLP CFS.
   - Tính độ tương đồng cosine giữa các candidate.
   - Chọn ra các candidate ít giống nhau hơn, tức là đa dạng hơn.
4. Dùng các feature được chọn để train CRCT.

Ý tưởng chính: vẫn dùng phân phối Gaussian của HRM-PET, nhưng thay sampling ngẫu nhiên bằng sampling có chọn lọc.

## 4. CFS đang áp dụng vào đâu

CFS hiện được áp dụng vào phần CRCT feature replay.

Cụ thể:

- Trong TII/HidePrompt engine, CFS được dùng khi sinh feature giả để cân chỉnh classifier.
- Trong LoRA HRM engine, CFS cũng được dùng ở bước sinh feature giả cho CRCT.
- Với mỗi class, CFS học từ feature thật của class đó.
- Khi CRCT cần sample feature, CFS chọn các sample đa dạng hơn từ Gaussian candidate.

CFS hiện không áp dụng vào các phần sau:

- Không áp dụng trực tiếp vào CTIRD.
- Không áp dụng vào ảnh đầu vào.
- Không thay đổi dataloader.
- Không thay đổi kiến trúc ViT backbone.
- Không thay đổi cơ chế LoRA chính.
- Không thay thế hoàn toàn Gaussian replay, mà chỉ cải thiện bước chọn sample từ Gaussian.

## 5. Các file đã thay đổi

### `utils.py`

File này được thêm nhiều hàm hỗ trợ chính:

- `CFSContrastiveMLP`
  - MLP nhỏ dùng để chiếu feature sang một không gian contrastive.
  - Không gian này được dùng để đo độ giống/khác nhau giữa các feature.

- `use_cfs_sampling(args)`
  - Kiểm tra xem CFS có được bật hay không.
  - Nếu không bật `--cfs_sampling`, code sẽ chạy như bản gốc.

- `train_cfs_model(features, args, device)`
  - Train MLP CFS cho từng class dựa trên feature thật của class đó.
  - Mục tiêu là học biểu diễn giúp phân biệt sự tương đồng giữa các feature trong cùng class.

- `sample_cfs_features(mean, cov, num_samples, args, device, cfs_model=None)`
  - Sinh nhiều candidate feature từ Gaussian.
  - Nếu có CFS model thì chọn các candidate đa dạng hơn.
  - Nếu không có CFS hoặc CFS bị tắt, hàm fallback về sampling Gaussian gốc.

Ngoài CFS, `utils.py` cũng được bổ sung một số hàm an toàn hơn cho distributed training và load checkpoint:

- `load_checkpoint`
  - Dùng để load checkpoint ổn định hơn với các version PyTorch mới.

- `distributed_barrier`
  - Bọc lại `torch.distributed.barrier` để tránh lỗi NCCL device không rõ ràng.

- `cleanup_distributed`
  - Dọn distributed process group sau khi chạy xong hoặc khi lỗi.

### `engines/hide_tii_engine.py`

File này là nơi CFS được gắn vào luồng TII/HidePrompt.

Thay đổi chính:

1. Khi tính mean/covariance cho từng class, code lấy toàn bộ feature thật của class đó.
2. Nếu bật CFS, code train một CFS MLP riêng cho class đó.
3. Khi CRCT cần sinh feature giả, thay vì dùng trực tiếp `MultivariateNormal(...).sample(...)`, code gọi `utils.sample_cfs_features(...)`.

Như vậy, TII vẫn giữ logic gốc, nhưng feature giả dùng cho CRCT được chọn tốt hơn.

### `engines/hrm_lora_wtp_and_tap_engine.py`

File này là engine chính cho phần LoRA HRM.

Thay đổi tương tự TII:

1. Train CFS MLP theo từng class sau khi thu feature thật.
2. Lưu CFS model theo class trong `cls_cfs_model`.
3. Khi CRCT sample feature giả, dùng CFS để chọn sample đa dạng.

Điểm này quan trọng vì kết quả cuối cùng thường được lấy từ stage LoRA, nên CFS cần được áp dụng ở cả LoRA engine chứ không chỉ TII.

### `configs/*.py`

Một số file config được thêm tham số CFS:

- `configs/cifar100_hideprompt_5e.py`
- `configs/cifar100_lora.py`
- `configs/five_datasets_hideprompt_5e.py`
- `configs/five_datasets_lora.py`
- `configs/ima_hideprompt_5e.py`
- `configs/ima_lora.py`
- `configs/imr_hideprompt_5e.py`
- `configs/imr_lora.py`

Các tham số mới:

```bash
--cfs_sampling
--cfs_epochs
--cfs_lr
--cfs_momentum
--cfs_hidden_dim
--cfs_batch_size
--cfs_train_max_samples
--cfs_candidate_multiplier
--cfs_tau
```

Ý nghĩa:

- `--cfs_sampling`: bật CFS.
- `--cfs_epochs`: số epoch train MLP CFS.
- `--cfs_lr`: learning rate cho CFS.
- `--cfs_hidden_dim`: hidden dimension của MLP CFS.
- `--cfs_batch_size`: batch size khi train CFS.
- `--cfs_train_max_samples`: giới hạn số feature thật dùng để train CFS cho mỗi class.
- `--cfs_candidate_multiplier`: sinh nhiều candidate hơn số cần lấy bao nhiêu lần.
- `--cfs_tau`: hệ số nhiệt độ khi tính contrastive similarity.

### `training_scripts/kaggle_train_imr_lora_sup21k.sh`

Script Kaggle được thêm khả năng bật CFS bằng biến môi trường.

Ví dụ:

```bash
export CFS_SAMPLING=1
export CFS_EPOCHS=50
export CFS_LR=0.01
export CFS_HIDDEN_DIM=512
export CFS_BATCH_SIZE=256
export CFS_TRAIN_MAX_SAMPLES=1024
export CFS_CANDIDATE_MULTIPLIER=3
export CFS_TAU=1.0
```

Nếu `CFS_SAMPLING=1`, script sẽ tự thêm các tham số CFS vào lệnh chạy TII và LoRA.

Nếu không set `CFS_SAMPLING=1`, script chạy như bản gốc.

### `main.py`

`main.py` được bọc thêm `try/finally` để gọi `utils.cleanup_distributed()`.

Mục đích là tránh lỗi treo process hoặc lỗi distributed khi chạy nhiều GPU, nhất là trên Kaggle/Colab.

## 6. Logic CFS chi tiết

### Bước 1: Thu feature thật theo class

Sau khi train xong một task, code chạy qua dữ liệu để lấy feature của từng class. Ví dụ class `c` có một tập feature:

```text
F_c = {f1, f2, f3, ..., fn}
```

Từ đó HRM-PET gốc tính:

```text
mean_c = mean(F_c)
cov_c  = covariance(F_c)
```

CFS dùng chính `F_c` để train MLP.

### Bước 2: Train CFS MLP

CFS MLP nhận feature gốc và chiếu sang không gian mới:

```text
z = MLP(f)
```

Sau đó normalize:

```text
z = normalize(z)
```

Trong không gian này, code có thể đo cosine similarity giữa các feature.

Mục tiêu là học một không gian mà sự giống nhau giữa feature được thể hiện rõ hơn. Khi chọn sample, ta ưu tiên các sample không quá giống nhau.

### Bước 3: Sinh candidate từ Gaussian

Khi CRCT cần `N` feature giả cho một class, bản gốc sinh đúng `N` feature:

```text
x_fake ~ Gaussian(mean_c, cov_c)
```

CFS sinh nhiều hơn:

```text
num_candidates = N * cfs_candidate_multiplier
```

Ví dụ cần 256 feature và `cfs_candidate_multiplier=3`, code sinh 768 candidate.

### Bước 4: Chọn feature đa dạng

Các candidate được đưa qua CFS MLP:

```text
z_candidate = MLP(x_candidate)
```

Sau đó code chọn dần các candidate có độ tương đồng trung bình thấp hơn với các sample đã chọn.

Hiểu đơn giản:

- Nếu một candidate quá giống những sample đã chọn, bỏ qua.
- Nếu một candidate khác biệt hơn, giữ lại.

Kết quả là tập feature giả cuối cùng đa dạng hơn sampling ngẫu nhiên.

### Bước 5: Đưa feature đã chọn vào CRCT

Các feature giả sau khi chọn được dùng giống như bản gốc:

```text
classifier.train(fake_features, labels)
```

Vì vậy CFS không phá vỡ pipeline cũ. Nó chỉ thay chất lượng feature replay.

## 7. Khác gì so với bản gốc

Bản gốc:

```text
Gaussian statistics -> random sampling -> CRCT
```

Bản đã thêm CFS:

```text
Gaussian statistics -> generate candidates -> CFS selection -> CRCT
```

Điểm khác biệt quan trọng:

- Bản gốc chọn sample ngẫu nhiên.
- Bản CFS chọn sample có tính đa dạng hơn.
- Bản gốc không học thêm module phụ cho feature selection.
- Bản CFS train thêm MLP nhỏ theo từng class.
- CFS chỉ ảnh hưởng CRCT, không làm thay đổi task training chính.

## 8. Vì sao cách áp dụng này hợp lý

HRM-PET đã có sẵn feature statistics của từng class. CFS trong paper cũng hoạt động trên feature space và dùng Gaussian distribution để tạo candidate. Vì vậy điểm ghép tự nhiên nhất là chỗ HRM-PET đang sample feature giả cho CRCT.

Nếu áp dụng CFS quá sớm vào ảnh hoặc dataloader thì sẽ phải thay đổi pipeline lớn hơn, rủi ro cao hơn. Còn nếu áp dụng vào CRCT thì:

- Ít phá code gốc.
- Đúng với bản chất feature replay.
- Dễ bật/tắt để so sánh ablation.
- Có thể kiểm tra bằng kết quả cuối cùng.

## 9. Kết quả thử nghiệm

Đã chạy thử trên CIFAR100 10 task.

Cấu hình CFS:

```text
Dataset: CIFAR100
Number of tasks: 10
TII epochs: 5
LoRA epochs: 5
CRCT epochs: 10
GPU: Colab T4
CFS: enabled
```

Kết quả cuối của LoRA + CFS:

```text
Average accuracy till task10
Acc@task:   88.0600
Acc@1:      88.0000
Acc@5:      97.9400
Loss:       0.5232
Forgetting: 3.8889
Backward:  -3.7000
```

So với bản không bật CFS đã chạy trước đó:

```text
Non-CFS:
Acc@task: 87.0500
Acc@1:    86.9000
Acc@5:    97.5900
Loss:     0.5860

CFS:
Acc@task: 88.0600
Acc@1:    88.0000
Acc@5:    97.9400
Loss:     0.5232
```

Mức cải thiện:

```text
Acc@task: +1.01
Acc@1:    +1.10
Acc@5:    +0.35
Loss:     -0.0628
```

Nhận xét:

- CFS giúp tăng accuracy cuối cùng.
- Loss giảm, cho thấy classifier sau CRCT ổn định hơn.
- Mức cải thiện không quá lớn nhưng có ý nghĩa, vì thay đổi chủ yếu chỉ nằm ở sampling feature replay.

## 10. Hạn chế hiện tại

### 10.1 CFS mới áp dụng vào CRCT

CFS hiện chưa được áp dụng vào CTIRD. Điều này là có chủ ý để giữ thay đổi gọn và dễ kiểm chứng.

Nếu muốn áp dụng tiếp vào CTIRD, hướng hợp lý là dùng CFS để chọn feature/anchor quan trọng hơn cho distillation, hoặc dùng semantic-aware projection để điều chỉnh quan hệ giữa class cũ và class mới.

### 10.2 Mỗi class train một MLP nhỏ

Cách này bám sát ý tưởng CFS nhưng có thêm chi phí tính toán. Với dataset lớn như ImageNet-R, cần cân nhắc:

- giảm `cfs_epochs`,
- giảm `cfs_train_max_samples`,
- giảm `cfs_candidate_multiplier`.

### 10.3 Multi-centroid chưa phải bản paper gốc tuyệt đối

Với chế độ multi-centroid, code dùng CFS model theo class để chọn sample từ từng centroid. Đây là cách mở rộng hợp lý cho HRM-PET, nhưng không phải bê nguyên xi từ paper.

## 11. Semantic-aware feature projection thì sao

Phần semantic-aware feature projection trong paper dùng thông tin ngữ nghĩa, thường là CLIP text/image feature, để điều chỉnh feature theo hướng quan hệ giữa class.

Ý tưởng này có thể áp dụng vào HRM-PET/CTIRD nhưng chưa được code trong lần này.

Hướng áp dụng hợp lý:

1. Lấy text embedding của class name bằng CLIP.
2. Tính quan hệ semantic giữa class cũ và class mới.
3. Khi distillation hoặc replay, điều chỉnh feature cũ theo hướng semantic phù hợp.
4. Dùng quan hệ semantic để chọn class liên quan hơn thay vì distill tất cả ngang nhau.

Với CTIRD, semantic-aware projection có thể dùng để làm relation distillation thông minh hơn:

- Class mới nào gần class cũ hơn về ngữ nghĩa thì giữ quan hệ mạnh hơn.
- Class ít liên quan thì giảm ảnh hưởng.
- Điều này có thể giúp giảm nhầm lẫn giữa các task.

Tuy nhiên, phần này cần thêm CLIP dependency và class-name mapping chuẩn, nên chưa nên ghép ngay trước khi CFS được kiểm chứng ổn định.

## 12. Cách chạy CFS

Ví dụ bật CFS trên Kaggle hoặc Colab:

```bash
export CFS_SAMPLING=1
export CFS_EPOCHS=50
export CFS_LR=0.01
export CFS_HIDDEN_DIM=512
export CFS_BATCH_SIZE=256
export CFS_TRAIN_MAX_SAMPLES=1024
export CFS_CANDIDATE_MULTIPLIER=3
export CFS_TAU=1.0
```

Nếu chạy trực tiếp bằng `main.py`, thêm:

```bash
--cfs_sampling \
--cfs_epochs 50 \
--cfs_lr 0.01 \
--cfs_hidden_dim 512 \
--cfs_batch_size 256 \
--cfs_train_max_samples 1024 \
--cfs_candidate_multiplier 3 \
--cfs_tau 1.0
```

Nếu muốn tắt CFS, chỉ cần bỏ `--cfs_sampling` hoặc set:

```bash
export CFS_SAMPLING=0
```

## 13. Tóm tắt để trình bày

Trong đề tài này, nhóm đã bổ sung Contrastive Feature Selection vào HRM-PET để cải thiện bước replay ở feature space. Bản gốc HRM-PET sinh feature giả bằng cách lấy mẫu ngẫu nhiên từ Gaussian distribution của từng class. Cách này đơn giản nhưng có thể tạo ra các sample trùng lặp hoặc kém đại diện.

Phương pháp đề xuất giữ nguyên pipeline chính của HRM-PET, nhưng thêm một MLP contrastive nhỏ cho từng class. MLP này học từ feature thật của class đó. Khi CRCT cần feature giả, hệ thống sinh nhiều candidate từ Gaussian, đưa qua CFS MLP, sau đó chọn các candidate đa dạng hơn dựa trên cosine similarity trong không gian contrastive.

CFS được áp dụng vào cả TII engine và LoRA HRM engine ở bước CRCT. Các phần backbone, dataloader, task training và LoRA training chính được giữ nguyên. Nhờ vậy, thay đổi có thể bật/tắt dễ dàng bằng tham số `--cfs_sampling`, thuận tiện cho so sánh ablation.

Kết quả thử nghiệm trên CIFAR100 10 task cho thấy CFS cải thiện Acc@1 từ 86.90 lên 88.00, Acc@task từ 87.05 lên 88.06 và giảm loss từ 0.5860 xuống 0.5232. Điều này cho thấy việc chọn feature replay đa dạng hơn có thể giúp classifier sau CRCT ổn định hơn và giảm quên lãng tốt hơn.

## 14. Kết luận

Việc áp dụng CFS vào HRM-PET là hợp lý vì cả hai đều làm việc trên feature space. CFS không thay thế toàn bộ phương pháp gốc mà đóng vai trò cải thiện chất lượng feature replay trong CRCT. Thay đổi này ít phá vỡ code cũ, dễ kiểm chứng và đã cho kết quả cải thiện trên CIFAR100.

Hướng phát triển tiếp theo là đưa semantic-aware feature projection vào CTIRD hoặc relation distillation để tận dụng thông tin ngữ nghĩa giữa các class, từ đó cải thiện khả năng giữ kiến thức cũ khi học task mới.

## 15. Ph?n semantic-aware distillation m?i th�m

Sau ph?n CFS, code du?c b? sung th�m semantic-aware relation distillation cho CTIRD trong LoRA engine.

L� do th�m v�o CTIRD:

- CTIRD hi?n dang gi? quan h? gi?a c�c sample b?ng c�ch distill ma tr?n similarity feature t? model cu sang model m?i.
- Tuy nhi�n, b?n g?c coi m?i quan h? trong batch g?n nhu ngang nhau.
- Semantic-aware distillation th�m th�ng tin t�n class v�o quan h? n�y, d? c�c class g?n nghia nhau du?c gi? relation m?nh hon, c�n c�c class �t li�n quan du?c gi?m ?nh hu?ng.

C�ch l�m hi?n t?i:

1. `datasets.py` l?y `dataset.classes` v� luu v�o `args.class_names`.
2. `utils.py` t?o semantic embedding nh? t? t�n class b?ng token/char n-gram hashing.
3. T? semantic embedding, code t?o ma tr?n semantic similarity gi?a c�c class.
4. Trong `hrm_lora_wtp_and_tap_engine.py`, tru?c khi t�nh KL loss c?a CTIRD, target relation t? model cu du?c nh�n v?i semantic weight r?i normalize l?i.
5. Loss CTIRD sau d� distill theo target relation d� c� semantic.

C�ng th?c � tu?ng:

```text
old_relation = similarity(feature_old)
semantic_weight = similarity(text_class_i, text_class_j)
semantic_relation = normalize(old_relation * semantic_weight)
loss_CTIRD = KL(new_relation || semantic_relation)
```

C�c tham s? m?i:

```bash
--semantic_distill
--semantic_dim 512
--semantic_alpha 1.0
--semantic_floor 0.2
--semantic_sharpness 1.0
```

� nghia:

- `--semantic_distill`: b?t semantic-aware CTIRD.
- `--semantic_dim`: k�ch thu?c vector semantic hashing.
- `--semantic_alpha`: m?c pha semantic v�o relation target. `1.0` l� d�ng semantic d?y d?, `0.0` l� g?n nhu b?n g?c.
- `--semantic_floor`: gi? l?i m?c relation t?i thi?u d? tr�nh l�m m?t ho�n to�n quan h? gi?a class xa nghia.
- `--semantic_sharpness`: l�m semantic similarity s?c hon n?u tang l?n hon 1.

�i?m quan tr?ng:

- Semantic hi?n du?c g?n v�o CTIRD, kh�ng ph?i CRCT.
- CFS v?n n?m ? CRCT feature replay.
- Hai ph?n n�y d?c l?p, c� th? b?t ri�ng ho?c b?t c�ng nhau.
- Kh�ng c?n t?i CLIP, n�n ch?y du?c tr�n Kaggle/Colab kh�ng c?n th�m dependency l?n.
- ��y l� b?n �p d?ng � tu?ng semantic-aware projection theo hu?ng nh? v� ?n d?nh tru?c. N?u mu?n b�m s�t paper hon, bu?c ti?p theo l� thay semantic hashing b?ng CLIP text embedding ho?c WordNet class-name embedding.

C�ch b?t tr�n Kaggle script:

```bash
export SEMANTIC_DISTILL=1
export SEMANTIC_ALPHA=1.0
export SEMANTIC_FLOOR=0.2
export SEMANTIC_SHARPNESS=1.0
```

N?u mu?n ch?y c? CFS v� semantic:

```bash
export CFS_SAMPLING=1
export SEMANTIC_DISTILL=1
```

T�m t?t d? tr�nh b�y th�m:

Sau khi th�m CFS cho CRCT, nh�m ti?p t?c b? sung semantic-aware relation distillation cho CTIRD. Thay v� distill ma tr?n quan h? feature m?t c�ch thu?n t�y, phuong ph�p m?i d�ng t�n class d? t?o semantic similarity, sau d� d�ng semantic similarity n�y d? di?u ch?nh target relation t? model cu. Nh? v?y, qu� tr�nh distillation uu ti�n gi? c�c quan h? c� � nghia ng? nghia hon, gi�p gi?m nhi?u khi h?c task m?i.