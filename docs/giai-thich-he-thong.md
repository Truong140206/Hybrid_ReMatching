# Hệ thống này làm gì — giải thích từ đầu

Tài liệu này viết cho người chưa biết gì về đề tài. Đọc hết sẽ hiểu: bài toán là
gì, bài báo gốc HRM-PET hoạt động ra sao, chúng ta phát hiện điều gì, thêm cái gì
vào, thêm ở chỗ nào trong mã nguồn, và tại sao lại thêm đúng chỗ đó.

Không cần đọc mã nguồn để hiểu tài liệu này. Nhưng mọi chỗ nhắc tới mã đều có
đường dẫn kèm số dòng để tra khi cần.

---

## Mục lục

1. [Bài toán: học liên tục](#1-bài-toán-học-liên-tục)
2. [HRM-PET hoạt động như thế nào](#2-hrm-pet-hoạt-động-như-thế-nào)
3. [Ba phát hiện dẫn tới đề xuất](#3-ba-phát-hiện-dẫn-tới-đề-xuất)
4. [Thành phần thứ nhất: đầu RP](#4-thành-phần-thứ-nhất-đầu-rp)
5. [Thành phần thứ hai: hợp nhất định tuyến](#5-thành-phần-thứ-hai-hợp-nhất-định-tuyến)
6. [Thành phần thứ ba: hợp nhất phân lớp có cổng](#6-thành-phần-thứ-ba-hợp-nhất-phân-lớp-có-cổng)
7. [Bốn cạm bẫy đã gặp](#7-bốn-cạm-bẫy-đã-gặp)
8. [Kết quả và cách đọc chúng](#8-kết-quả-và-cách-đọc-chúng)
9. [Cách chạy lại toàn bộ](#9-cách-chạy-lại-toàn-bộ)
10. [Những hướng đã thử và đã đóng](#10-những-hướng-đã-thử-và-đã-đóng)

---

## 1. Bài toán: học liên tục

### 1.1. Vấn đề

Một mô hình nhận diện ảnh bình thường được huấn luyện một lần trên toàn bộ dữ
liệu, rồi đem dùng. **Học liên tục** (continual learning) là tình huống khác: dữ
liệu đến theo từng đợt, và mô hình phải học đợt mới mà không được xem lại đợt cũ.

Ví dụ cụ thể với bộ dữ liệu Split-CIFAR100 mà chúng ta dùng:

| Đợt (tác vụ) | Các lớp phải học |
|---|---|
| Tác vụ 1 | lớp 0–9 |
| Tác vụ 2 | lớp 10–19 |
| ... | ... |
| Tác vụ 10 | lớp 90–99 |

Khi học tác vụ 2, mô hình **không còn được xem ảnh của lớp 0–9 nữa**. Đến cuối
cùng, mô hình phải phân loại đúng cả 100 lớp.

### 1.2. Vì sao khó: quên thảm khốc

Nếu cứ huấn luyện tiếp một cách ngây thơ, mạng nơ-ron sẽ ghi đè các trọng số cũ
để tối ưu cho dữ liệu mới. Kết quả là nó học tác vụ 10 rất tốt nhưng gần như quên
sạch tác vụ 1. Hiện tượng này gọi là **quên thảm khốc** (catastrophic forgetting)
và là trở ngại trung tâm của cả lĩnh vực.

### 1.3. Ràng buộc: không lưu mẫu

Cách chống quên đơn giản nhất là giữ lại vài trăm ảnh cũ để ôn lại — gọi là
*replay*. Nhưng nhiều tình huống thực tế cấm điều đó: dữ liệu y tế không được lưu
vì lý do riêng tư, dữ liệu người dùng bị giới hạn bởi luật.

Đề tài này theo giao thức **không lưu mẫu** (exemplar-free) nghiêm ngặt: không
giữ bất kỳ ảnh nào, cũng không giữ đặc trưng của từng ảnh riêng lẻ. Chỉ được giữ
các đại lượng thống kê tổng hợp — ví dụ trung bình và hiệp phương sai của cả một
lớp. Toàn bộ mã đánh giá chạy dưới cờ `--strict_exemplar_free`, và có một bước
kiểm tra checkpoint sẽ báo lỗi nếu phát hiện đặc trưng theo mẫu bị lưu lén
([training_scripts/eval_rp_head_any_4090.sh:63](../training_scripts/eval_rp_head_any_4090.sh#L63)).

### 1.4. Cách tiếp cận hiện đại: đóng băng mạng lớn, gắn bộ điều hợp nhỏ

Thay vì huấn luyện lại toàn bộ mạng, hướng hiện đại làm thế này:

1. Lấy một mạng thị giác lớn đã được huấn luyện sẵn trên tập dữ liệu khổng lồ
   (ở đây là **ViT-B/16** huấn luyện trên ImageNet-21K, khoảng 14 triệu ảnh).
2. **Đóng băng** toàn bộ mạng đó — không đụng vào một trọng số nào.
3. Với mỗi tác vụ mới, huấn luyện một **bộ điều hợp** (adapter) rất nhỏ gắn thêm
   vào. Ở đây dùng **LoRA hạng 8**, chỉ vài trăm nghìn tham số.

Vì mạng gốc không bao giờ thay đổi, kiến thức cũ trong nó không thể bị ghi đè.
Mỗi tác vụ có LoRA riêng, và LoRA của tác vụ 1 vẫn nguyên vẹn sau khi học tác
vụ 10.

Nhưng cách này đẻ ra một vấn đề mới, và đó chính là chỗ HRM-PET bước vào.

---

## 2. HRM-PET hoạt động như thế nào

### 2.1. Vấn đề mà HRM-PET giải quyết

Ta có 10 bộ LoRA, mỗi bộ chuyên cho một tác vụ. Khi một ảnh mới đến lúc kiểm tra,
**ta không được cho biết ảnh đó thuộc tác vụ nào**. Phải tự đoán để chọn đúng
LoRA.

Hình dung như một thư viện có 10 chuyên gia, mỗi người giỏi một lĩnh vực. Có
người mang ảnh đến hỏi, nhưng không nói ảnh thuộc lĩnh vực nào. Người trực cửa
phải đoán để chuyển đến đúng chuyên gia. Đoán sai thì câu trả lời có nguy cơ sai.

Bước đoán đó gọi là **suy luận danh tính tác vụ** (task-identity inference, viết
tắt TII).

### 2.2. Bốn bước của HRM-PET

**Bước 1 — TII: đoán tác vụ.**
Một mô hình phụ (trong mã gọi là `original_model`) chạy ảnh qua và đo *năng lượng
prompt*. Ý tưởng: mỗi tác vụ có một tập prompt riêng, và prompt của tác vụ đúng
sẽ phản ứng mạnh hơn với ảnh thuộc tác vụ đó. Hàm đo dùng ở đây là **entropy suy
rộng** (generalized entropy, cờ `--En gen`), cài ở
[engines/hrm_lora_wtp_and_tap_engine.py:67](../engines/hrm_lora_wtp_and_tap_engine.py#L67).

**Bước 2 — Định tuyến.**
Chọn tác vụ có điểm cao nhất, nạp LoRA của tác vụ đó vào mạng, rồi chạy ảnh qua.

**Bước 3 — Re-matching bằng DRM và CRM.**
Đây là đóng góp chính của bài báo gốc. Sau khi chạy LoRA đầu tiên, mô hình đưa ra
một lớp dự đoán. Mỗi lớp thuộc về đúng một tác vụ, nên **bản thân lớp được dự
đoán cũng là một phiếu bầu cho tác vụ**.

- **DRM** (Direct Re-Matching) dùng luôn lớp dự đoán đó để định tuyến lại: nếu
  LoRA của tác vụ 3 lại dự đoán ra một lớp thuộc tác vụ 7, thì có lẽ nên thử LoRA
  của tác vụ 7.
- **CRM** (Confidence Re-Matching) chỉ kích hoạt việc định tuyến lại khi độ tin
  cậy thấp, đo bằng chính hàm GEN. Dự đoán đã chắc chắn thì để yên.

Đây là lý do HRM-PET tốn trung bình **1.83 lượt gọi LoRA cho mỗi ảnh** trên
ImageNet-R chứ không phải 1.0 — một số ảnh phải chạy lại lần hai.

**Bước 4 — Phân lớp.**
Đưa đặc trưng cuối vào một **đầu phân lớp dùng chung**, trải trên *toàn bộ* các
lớp đã gặp từ đầu đến giờ.

> **Ghi nhớ chi tiết này.** Đầu phân lớp là **dùng chung cho mọi tác vụ**, không
> phải mỗi tác vụ một đầu riêng. Toàn bộ phần 3.2 xoay quanh hệ quả của nó.

### 2.3. Sơ đồ

```
                    ┌──────────┐
                    │ TII      │  đoán tác vụ bằng năng lượng prompt
   ảnh vào ────────►│ (bước 1) │
                    └────┬─────┘
                         │ tác vụ được đoán
                         ▼
                    ┌──────────┐
                    │ Chọn     │  nạp LoRA của tác vụ đó
                    │ LoRA     │
                    └────┬─────┘
                         ▼
                    ┌──────────┐
                    │ DRM/CRM  │  dùng lớp dự đoán để định tuyến lại
                    │ (bước 3) │  nếu chưa chắc chắn
                    └────┬─────┘
                         ▼
                    ┌──────────┐
                    │ Đầu phân │  trải trên MỌI lớp đã gặp
                    │ lớp chung│
                    └────┬─────┘
                         ▼
                      dự đoán
```

---

## 3. Ba phát hiện dẫn tới đề xuất

### 3.1. Phát hiện 1 — TII không phải tín hiệu định tuyến tốt nhất

Chúng ta thử một cách đoán tác vụ hoàn toàn khác, gọi là **đầu RP** (mô tả kỹ ở
phần 4). Nó không dùng prompt, không dùng năng lượng, mà dùng đặc trưng ảnh và
một phép hồi quy tuyến tính.

Kết quả đo trên seed 42:

| Cách đoán tác vụ | ImageNet-R | CIFAR-100 | CUB-200 |
|---|---|---|---|
| TII (bài báo gốc, bước 1) | 63.80 | 81.21 | 92.96 |
| HRM-PET sau khi có DRM/CRM | 77.79 | 89.82 | 93.21 |
| Đầu RP | 77.12 | 89.51 | 94.02 |
| **Hợp của TII và đầu RP** | **82.99** | **92.86** | **96.26** |

Hàng cuối cần giải thích. Đó **không phải** một phương pháp — đó là phép đo
"nếu ta có một vị thần biết chọn giữa hai câu trả lời thì đúng được bao nhiêu".
Nói cách khác: tỉ lệ ảnh mà **ít nhất một** trong hai cách đoán ra kết quả đúng.

Con số 82.99 so với 77.79 nói lên điều then chốt: **hai cách đoán sai ở những ảnh
khác nhau**. Trên ImageNet-R, đầu RP đoán đúng 19.2 % số ảnh mà TII đoán sai.

Đây là điều kiện bắt buộc để việc kết hợp có ý nghĩa. Nếu hai nguồn cùng sai ở
cùng những ảnh, gộp lại chẳng thêm được gì.

### 3.2. Phát hiện 2 — định tuyến tốt hơn không tự động thành phân lớp tốt hơn

Sau khi cải thiện định tuyến, chúng ta gặp một điều bất ngờ: **định tuyến tăng
hơn 2 điểm, nhưng độ chính xác phân lớp chỉ tăng 0.29 điểm.** Trên CUB-200 thì
còn tệ hơn: định tuyến tăng 1.13 điểm mà phân lớp *giảm* 0.05 điểm.

Nguyên nhân nằm ở chi tiết đã đánh dấu ở mục 2.2: **đầu phân lớp dùng chung**.

Giả sử một ảnh thuộc tác vụ 3 nhưng bị định tuyến nhầm sang LoRA của tác vụ 7.
LoRA số 7 vẫn là một bộ điều hợp nhỏ gắn trên cùng mạng xương sống, nên đặc trưng
nó tạo ra không quá khác biệt. Đầu phân lớp lại nhìn thấy *tất cả* 200 lớp, nên
nó hoàn toàn có thể vẫn chọn đúng lớp thuộc tác vụ 3.

Từ đó suy ra hai điều:

- Sửa đường định tuyến cho ảnh đó **không mua được gì** — nó vốn đã đúng.
- Ngược lại, định tuyến lại một ảnh vốn đã đúng **có thể làm nó sai đi**.

Đó là lý do phần lợi ích bị thất thoát. Muốn Acc@1 tăng thật thì phải can thiệp
vào chính bước phân lớp, không chỉ bước định tuyến.

### 3.3. Phát hiện 3 — đầu RP còn là một bộ phân lớp bị bỏ phí

Đầu RP không chỉ đoán tác vụ. Bản thân nó là **một bộ phân lớp hoàn chỉnh** —
nó cho điểm số cho từng lớp cụ thể, chứ không chỉ từng tác vụ. Chúng ta đang dùng
nó để đoán tác vụ rồi vứt phần còn lại đi.

Đo mức bổ trợ ở cấp phân lớp (seed 42):

| Bộ dữ liệu | Đầu phân lớp chính | Đầu RP | Hợp | Đầu RP cứu riêng |
|---|---|---|---|---|
| ImageNet-R | 74.31 | 70.08 | 78.89 | **+4.58** |
| CIFAR-100 | 89.97 | 88.76 | 92.40 | **+2.43** |
| CUB-200 | 86.48 | **87.50** | 90.35 | **+3.87** |

Cột cuối là tỉ lệ ảnh mà đầu phân lớp chính sai **còn đầu RP đúng**.

Điểm đáng chú ý: trên ImageNet-R đầu RP yếu hơn tới 4 điểm về tổng thể (70.08 so
với 74.31), **nhưng nó vẫn đúng ở 4.58 % số ảnh mà đầu chính bỏ lỡ**. Một bộ phân
lớp yếu hơn vẫn có giá trị nếu nó sai ở chỗ khác.

Trên CUB-200 thì đầu RP còn mạnh hơn cả đầu chính (87.50 so với 86.48).

---

## 4. Thành phần thứ nhất: đầu RP

### 4.1. Ý tưởng

Đầu RP là phiên bản của kỹ thuật trong bài báo **RanPAC**. Ý tưởng nghe hơi phản
trực giác nhưng rất hiệu quả:

> Chiếu đặc trưng lên một không gian ngẫu nhiên có số chiều rất lớn, rồi giải một
> bài toán hồi quy tuyến tính trong không gian đó.

Cụ thể ba bước:

1. **Chiếu ngẫu nhiên.** Nhân đặc trưng 768 chiều với một ma trận ngẫu nhiên
   `W` để ra 10 000 chiều, rồi cho qua ReLU (bỏ phần âm).
   Ma trận `W` **được sinh ra một lần từ một seed cố định và không bao giờ được
   huấn luyện**.

2. **Tích lũy thống kê.** Với mỗi ảnh huấn luyện, cộng dồn vào hai ma trận:
   - `G` — ma trận Gram, ghi lại các chiều nào hay xuất hiện cùng nhau.
   - `C` — ma trận tương quan lớp, ghi lại chiều nào ứng với lớp nào.

3. **Giải hồi quy ridge.** Trọng số phân lớp là `(G + λI)⁻¹ C`.

```
h = ReLU(f · W)          W đóng băng, sinh lại từ seed
G = Σ h hᵀ               cộng dồn qua mọi ảnh đã thấy
C = Σ h onehot(y)ᵀ       cộng dồn qua mọi ảnh đã thấy
s = hᵀ (G + λI)⁻¹ C      điểm số cho từng lớp
```

### 4.2. Vì sao cách này hợp với học liên tục

Ba lý do, và cả ba đều quan trọng:

**Nó tuân thủ ràng buộc không lưu mẫu.** `G` và `C` là *tổng cộng dồn*. Khi tác
vụ mới đến, chỉ cần cộng thêm vào, không cần xem lại dữ liệu cũ. Ma trận `W` sinh
lại được từ seed nên cũng không cần lưu. Không có ảnh nào, không có đặc trưng
riêng lẻ nào được giữ.

**Nó không hề quên.** Không có bước huấn luyện lặp nào để ghi đè kiến thức cũ.
Nghiệm hồi quy trên `G` và `C` đầy đủ tương đương với việc huấn luyện tuyến tính
trên toàn bộ dữ liệu cùng lúc.

**Nó không tốn gì thêm.** Toàn bộ phép tính là cộng ma trận và một lần giải hệ
tuyến tính ở cuối mỗi tác vụ.

### 4.3. Một chi tiết dễ sai

Đặc trưng `f` phải lấy từ **một bộ LoRA cố định duy nhất** (cờ `--rp_lora_task 0`
— luôn dùng LoRA của tác vụ đầu tiên), chứ không phải mỗi tác vụ dùng LoRA riêng
của nó.

Lý do: `G` và `C` cộng dồn qua mọi tác vụ. Nếu mỗi tác vụ dùng một LoRA khác, các
đặc trưng sẽ nằm ở những không gian khác nhau và việc cộng dồn trở nên vô nghĩa.
Dùng một LoRA cố định chính là vai trò **thích nghi phiên đầu**
(first-session adaptation) trong RanPAC — chỉ khác là ở đây nó có sẵn từ HRM-PET
nên không phải huấn luyện thêm.

### 4.4. Thêm vào đâu trong mã nguồn

| Việc | Vị trí |
|---|---|
| Toàn bộ đầu RP | [engines/random_projection_head.py](../engines/random_projection_head.py) — file mới |
| Sinh ma trận chiếu | `_ensure_projection`, dòng 72 |
| Cộng dồn `G` và `C` | `accumulate_rp_statistics`, dòng 115 |
| Giải hồi quy ridge | `solve_rp_head`, dòng 135 |
| Gọi ở cuối mỗi tác vụ | [trainers/lora_trainer.py:312–321](../trainers/lora_trainer.py#L312) |
| Trích đặc trưng | `pin_rp_extractor` / `rp_extractor`, engine dòng 690 và 711 |

Cờ bật: `--rp_head --rp_dim 10000 --rp_lambda 1e4 --rp_feature_source lora --rp_lora_task 0`

---

## 5. Thành phần thứ hai: hợp nhất định tuyến

### 5.1. Làm gì

Thay vì để TII một mình quyết định tác vụ, ta trộn điểm số của TII với điểm số
của đầu RP:

```
fused = w · z(TII) + (1 − w) · z(RP)          w = 0.7
```

Tác vụ có điểm cao nhất được chọn, rồi **giao cho DRM/CRM và đầu phân lớp của
HRM-PET xử lý tiếp như bình thường**.

Điểm quan trọng: chúng ta **không bỏ đi thành phần nào của HRM-PET**. DRM, CRM,
đầu phân lớp chung, cơ chế CRCT — tất cả giữ nguyên. Chỉ bước 1 (TII đơn độc)
được thay bằng bước 1 mới (TII kết hợp đầu RP).

### 5.2. Ký hiệu z là gì và tại sao bắt buộc phải có

`z(·)` là phép **chuẩn hóa theo từng ảnh**: trừ đi trung bình rồi chia cho độ
lệch chuẩn, tính riêng trên tập tác vụ đã thấy của chính ảnh đó.

Bỏ bước này đi thì phương pháp hỏng hoàn toàn, và đây là lỗi chúng ta thực sự đã
mắc phải. Nguyên nhân:

- Logit của TII có biên độ rất rộng, ví dụ trải từ −20 đến +15.
- Điểm số hồi quy ridge của đầu RP có biên độ hẹp, ví dụ từ −0.3 đến +0.8.

Cộng thẳng hai đại lượng này thì TII át hoàn toàn ở **mọi** giá trị `w`, kể cả
`w = 0.1`. Khi chưa sửa, độ chính xác định tuyến rơi xuống **68.24** — thấp hơn
cả mức 77.12 của riêng đầu RP. Trộn hai nguồn mà lại tệ hơn từng nguồn riêng lẻ
chính là dấu hiệu của lỗi thang đo.

Sau khi chuẩn hóa, cả hai về cùng thang, và `w` mới thực sự mang nghĩa "trọng số".

### 5.3. Thêm vào đâu

| Việc | Vị trí |
|---|---|
| Hàm hợp nhất | `fuse_routers`, [engine dòng 763](../engines/hrm_lora_wtp_and_tap_engine.py#L763) |
| Phép chuẩn hóa `z` | hàm `standardize` bên trong `fuse_routers` |
| Đổi điểm lớp thành điểm tác vụ | `_task_scores_from_class_scores`, dòng 719 |
| Tiêm kết quả vào luồng DRM | [engine dòng 2006](../engines/hrm_lora_wtp_and_tap_engine.py#L2006), ngay trước `id_logits = F.softmax(...)` ở dòng 2015 |

Vị trí tiêm là chỗ đáng chú ý nhất. Chúng ta chèn đúng vào **trước** khi HRM-PET
chuẩn hóa điểm định tuyến thành xác suất, nên toàn bộ DRM/CRM phía sau vẫn chạy
bình thường trên đường định tuyến mới. Không dòng nào của HRM-PET phải sửa.

Cờ bật: `--rp_route_fusion_drm --rp_route_fusion_weight 0.7`

---

## 6. Thành phần thứ ba: hợp nhất phân lớp có cổng

### 6.1. Phiên bản đầu tiên và lỗi của nó

Từ phát hiện 3, ý tưởng hiển nhiên là trộn luôn hai bộ phân lớp:

```
mixed = (1 − β) · z(đầu chính) + β · z(đầu RP)          β cố định = 0.3
```

Kết quả: Acc@1 tăng đáng kể trên cả ba bộ dữ liệu (+1.40 / +0.76 / +1.61). Nhưng
xuất hiện một vấn đề dai dẳng: **hàm mất mát trên CIFAR-100 xấu đi
+0.019 ± 0.003**.

Con số này nhỏ nhưng phải xử lý, vì độ lệch chuẩn 0.003 chỉ bằng một phần sáu giá
trị trung bình — nghĩa là nó **lặp lại y hệt trên mọi seed**. Đó là lỗi hệ thống,
không phải nhiễu ngẫu nhiên. (Đối chiếu: chỉ số Backward có độ lệch chuẩn *lớn
hơn* trung bình nhiều lần, và đó mới là nhiễu thật.)

### 6.2. Truy nguyên

Trọng số cố định đối xử **như nhau** với hai loại ảnh có bản chất hoàn toàn khác:

- Ảnh mà đầu phân lớp chính đã trả lời chắc nịch, xác suất 0.97.
- Ảnh mà nó đang phân vân giữa hai lớp, 0.35 với 0.30.

Trên CIFAR-100, loại thứ nhất chiếm đa số áp đảo — mô hình vốn đã đúng gần 90 %.
Dành 30 % quyền quyết định cho ý kiến thứ hai trên chính những ảnh đó chỉ **làm
phẳng một đỉnh xác suất vốn đã đúng**. Hàm mất mát phạt ngay lập tức, trong khi
dự đoán cuối cùng chẳng đổi.

Nói ngắn gọn: ta đang trả giá ở 90 % số ảnh để mua lợi ích ở 10 % còn lại.

### 6.3. Cách sửa: cơ chế cổng theo biên top-2

Cho `β` thay đổi theo từng ảnh, tỉ lệ nghịch với mức độ chắc chắn của đầu phân
lớp chính:

```
βᵢ = β · (1 − (p₁ − p₂) / (p₁ + p₂))          β = 0.5
```

trong đó `p₁` và `p₂` là hai xác suất cao nhất mà đầu phân lớp chính đưa ra.

**Vì sao chọn biên giữa hai lớp cao nhất, chứ không phải entropy hay xác suất cao
nhất?** Lập luận rất cụ thể: việc trộn chỉ có thể thay đổi dự đoán bằng cách hất
lớp đang đứng đầu xuống. Mà ứng viên khả dĩ nhất để thay thế nó chính là lớp
đứng thứ hai. Vậy đại lượng cần đo là khoảng cách giữa đúng hai lớp đó.

Xem cơ chế chạy thực tế:

| Tình huống | p₁ | p₂ | βᵢ | Hệ quả |
|---|---|---|---|---|
| Mô hình rất chắc chắn | 0.90 | 0.02 | ≈ 0.007 | gần như không đụng tới |
| Mô hình đang phân vân | 0.35 | 0.30 | ≈ 0.28 | trộn gần hết công suất |

Đúng như mong muốn: giữ nguyên chỗ mô hình đã đúng, can thiệp mạnh chỗ nó đang
lung lay.

Vì cơ chế cổng kéo trọng số hiệu dụng xuống rất nhiều, `β` danh nghĩa phải nâng
từ 0.3 lên **0.5**. Chúng ta có thử 0.8 nhưng lúc đó trộn quá mạnh và ImageNet-R
hỏng cả ba chỉ số Loss, Forgetting, Backward.

Kết quả sau khi thêm cổng: Loss trên CIFAR-100 từ **+0.019 ± 0.003** về
**−0.000 ± 0.001**. Suy giảm hệ thống biến mất hoàn toàn.

### 6.4. Bước cuối: đưa về lại thang đo cũ

Sau khi trộn, kết quả nằm ở thang "trung bình 0, phương sai 1", không phải thang
của logit ban đầu. Đưa nó về:

```
ℓᵢ = mᵢ · std(logit gốc) + mean(logit gốc)
```

Đây là **biến đổi affine theo từng ảnh**, tức là đơn điệu, nên **thứ tự xếp hạng
giữa các lớp không đổi** — mọi chỉ số độ chính xác giữ nguyên bit-for-bit. Nó chỉ
đưa thang đo về mức so sánh được với baseline khi tính hàm mất mát.

### 6.5. Thêm vào đâu

| Việc | Vị trí |
|---|---|
| Hàm trộn phân lớp | `fuse_class_scores`, [engine dòng 854](../engines/hrm_lora_wtp_and_tap_engine.py#L854) |
| Cơ chế cổng | `_fusion_gate`, dòng 824 |
| Chuẩn hóa trên lớp hợp lệ | `_valid_moments` dòng 808, `_standardize_valid` dòng 818 |
| Nơi gọi | [engine dòng 2178–2179](../engines/hrm_lora_wtp_and_tap_engine.py#L2178), ngay sau khi đã có `logits` và trước khi tính loss |

Cờ bật: `--rp_class_fusion_weight 0.5 --rp_class_fusion_gate margin`

**Chi phí bằng không.** Điểm số của đầu RP đã được tính ở tầng định tuyến rồi;
tầng này chỉ dùng lại. Toàn bộ phần tăng thêm của hệ thống là đúng **một lượt
chạy mạng** cho đầu RP.

---

## 7. Bốn cạm bẫy đã gặp

Ghi lại để người sau không mất công lặp lại.

### 7.1. Lệch thang đo khi trộn

Đã mô tả ở 5.2. Dấu hiệu nhận biết: **kết quả trộn tệ hơn từng nguồn riêng lẻ**.
Nếu gặp lại triệu chứng này ở bất kỳ phép trộn nào, hãy nghi ngờ thang đo trước
tiên.

Cách kiểm chứng đã dùng: đặt `w = 1.0` phải tái lập baseline **chính xác đến từng
chữ số**. Chúng ta xác nhận được 77.7914 / 74.0191 / 1.2305 trên ba bộ dữ liệu,
không sai khác ở chữ số thập phân thứ tư.

### 7.2. Lỗi chuẩn hóa ảnh đầu vào — lỗi thật của repo

Hàm `build_transform` trong [datasets.py:347](../datasets.py#L347) gọi `ToTensor()`
mà **không** gọi `Normalize()`. Ảnh do đó nằm trong khoảng [0, 1], trong khi
checkpoint ViT-B/16 của ImageNet-21K mong đợi khoảng [−1, 1].

HRM-PET miễn nhiễm với lỗi này vì nó *huấn luyện* trên chính những đặc trưng bị
lệch đó. Nhưng một đầu phân lớp không huấn luyện như đầu RP thì không miễn nhiễm.
Sửa lại đúng khoảng giá trị đem lại **+2.7 điểm** trên đặc trưng đóng băng của
CUB-200.

Cảnh báo kèm theo: `build_cifar_transform` ([datasets.py:397](../datasets.py#L397))
**có** gọi `Normalize` với thống kê CIFAR. Chồng thêm phép chuẩn hóa lên đó làm
CIFAR sập xuống 50.42. Vì vậy có hai chế độ riêng: `half` và `cifar_half`.

### 7.3. Lệnh chạy nền tự tìm thấy chính nó

Vòng lặp chờ GPU ban đầu viết là `pgrep -f "########## SURVEY"` đặt ngay trong
chuỗi `bash -c`. Nhưng `pgrep -f` so khớp với **toàn bộ dòng lệnh** của mọi tiến
trình — và dòng lệnh của chính tiến trình đó chứa nguyên văn chuỗi đang tìm. Nó
tự tìm thấy mình và chờ vĩnh viễn.

Cách sửa: đưa mẫu tìm kiếm vào một file script riêng, để nó không còn nằm trên
dòng lệnh của bên gọi. Xem
[training_scripts/wait_for_gpu.sh](../training_scripts/wait_for_gpu.sh).

### 7.4. `grep` tự đọc và ghi cùng một file

Có lần lệnh viết dạng `nohup bash -c '...; grep ... ~/cost.log' > ~/cost.log`.
`grep` vừa đọc vừa ghi cùng một file, sinh vòng lặp vô hạn và **lấp đầy ổ đĩa đến
99 %**. Quy tắc rút ra: không bao giờ grep chính file mà lệnh đang ghi vào.

---

## 8. Kết quả và cách đọc chúng

### 8.1. Sáu chỉ số nghĩa là gì

| Chỉ số | Nghĩa | Chiều tốt |
|---|---|---|
| **Acc@task** | Tỉ lệ ảnh được gán **đúng tác vụ** (chất lượng định tuyến) | cao |
| **Acc@1** | Tỉ lệ ảnh được gán **đúng lớp** — chỉ số quan trọng nhất | cao |
| **Acc@5** | Tỉ lệ ảnh có lớp đúng nằm trong 5 lớp dẫn đầu | cao |
| **Loss** | Hàm mất mát cross-entropy; đo cả độ chắc chắn chứ không chỉ đúng/sai | thấp |
| **Forgetting** | Trung bình mức tụt so với **đỉnh của chính tác vụ đó** | thấp |
| **Backward** | Trung bình chênh lệch giữa độ chính xác cuối và độ chính xác ngay sau khi học | cao |

Hai chỉ số cuối cần đọc cẩn thận: chúng so một tác vụ **với chính nó ở thời điểm
trước**, chứ không so giữa các phương pháp. Đặc điểm này sẽ quay lại ở mục 10.

### 8.2. Bảng kết quả, bốn seed

| Bộ dữ liệu | Chỉ số | HRM-PET | Đề xuất | Thay đổi |
|---|---|---|---|---|
| **ImageNet-R** | Acc@task | 77.58 ± 0.33 | 79.85 ± 0.18 | +2.264 ± 0.158 |
| | Acc@1 | 73.94 ± 0.48 | 75.18 ± 0.24 | +1.236 ± 0.416 |
| | Acc@5 | 86.71 ± 0.21 | 88.02 ± 0.17 | +1.312 ± 0.148 |
| | Loss | 1.23 ± 0.01 | 1.19 ± 0.01 | −0.040 ± 0.002 |
| | Forgetting | 3.36 ± 0.56 | 3.30 ± 0.35 | −0.056 ± 0.334 |
| | Backward | −3.16 ± 0.61 | −3.20 ± 0.42 | −0.047 ± 0.282 |
| **CIFAR-100** | Acc@task | 89.77 ± 0.11 | 90.87 ± 0.09 | +1.103 ± 0.066 |
| | Acc@1 | 89.78 ± 0.03 | 90.47 ± 0.05 | +0.690 ± 0.063 |
| | Acc@5 | 98.65 ± 0.03 | 98.86 ± 0.07 | +0.217 ± 0.057 |
| | Loss | 0.39 ± 0.00 | 0.39 ± 0.00 | −0.000 ± 0.001 |
| | Forgetting | 3.89 ± 0.08 | 3.74 ± 0.12 | −0.156 ± 0.098 |
| | Backward | −3.87 ± 0.07 | −3.73 ± 0.11 | +0.142 ± 0.097 |
| **CUB-200** | Acc@task | 93.12 ± 0.14 | 94.34 ± 0.10 | +1.226 ± 0.149 |
| | Acc@1 | 86.38 ± 0.25 | 87.88 ± 0.27 | +1.495 ± 0.353 |
| | Acc@5 | 97.04 ± 0.10 | 97.44 ± 0.13 | +0.400 ± 0.103 |
| | Loss | 0.57 ± 0.00 | 0.53 ± 0.01 | −0.036 ± 0.004 |
| | Forgetting | 2.54 ± 0.35 | 2.18 ± 0.12 | −0.359 ± 0.325 |
| | Backward | −2.42 ± 0.39 | −2.14 ± 0.13 | +0.279 ± 0.391 |

### 8.3. Cách đọc cột "Thay đổi"

Cột này **không phải** hiệu của hai cột bên trái. Nó được tính theo cặp: với từng
seed, lấy kết quả đề xuất trừ kết quả baseline, rồi mới lấy trung bình của bốn
hiệu đó.

Khác biệt nằm ở độ lệch chuẩn. Có seed dễ có seed khó; nếu lấy hiệu của hai trung
bình, độ biến thiên do độ khó của seed sẽ trộn lẫn vào. Tính theo cặp thì độ khó
của seed triệt tiêu, và độ lệch chuẩn còn lại **chỉ phản ánh độ ổn định của chính
mức cải thiện**.

Quy tắc đọc:

- **Độ lệch chuẩn nhỏ hơn trung bình nhiều lần** → cải thiện thật, lặp lại được.
  Ví dụ Acc@task trên CIFAR-100: +1.103 ± 0.066, tức là tỉ lệ 17 lần.
- **Độ lệch chuẩn lớn hơn trung bình** → không kết luận được gì.
  Ví dụ Backward trên ImageNet-R: −0.047 ± 0.282, tức là độ lệch gấp sáu lần trị
  tuyệt đối của trung bình. Đây là ô duy nhất không dương trong toàn bộ 18 phép
  so sánh, **và nó không phải suy giảm — nó là nhiễu**.

### 8.4. Chi phí

| Bộ dữ liệu | HRM-PET | Đề xuất | Tăng thêm |
|---|---|---|---|
| ImageNet-R | 1.83 | 2.85 | +1.02 |
| CIFAR-100 | 1.40 | 2.41 | +1.01 |
| CUB-200 | 1.60 | 2.60 | +1.00 |

Đơn vị là số lượt gọi LoRA trung bình cho mỗi ảnh. Mức tăng đúng bằng **một lượt**
— chính là lượt chạy cho đầu RP. Tầng hợp nhất phân lớp không tốn thêm gì.

Bộ nhớ tăng thêm: một ma trận Gram, một ma trận tương quan lớp, một seed.

---

## 9. Cách chạy lại toàn bộ

### 9.1. Cấu hình đã chốt

```
RP_SOURCE    = lora     đặc trưng lấy từ LoRA, không phải mạng đóng băng
RP_INORM     = none     không đổi khoảng giá trị đầu vào
RP_FUSE      = 1        bật hợp nhất định tuyến
RP_FUSE_DRM  = 1        tiêm kết quả vào luồng DRM/CRM
RP_FUSE_W    = 0.7      trọng số trên TII
RP_CLS_W     = 0.5      trọng số danh nghĩa của hợp nhất phân lớp
RP_CLS_GATE  = margin   cơ chế cổng theo biên top-2
RP_RAMP      = 0.0      tắt cơ chế tăng dần (xem mục 10.6)
CALIBRATE    = 0        không hiệu chỉnh nhiệt độ
RP_DIM       = 10000
RP_LAMBDA    = 10000
```

### 9.2. Chạy đánh giá cho một seed

```bash
cd ~/Documents/truongnguyen/Hybrid_ReMatching && nohup bash -c 'bash training_scripts/wait_for_gpu.sh 12000 120; RP_SOURCE=lora RP_INORM=none RP_FUSE=1 RP_FUSE_DRM=1 RP_FUSE_W=0.7 RP_CLS_W=0.5 RP_CLS_GATE=margin CALIBRATE=0 RP_DIM=10000 RP_LAMBDA=10000 bash training_scripts/rollout_hybrid_4090.sh 42' > ~/run.log 2>&1 & tail -F --pid=$! ~/run.log
```

### 9.3. Lấy bảng kết quả

```bash
cd ~/Documents/truongnguyen/Hybrid_ReMatching && ./.venv/bin/python training_scripts/summarize_hybrid_multiseed.py ~/Documents/truongnguyen/hrm-pet-output "_eval_rp_lora_*f1d1w0p7*cw0p5sh1p0m1gmargin.log"
```

### 9.4. File nào làm gì

| File | Vai trò |
|---|---|
| [engines/random_projection_head.py](../engines/random_projection_head.py) | Toàn bộ đầu RP: chiếu, tích lũy, giải ridge |
| [engines/hrm_lora_wtp_and_tap_engine.py](../engines/hrm_lora_wtp_and_tap_engine.py) | Hai tầng hợp nhất và điểm tiêm vào HRM-PET |
| [trainers/lora_trainer.py](../trainers/lora_trainer.py) | Gọi tích lũy và giải đầu RP sau mỗi tác vụ |
| [configs/imr_lora.py](../configs/imr_lora.py) | Khai báo cờ (ImageNet-R và CUB-200) |
| [configs/cifar100_lora.py](../configs/cifar100_lora.py) | Khai báo cờ (CIFAR-100) |
| [training_scripts/eval_rp_head_any_4090.sh](../training_scripts/eval_rp_head_any_4090.sh) | Chạy đánh giá cho một bộ dữ liệu |
| [training_scripts/rollout_hybrid_4090.sh](../training_scripts/rollout_hybrid_4090.sh) | Chạy cả ba bộ dữ liệu cho một seed |
| [training_scripts/summarize_hybrid_multiseed.py](../training_scripts/summarize_hybrid_multiseed.py) | Bảng trung bình ± độ lệch chuẩn |
| [training_scripts/incremental_accuracy.py](../training_scripts/incremental_accuracy.py) | Độ chính xác theo từng giai đoạn |
| [training_scripts/wait_for_gpu.sh](../training_scripts/wait_for_gpu.sh) | Chờ GPU trống rồi mới chạy |

---

## 10. Những hướng đã thử và đã đóng

Phần này quan trọng không kém phần thành công. Ghi lại để không ai mất công làm
lại.

### 10.1. Re-matching vét cạn

Thử mọi LoRA cho mọi ảnh rồi chọn kết quả tốt nhất. Kết quả **không vượt được
baseline**, dù tốn gấp 10 lần. Nguyên nhân: đầu phân lớp dùng chung khiến LoRA
của tác vụ khác vẫn có thể tạo ra điểm số cao cho lớp không thuộc nó — hiện tượng
"chiếm quyền" giữa các tác vụ.

### 10.2. Tái chấm điểm bằng Gaussian / Mahalanobis

Mô hình hóa đặc trưng 768 chiều của mỗi tác vụ bằng một Gaussian rồi định tuyến
theo khoảng cách Mahalanobis. Định tuyến rơi xuống **84.3 so với 93.21**. Thử cả
hiệp phương sai riêng lẫn hiệp phương sai chung, đều hỏng.

Nguyên nhân cấu trúc: LoRA gắn trên mạng đóng băng làm đặc trưng thay đổi *quá
ít* để tạo được tín hiệu phân biệt trong ngoài phân bố.

### 10.3. Tăng hạng LoRA từ 8 lên 16

Acc@1 gần như không đổi (86.46 so với 86.53). Kết luận: **đây không phải vấn đề
dung lượng mô hình**, nên tăng tham số không giúp gì.

### 10.4. Đầu RP dùng một mình, bỏ hẳn định tuyến

Thắng trên CUB-200 (87.96 so với 86.53) nhưng thua nặng trên ImageNet-R (70.08 so
với 74.02). Lý do rất rõ: CUB-200 là ảnh chim, mà ImageNet-21K vốn giàu lớp chim,
nên đặc trưng đóng băng đã đủ tốt. ImageNet-R là ảnh phong cách hóa, xa phân bố
tiền huấn luyện, nên **bắt buộc phải có thích nghi theo tác vụ** — và đó chính là
thứ các LoRA của HRM-PET cung cấp.

Bài học: hai bên bổ trợ nhau, không thay thế nhau.

### 10.5. Thống kê theo tầng, lấy từ bài báo PMI-CFS

Cô hướng dẫn gợi ý áp dụng ý tưởng từ bài PMI-CFS. Đọc kỹ cả phụ lục, chỉ có
**một** thành phần khả dụng: thống kê kích hoạt theo tầng làm bộ định tuyến thứ
ba. Đã cài ở [engines/layer_stat_router.py](../engines/layer_stat_router.py)
nhưng chưa chạy hoàn chỉnh; hướng này đã dừng theo yêu cầu.

Ba thành phần còn lại không áp dụng được: PMI cần sinh ảnh giả (HRM-PET phát lại
ở không gian đặc trưng nên không cần), CFS đã có sẵn trong repo và cho kết quả
*thấp hơn* baseline, còn phép chiếu ngữ nghĩa cần bộ mã hóa văn bản CLIP mà kiến
trúc của ta không có.

### 10.6. Trọng số tăng dần theo giai đoạn — đã cài, đã đo, đã loại

Đây là ví dụ đáng học nhất về việc **một cải tiến có vẻ hoàn hảo lại phải bị loại**.

Xuất phát từ một quan sát đúng: Forgetting và Backward gần như không nhúc nhích.
Lý do là cả hai đều so một tác vụ với đỉnh của chính nó, nên một hiệu chỉnh áp
dụng đồng đều ở mọi giai đoạn sẽ **triệt tiêu khỏi cả hai**.

Giải pháp thử nghiệm: cho trọng số trộn tăng dần theo số tác vụ đã thấy, dạng
`(số tác vụ đã thấy / T)^γ`. Vì hệ số này đạt 1.0 ở giai đoạn cuối nên hàng kết
quả cuối cùng **không đổi một chữ số nào**, chỉ hai chỉ số ma trận mới thay đổi.

Và nó chạy đúng như thiết kế. Với γ = 2, Forgetting cải thiện lên −0.415 / −0.433
/ −0.462 và Backward lên +0.357 / +0.411 / +0.462 trên ba bộ dữ liệu.

**Nhưng chính điều đó tố cáo nó.** Nếu trạng thái cuối không đổi mà hai chỉ số
so-với-đỉnh lại đẹp lên, thì mức đẹp đó chỉ có thể đến từ việc mô hình **kém đi ở
các giai đoạn giữa**. Đo độ chính xác trung bình qua cả mười giai đoạn xác nhận
đúng như vậy: trên CUB-200 nó tụt từ 88.35 xuống 88.16 rồi 87.91.

Cải thiện chỉ số ghi nhớ bằng cách hạ thấp đường đi thì không phải cải tiến, và
phản biện sẽ chỉ ra ngay. Cơ chế vẫn còn trong mã (`--rp_fusion_ramp`) làm dòng
ablation, nhưng **mặc định tắt**.

Hai điều có giá trị vẫn thu được:

1. Hợp nhất phân lớp **thực sự** gây hại ở giai đoạn đầu — ImageNet-R chỉ đạt
   86.0 so với 87.5 của baseline ở tác vụ đầu tiên. Đúng như dự đoán rằng ước
   lượng ma trận Gram cần đủ số lớp mới đáng tin.
2. Nhưng cùng đầu RP đó lại là **bộ định tuyến mạnh ngay từ đầu**. Nên giảm trọng
   số định tuyến ở giai đoạn sớm chỉ làm mất độ chính xác định tuyến mà không
   được gì bù lại.

### 10.7. Bài học về phương pháp

Trước khi chạy thí nghiệm ramp, chúng ta **ghi trước ba điều kiện nghiệm thu**.
Kết quả: một điều kiện đạt, một điều kiện trượt, và một điều kiện hóa ra không
phải phép thử (chỉ số Backward trên seed 42 vốn đã dương sẵn nên không kiểm
chứng được gì).

Nếu không ghi trước, thí nghiệm này sẽ trông như một thắng lợi sạch trên mọi chỉ
số của bảng tổng kết. Việc ghi trước là thứ đã bắt được nó.

---

## Trạng thái hiện tại

- Cấu hình đã chốt, đủ bốn seed trên cả ba bộ dữ liệu, **17 trong 18 phép so sánh
  cải thiện**.
- Báo cáo LaTeX ở [reports/bao_cao_fusion.tex](../reports/bao_cao_fusion.tex).
- Việc còn lại đáng làm nhất: bổ sung **ImageNet-A** làm bộ dữ liệu thứ tư. Phần
  mã đã sẵn sàng — lớp nạp dữ liệu ở
  [continual_datasets/continual_datasets.py:712](../continual_datasets/continual_datasets.py#L712)
  và cấu hình ở [configs/ima_lora.py](../configs/ima_lora.py). Thứ duy nhất còn
  thiếu là bản thân dữ liệu, phải tải thủ công về `datasets/imagenet-a/` vì bộ
  này không tải tự động được.
