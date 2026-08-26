# Bản lưu ngày 2026-08-25

Ba file `.tex` trong thư mục này là **bản đã dùng để làm báo cáo**, chụp lại
trước khi sửa theo phát hiện về `RP_SOURCE=original`. Giữ nguyên, không sửa.

| File | Nội dung |
|---|---|
| `giai-thich-he-thong.tex` | giải thích toàn hệ thống cho người mới, 1913 dòng |
| `tom-tat-ranpac.tex` | đọc hiểu bài báo RanPAC (arXiv 2307.02251) |
| `bao_cao_fusion.tex` | báo cáo kỹ thuật về hai tầng hợp nhất |

Cả ba đều dán thẳng lên Overleaf được (pdfLaTeX, không cần đổi gì).

## Vì sao có bản lưu này

Ngày 2026-08-25, chẩn đoán trên `~/diag.log` và `~/diag2.log` cho thấy cờ
`RP_SOURCE=original` **không** lấy đặc trưng từ mạng ViT tiền huấn luyện nguyên
vẹn như tài liệu mô tả. Nó lấy từ `original_model`, mà `original_model` được nạp
đè toàn bộ trọng số từ checkpoint TII (`trainers/lora_trainer.py:171`), tức một
ViT đã fine-tune trên chuỗi tác vụ và bị ghim ở tác vụ 1.

Hệ quả: mọi chỗ trong bản lưu này gọi 55.79 / 56.36 là "đặc trưng đóng băng
thuần" hay "mạng trần" đều **mô tả sai nguồn đặc trưng**, và phép đối chiếu với
mốc "No Phase 1" của RanPAC không cùng điều kiện.

Các con số đo được vẫn đúng; chỉ nhãn và cách diễn giải là sai. Kết quả chính
(bảng 4 seed, 17/18 ô cải thiện) **không bị ảnh hưởng**, vì đó là phép so nội bộ
giữa phương pháp và baseline trên cùng dữ liệu, cùng đường trích đặc trưng.

Bản đã sửa nằm ở `docs/` và `reports/` như thường lệ.

## Số liệu chẩn đoán, để tiện tra

Đầu RP trên đặc trưng của mô hình TII ghim ở tác vụ 1, seed 42:

| Bộ dữ liệu | Cờ tiền xử lý | Acc@1 | RanPAC No Phase 1 | Chênh |
|---|---|---|---|---|
| CUB-200 | `half` | 87.96 | ~90 | −2.0 |
| CIFAR-100 | `cifar_half` | 85.40 | 89.0 | −3.6 |
| CIFAR-100 | `none` (sai cờ) | 80.87 | 89.0 | −8.1 |
| ImageNet-R | `half` | 56.36 | 71.8 | −15.4 |

Bộ dữ liệu ImageNet-R đã kiểm: đủ 200 lớp, 24 000 ảnh train + 6 000 ảnh test.
Không thiếu gì.
