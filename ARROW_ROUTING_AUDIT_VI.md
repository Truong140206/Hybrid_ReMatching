# Audit định tuyến LoRA theo tham số Arrow

## Vấn đề cần giải quyết

Exhaustive rematching cho kết quả ImageNet-R tốt nhất nhưng phải chạy toàn bộ
10 LoRA cho mỗi ảnh. Audit oracle trước đó cho thấy nếu biết đúng LoRA thắng
cuộc thì trung bình chỉ cần khoảng 3.1 LoRA/mẫu, nhưng ranking TII hiện tại chỉ
đưa LoRA thắng cuộc vào top-4 khoảng 89.07% số mẫu.

CFS đã được thử tại CRCT và TII. Cả hai ablation end-to-end đều không đạt gate,
do đó không tiếp tục ép CFS vào một vị trí không phù hợp.

## Ý tưởng được kiểm tra

Arrow biểu diễn mỗi adapter bằng hướng riêng trội của chính ma trận cập nhật
LoRA. Với quy ước nhân trong repository này, đầu vào attention là vector hàng
và cập nhật là `x @ (A @ B)`, nên chữ ký đầu vào là vector singular trái trội
của cập nhật. HideLoRA chỉ sửa K và V; audit lấy hướng trội chung của
`[A_k B_k, A_v B_v]` tại từng block.

Ảnh được chạy qua backbone không adapter đến các block có LoRA. Tại mỗi block,
đầu vào attention được so cosine tuyệt đối với chữ ký của từng task. Điểm task
là trung bình trên token và trên các block. Không train router, không dùng hoặc
lưu ảnh/feature task cũ, không thay checkpoint.

## Giao thức audit

Audit vẫn chạy exhaustive để biết LoRA thắng cuộc thật sự, nhưng không thay đổi
logits hay kết quả exhaustive. Nó báo:

- `WinnerRecall@4`: recall top-4 của ranking TII hiện tại.
- `ArrowRecall@4`: recall top-4 của ranking Arrow theo tham số.
- `UnionRecall@2x2`: recall của hợp TII top-2 và Arrow top-2.
- `UnionLoRA/sample`: số LoRA duy nhất trong tập hợp, luôn không quá 4.
- `TIIArrowTop1Agree`: mức hai ranking đồng ý ở vị trí đầu.

Gate được ấn định trước khi xem kết quả: chỉ triển khai inference chọn lọc nếu
`UnionRecall@2x2` cao hơn `WinnerRecall@4` ít nhất 3 điểm phần trăm và
`UnionLoRA/sample <= 4`. Nếu fail, hướng này bị loại bỏ, không tuning trên test.

## Chi phí

Audit tốn exhaustive cộng phần backbone không adapter đến hết các block LoRA.
Đây là chi phí chẩn đoán một lần. Nếu gate pass, phiên bản inference thật dự
kiến chạy phần định tuyến này và tối đa 4 LoRA thay vì 10; phiên bản đó chỉ được
triển khai sau audit.
