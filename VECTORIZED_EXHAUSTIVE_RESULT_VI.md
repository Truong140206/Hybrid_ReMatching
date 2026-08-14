# Kết quả exhaustive vector hóa trên ImageNet-R

## Tính tương đương

Với task chunk size 4, toàn bộ sáu metric khớp tuyệt đối exhaustive tuần tự:

- Acc@task: `81.1182`.
- Acc@1: `75.4277`.
- Acc@5: `88.9183`.
- Loss: `1.0860`.
- Forgetting: `3.0772`.
- Backward: `-2.9088`.
- `VECTORIZED_EQUIVALENCE_GATE=PASS`.

Số LoRA được đánh giá vẫn là `10.0` mỗi mẫu; số model call giảm từ 10 xuống 3.

## Chi phí thực tế

Wall-time speedup bảo thủ chỉ đạt `1.032x`, tương đương khoảng 3.2%. Kết quả
nhỏ vì vector hóa không loại bỏ FLOPs: mỗi model call xử lý batch lớn hơn và
GPU vẫn thực hiện gần đủ phép toán của 10 nhánh LoRA.

## Kết luận

Giữ implementation như một tối ưu kỹ thuật chính xác, nhưng không coi đây là
đóng góp hiệu quả tính toán chính. Muốn giảm chi phí đáng kể phải giảm số nhánh
LoRA thực sự được chạy hoặc thay đổi cách kết hợp adapter; chỉ gom batch là
không đủ.

Hướng parameter-only routing đã bị loại bỏ bởi cả Arrow rank-1 và full-rank
response. Hướng tiếp theo được xem xét là soft mixture: dùng posterior TII để
kết hợp có trọng số một số LoRA trong một forward, thay vì dự đoán cứng LoRA
thắng cuộc. Code hiện có chỉ chứa nhánh cộng ensemble với hệ số cố định 0.4 và
nhánh này không được dùng trong inference hiện tại.
