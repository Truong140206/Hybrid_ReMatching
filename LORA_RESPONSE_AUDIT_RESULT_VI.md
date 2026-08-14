# Kết quả audit full-rank LoRA response

## Kết quả ImageNet-R

- `WinnerRecall@4` của TII: `88.8299`.
- `ResponseRecall@4`: `38.0790`.
- `ResponseUnionRecall@2x2`: `81.5903`.
- `ResponseUnionLoRA/sample`: `3.6178`.
- Mức đồng ý top-1 với TII: `9.9213`.
- Gain của union so với TII top-4: `-7.2396` điểm.
- `LORA_RESPONSE_AUDIT_GATE=FAIL`.

## Kết luận

Giữ toàn bộ rank 8 không cải thiện Arrow rank-1. Hai phép đo đều chỉ đạt khoảng
38% winner recall top-4 và có mức đồng ý với TII dưới 10%. Vì vậy nguyên nhân
không phải mất thông tin do SVD rank-1; bản thân hướng/đáp ứng tham số LoRA
không phản ánh adapter tạo bằng chứng phân loại tốt nhất sau toàn bộ backbone.

Đóng toàn bộ hướng parameter-only LoRA routing. Không thử thêm pooling,
threshold, trọng số hoặc union trên test.

## Bước tiếp theo

Giữ nguyên phép exhaustive và vector hóa nhiều task LoRA trong cùng một model
call. Cách này không giảm số phép toán lý thuyết nhưng có thể tăng throughput
GPU và giảm wall time mà không đánh đổi accuracy, loss hay forgetting. Gate
mới yêu cầu sáu metric cuối khớp exhaustive tuần tự trong tolerance cố định;
hiệu quả được báo bằng wall-time speedup thực đo.
