# Kết quả audit Arrow-LoRA trên ImageNet-R

## Kết quả

- `WinnerRecall@4` của TII: `88.8299`.
- `ArrowRecall@4`: `38.7908`.
- `UnionRecall@2x2`: `81.6861`.
- `UnionLoRA/sample`: `3.6417`.
- Mức đồng ý top-1 giữa TII và Arrow: `9.3378`.
- Gain của union so với TII top-4: `-7.1438` điểm.
- `ARROW_AUDIT_GATE=FAIL`.

## Kết luận

Arrow rank-1 không phù hợp để định tuyến HideLoRA của HRM-PET. Chữ ký singular
trội gần như không tương quan với LoRA thắng cuộc; thêm nó còn loại mất các ứng
viên tốt của TII. Không tuning threshold, pooling hoặc trọng số trên test.

Thí nghiệm tiếp theo chỉ kiểm tra một nguyên nhân cấu trúc cụ thể: LoRA đang có
rank 8 nhưng Arrow nén mỗi cập nhật xuống một hướng. Audit full-rank response sẽ
dùng toàn bộ đáp ứng K/V chuẩn hóa `||xAB|| / ||AB||_F`. Nếu cách này vẫn không
tăng recall union ít nhất 3 điểm so với TII top-4, toàn bộ hướng định tuyến theo
tham số LoRA sẽ bị loại bỏ.
