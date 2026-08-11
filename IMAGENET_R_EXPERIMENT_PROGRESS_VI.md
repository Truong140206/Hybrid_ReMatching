# Tiến độ thực nghiệm ImageNet-R

## 1. Thiết lập chung

- Dataset: Split-ImageNet-R, 200 lớp, 10 task, 20 lớp/task.
- Backbone: ViT-B/16.
- TII dùng chung: `imr_tii_original_10tasks_seed42`.
- Baseline và cải tiến trong mỗi lần chạy dùng cùng TII và cùng split dữ liệu.
- Cấu hình cải tiến được giữ lại:
  - legacy CTIRD;
  - CFS sampling;
  - full replay;
  - boundary replay ratio `0.10`;
  - CRCT `4` epoch;
  - CRCT learning rate `0.005`;
  - task-energy và semantic đều tắt.

Lưu ý: seed 0 hiện chỉ thay seed của giai đoạn LoRA/CRCT, còn TII vẫn cố định ở seed 42. Vì vậy đây là lặp lại giai đoạn 2 dưới TII cố định, chưa phải hai seed độc lập hoàn toàn của toàn pipeline.

## 2. Kết quả từng seed

| Seed LoRA/CRCT | Phương pháp | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | HRM-PET baseline | 77.3007 | 73.8379 | 86.0767 | 1.2399 | 3.5268 | -3.1815 |
| 42 | CFS + full replay + boundary 0.10 + CRCT4 | 77.7378 | 73.8498 | 86.2814 | 1.2488 | 3.3421 | -3.0883 |
| 0 | HRM-PET baseline | 78.3613 | 73.6062 | 86.4495 | 1.2267 | 3.4859 | -3.3020 |
| 0 | CFS + full replay + boundary 0.10 + CRCT4 | 77.9987 | 74.1049 | 86.3983 | 1.2489 | 3.3086 | -3.1867 |

## 3. Chênh lệch cải tiến trừ baseline

| Seed | Delta Acc@task | Delta Acc@1 | Delta Acc@5 | Delta Loss | Delta Forgetting | Delta Backward |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | +0.4371 | +0.0119 | +0.2047 | +0.0089 | -0.1847 | +0.0932 |
| 0 | -0.3626 | +0.4987 | -0.0512 | +0.0222 | -0.1773 | +0.1153 |

Trong cả hai seed, Acc@1 tăng, Forgetting giảm và Backward gần 0 hơn. Acc@task và Acc@5 dao động theo seed. Loss tăng nhẹ.

## 4. Trung bình hai seed

| Phương pháp | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward |
|---|---:|---:|---:|---:|---:|---:|
| HRM-PET baseline | 77.8310 | 73.7221 | 86.2631 | 1.2333 | 3.5064 | -3.2418 |
| Bản cải tiến | 77.8683 | 73.9774 | 86.3399 | 1.2489 | 3.3254 | -3.1375 |
| Cải tiến - baseline | +0.0373 | +0.2553 | +0.0768 | +0.0156 | -0.1810 | +0.1043 |

Kết luận tạm thời trên hai lần chạy giai đoạn 2:

- Acc@1 trung bình tăng `0.2553` điểm.
- Acc@task trung bình tăng `0.0373` điểm.
- Acc@5 trung bình tăng `0.0768` điểm.
- Forgetting trung bình giảm `0.1810` điểm, khoảng `5.16%` so với baseline.
- Backward tăng `0.1043` điểm, tức gần 0 hơn.
- Loss tăng `0.0156`; đây là chỉ số duy nhất đi theo hướng chưa tốt.

## 5. Các ablation quan trọng đã thực hiện

| Cấu hình | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward | Kết luận |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline seed42 | 77.3007 | 73.8379 | 86.0767 | 1.2399 | 3.5268 | -3.1815 | Mốc chính |
| CFS + task-energy + full replay CRCT30 | 77.3976 | 73.7220 | 86.2722 | 1.3896 | 4.2201 | -4.2201 | Correction quá mạnh; forgetting tăng |
| CFS + legacy CTIRD + full replay CRCT30 | 76.9255 | 73.1216 | 85.9195 | 1.4265 | 4.6460 | -4.5961 | Full replay 30 epoch không phù hợp |
| CFS, ngân sách batch như baseline | 77.0826 | 73.5323 | 85.8652 | 1.2502 | 3.3270 | -3.1276 | Retention tốt hơn, accuracy giảm nhẹ |
| CFS + full replay + boundary 0.10 + CRCT3 | 77.3035 | 73.7452 | 86.1310 | 1.2604 | 3.2234 | -3.0588 | Cân bằng tốt, Acc@1 còn thấp hơn baseline |
| CFS + full replay + boundary 0.10 + CRCT4 | 77.7378 | 73.8498 | 86.2814 | 1.2488 | 3.3421 | -3.0883 | Cấu hình tốt nhất seed42 |

## 6. Giải thích thay đổi CRCT

Với batch size 24, code sinh `24 x 5 = 120` pseudo-feature cho mỗi centroid. ImageNet-R dùng multi-centroid với tối đa 10 centroid/lớp. Khi có 200 lớp, full replay có thể tạo khoảng 2.000 batch correction mỗi epoch, trong khi baseline task 10 chỉ dùng khoảng 180 batch. Vì vậy full replay với 30 CRCT epoch tạo quá nhiều bước cập nhật và làm classifier bị over-correction.

Giảm CRCT xuống 3-4 epoch đưa tổng số update về gần ngân sách baseline. CRCT4 lấy lại accuracy tốt hơn CRCT3, trong khi Forgetting vẫn thấp hơn baseline.

## 7. Việc cần làm tiếp

1. Chạy thêm seed LoRA/CRCT dưới cùng TII để kiểm tra độ ổn định.
2. Nếu cần kết quả chuẩn publication, huấn luyện lại toàn bộ TII + LoRA/CRCT cho từng seed độc lập.
3. Báo cáo mean và độ lệch chuẩn khi có ít nhất 3 seed.
4. Không tiếp tục tối ưu riêng seed 42 để tránh chọn siêu tham số theo một lần chạy.
