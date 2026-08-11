# Hybrid Real-Feature Replay cho ImageNet-R

## 1. Van de can xu ly

Ket qua hien tai cho thay CFS chi cai thien nhe. Nguyen nhan la CFS van chon mau tu
phan phoi Gaussian xap xi. Tren ImageNet-R, feature cua mot lop co nhieu mode va
khong hoan toan Gaussian, nen classifier correction co the hoc rat tot tren feature
gia nhung van lech khoi feature that. Tang so epoch CRCT qua nhieu con lam forgetting
tang do classifier bi over-correction.

## 2. Thay doi moi

Phuong phap moi giu mot bo nho nho gom feature that, khong giu anh goc:

1. Sau khi hoc moi lop, trich xuat feature that bang LoRA cua task do.
2. Loai phan duoi xa tam lop de tranh giu outlier.
3. Chon toi da 48 feature/lop theo tieu chi vua gan tam vua bao phu nhieu huong.
4. Moi epoch CRCT, moi lop co cung mot ngan sach 120 mau:
   - lop cu: 35% feature that va 65% feature Gaussian+CFS;
   - lop cua task hien tai: 10% feature that va 90% feature Gaussian+CFS;
   - tong so mau cua moi lop van bang nhau.
5. Mot nua phan feature that duoc chon theo hard mining: cac feature co margin cua
   nhan dung thap nhat duoi classifier hien tai. Phan con lai duoc lay ngau nhien de
   tranh chi hoc cac truong hop kho.
6. Voi multi-centroid, ngan sach feature gia duoc chia deu cho cac centroid thay vi
   moi centroid sinh ca mot batch rieng. Dieu nay kiem soat so update CRCT.

## 3. Vi sao co kha nang cai thien

- Feature that neo bien quyet dinh vao phan phoi du lieu that, giam Gaussian bias.
- Hard mining danh truc tiep vao cac mau lop cu dang bi quen.
- CFS van giu vai tro mo rong do bao phu va bo sung mau gan bien.
- Ngan sach can bang theo lop tranh lop co nhieu centroid tao qua nhieu gradient.
- So epoch co the tang ma khong lam so update tang gap 10 lan nhu full multi-centroid.

## 4. Chi phi bo nho va tinh cong bang

Voi ViT-B/16 co feature 768 chieu, 48 feature/lop, 200 lop va FP16, bo nho toi da
xap xi 14.1 MiB. Phuong phap khong luu anh, nhung khong con la memory-free tuyet doi.
Bao cao can goi day la `low-memory feature replay` va ghi ro ngan sach bo nho.

## 5. Cau hinh full dau tien

- TII: dung lai checkpoint seed 42 hien co.
- LoRA: 50 epoch.
- CRCT: 30 epoch, 120 mau/lop/epoch.
- Real replay ratio: 0.35 cho lop cu, 0.10 cho lop hien tai.
- Memory: 48 feature/lop.
- Hard real ratio: 0.50.
- CFS boundary ratio: 0.10.
- Semantic va task-energy routing: tat de do rieng tac dong cua phuong phap moi.

## 6. Cach danh gia

So sanh truc tiep voi baseline ImageNet-R seed 42:

`Acc@task 77.3007, Acc@1 73.8379, Acc@5 86.0767, Loss 1.2399, Forgetting 3.5268, Backward -3.1815`

Va ban CFS tot nhat hien tai:

`Acc@task 77.7378, Acc@1 73.8498, Acc@5 86.2814, Loss 1.2488, Forgetting 3.3421, Backward -3.0883`

Muc tieu cua run dau tien la tang Acc@1 ro rang dong thoi khong lam Forgetting xau
hon 3.3421. Mot run chi dung de chon cau hinh; neu co ket qua tot can lap lai it nhat
3 seed LoRA/CRCT va ghi ro TII dang co dinh seed 42.
