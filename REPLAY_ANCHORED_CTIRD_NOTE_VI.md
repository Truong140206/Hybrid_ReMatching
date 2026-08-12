# Replay-Anchored CTIRD tren ImageNet-R

## 1. Muc tieu

Phuong phap moi ket hop hai y tuong dung vai tro cua chung:

- HRM-PET: CTIRD bao toan quan he giua cac mau thay vi ep feature/logit cu.
- PMI-CFS: CFS chon feature muc tieu da dang, sau do model inversion tao anh gia de replay.

Muc tieu la lam LoRA cua task moi van bieu dien duoc anh thuoc task cu, qua do giam forgetting va giam hau qua khi routing chon nham LoRA. Phuong phap khong them lan chay LoRA nao khi suy luan.

## 2. Thay doi so voi ban truoc

Ban truoc chi dung feature Gaussian/CFS trong CRCT de hieu chinh classifier. Feature cu khong di qua LoRA dang hoc, nen tac dong truc tiep den forgetting cua representation con han che.

Ban moi thuc hien:

1. Sau khi hoc xong mot task, tinh mean/variance va huan luyen CFS cho tung lop.
2. CFS chon 5 feature muc tieu moi lop.
3. Layer-specific warm start va full-model inversion tao anh gia 224x224.
4. Anh duoc luu dang `uint8` trong `replay_anchor_cache`, moi lop chi sinh mot lan.
5. Khi hoc task sau, old LoRA xu ly anh gia de tao quan he teacher.
6. Current LoRA xu ly cung anh va toi thieu hoa KL giua hai ma tran quan he.
7. Chi anh ma old LoRA phan loai dung va dat nguong confidence moi duoc dung.
8. Moi lop dao 10 candidate va giu 5 anh co do tin cay cao nhat; quan he tu-than tren duong cheo khong tham gia loss.

Loss moi:

```text
L = CE(new data) + lambda_CT * CTIRD(new data)
                 + lambda_replay * RelationKD(pseudo old data)
```

Khong dung cross-entropy tren anh gia cu trong buoc huan luyen LoRA. Lua chon nay tranh ep current LoRA ghi nho cung nhac logit cu va giu plasticity cho task moi.

## 3. Cau hinh thi nghiem chinh

- Backbone: ViT-B/16.
- LoRA rank: 8, theo phan phan tich tham so cua HRM-PET tren ImageNet-R.
- CFS: MLP hai tang, 512 hidden, 200 epoch, selection ratio 0.5, 5 buoc.
- Bo nho: 5 anh gia/lop, toi da 1000 anh cho 200 lop.
- Inversion: 200 buoc layer-specific va 600 buoc full-model.
- Replay relation weight ban dau: 0.05.
- Semantic, prototype va exhaustive rematching deu tat trong thi nghiem nay.

## 4. Cach chay

Smoke test hai task:

```bash
cd ~/Documents/truongnguyen/Hybrid_ReMatching
git pull --ff-only
bash training_scripts/run_imagenet_r_rank8_replay_anchor_4090.sh smoke
```

Rank-8 baseline:

```bash
cd ~/Documents/truongnguyen/Hybrid_ReMatching
git pull --ff-only
bash training_scripts/run_imagenet_r_rank8_replay_anchor_4090.sh baseline
```

Ban cai tien day du:

```bash
cd ~/Documents/truongnguyen/Hybrid_ReMatching
git pull --ff-only
bash training_scripts/run_imagenet_r_rank8_replay_anchor_4090.sh full
```

Script tu kiem tra 10 TII checkpoint, tu choi ghi de checkpoint/log cu, kiem tra cache anh va in dong ket qua task cuoi khi ket thuc.

## 5. Tieu chi chap nhan

Chi coi phuong phap co hieu qua khi so voi rank-8 baseline trong cung dieu kien:

- Acc@1 tang.
- Forgetting khong tang.
- Backward gan 0 hon hoac khong xau di.
- Khong dung exhaustive rematching khi bao cao ket qua chinh.
- Sau khi cau hinh seed 42 dat yeu cau, chay them nhieu seed de xac nhan.

## 6. Trang thai

Code da duoc trien khai va da qua kiem tra cu phap Python/Bash. Moi truong Windows hien tai khong co PyTorch, vi vay smoke test GPU phai duoc chay tren RTX 4090 truoc khi chay day du.
