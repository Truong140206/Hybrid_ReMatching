# CFS-only Validation-Gated CRCT tren ImageNet-R

## Ly do thay doi

Thi nghiem CFS-PMI Replay-Anchored CTIRD lam Acc@1 giam tu 74.0191 xuong
73.5688. Nguyen nhan chinh la replay anh gia can thiep truc tiep vao LoRA task
moi, lam giam tinh chuyen mon hoa cua tung LoRA trong HRM-PET.

Huong moi khong dung anh gia va khong them loss vao LoRA. LoRA rank 8 duoc hoc
giong baseline. CFS chi tac dong vao CRCT, la buoc hieu chinh classifier bang
feature replay trong thiet ke goc.

## Cach hoat dong

Moi lop co ba nhom feature replay can bang:

1. Core: feature gan tam Gaussian, giu dai dien on dinh cua lop.
2. Diverse: CFS chon feature it trung lap, tang do phu cua phan bo.
3. Boundary: feature gan bien quyet dinh, nam trong vung mat do hop ly va uu
   tien phia van duoc classifier gan dung nhan.

CRCT chi cap nhat classifier head; fc_norm va tat ca LoRA duoc giu nguyen.

Sau CRCT, mot tap anchor doc lap duoc lay lai tu mean/variance cua moi lop.
Code thu 21 muc noi suy giua classifier truoc va sau CRCT. Mot muc chi duoc
chap nhan khi:

- macro accuracy tren tat ca lop khong giam;
- top-5 accuracy tren anchor khong giam;
- task-routing accuracy (lop top-1 anh xa ve task) khong giam;
- macro cross-entropy khong tang;
- accuracy cua tung lop cu rieng le khong giam tren anchor holdout.
Cac dieu kien duoc kiem tra tren ba lan lay anchor doc lap de giam nhieu.

Neu khong co muc nao dat dieu kien, alpha bang 0 va classifier tu dong rollback.
Gate nay khong dung tap test va khong bao dam test accuracy se tang, nhung ngan
CRCT chap nhan mot cap nhat xau ngay tren feature-statistic holdout.

## Cach chay

Pilot 3 task, tu so sanh voi log baseline rank 8:

```bash
cd ~/Documents/truongnguyen/Hybrid_ReMatching
git pull --ff-only
bash training_scripts/run_imagenet_r_rank8_cfs_crct_4090.sh pilot
```

Chi chay full neu pilot in `PILOT_GATE=PASS`:

```bash
cd ~/Documents/truongnguyen/Hybrid_ReMatching
git pull --ff-only
bash training_scripts/run_imagenet_r_rank8_cfs_crct_4090.sh full
```

Thiet nghiem chinh khong bat semantic, prototype, exhaustive rematching, replay
anh gia hay real-feature memory.

Pilot chi PASS khi Acc@task, Acc@1, Acc@5 va Backward khong giam, dong thoi
Loss va Forgetting khong tang so voi cung dong task cua baseline rank 8. Chi
can mot chi so di sai huong thi script tra ve `PILOT_GATE=FAIL`.
