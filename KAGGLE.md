# Running HRM-PET on Kaggle

This repository can run in a Kaggle notebook with a GPU accelerator. The important Kaggle rule is that inputs under `/kaggle/input` are read-only, while the ImageNet-R loader creates `train/` and `test/` folders. Copy ImageNet-R to `/kaggle/working` before training.

## 1. Clone repo

```bash
%%bash
set -e

cd /kaggle/working
rm -rf Hybrid_ReMatching
git clone https://github.com/Truong140206/Hybrid_ReMatching.git
cd Hybrid_ReMatching
ls
```

## 2. Install dependencies

```bash
%%bash
set -e

cd /kaggle/working/Hybrid_ReMatching
pip install -q timm==0.6.7 scikit-learn scipy requests
```

## 3. Copy ImageNet-R to writable storage

The observed Kaggle input layout is:

```text
/kaggle/input/datasets/my1nonly/imagenet-r/imagenet-r
```

Copy the parent folder to `/kaggle/working`:

```bash
%%bash
set -e

rm -rf /kaggle/working/datasets
mkdir -p /kaggle/working/datasets
cp -r /kaggle/input/datasets/my1nonly/imagenet-r /kaggle/working/datasets/

ls /kaggle/working/datasets/imagenet-r
ls /kaggle/working/datasets/imagenet-r/imagenet-r | head
```

Use this runtime data path:

```text
/kaggle/working/datasets/imagenet-r
```

## 4. Smoke test stage 1 only

```bash
%%bash
set -e

cd /kaggle/working/Hybrid_ReMatching

python -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port=29500 \
  main.py \
  imr_hideprompt_5e \
  --model vit_base_patch16_224 \
  --original_model vit_base_patch16_224 \
  --batch-size 8 \
  --epochs 1 \
  --data-path /kaggle/working/datasets/imagenet-r \
  --lr 0.0005 \
  --ca_lr 0.005 \
  --crct_epochs 1 \
  --seed 42 \
  --train_inference_task_only \
  --num_workers 2 \
  --output_dir /kaggle/working/hrm-pet-output/debug_tii
```

## 5. Full two-stage run

The launcher copies ImageNet-R from `INPUT_DATA_PATH` to `DATA_PATH` if `DATA_PATH` does not exist yet.

```bash
%%bash
set -e

cd /kaggle/working/Hybrid_ReMatching

export INPUT_DATA_PATH="/kaggle/input/datasets/my1nonly/imagenet-r"
export DATA_PATH="/kaggle/working/datasets/imagenet-r"
export OUTPUT_ROOT="/kaggle/working/hrm-pet-output"
export GPUS=1
export SEED=42
export TII_EPOCHS=20
export LORA_EPOCHS=50
export CRCT_EPOCHS=30
export TII_BATCH_SIZE=64
export LORA_BATCH_SIZE=24

bash training_scripts/kaggle_train_imr_lora_sup21k.sh
```

For a quick end-to-end check, reduce epochs:

The launcher skips stage 1 when all TII checkpoints already exist, so a failed stage 2 run does not force you to retrain stage 1. Set SKIP_TII_IF_COMPLETE=0 only if you intentionally want to rebuild stage 1.


```bash
TII_EPOCHS=1 LORA_EPOCHS=1 CRCT_EPOCHS=1 bash training_scripts/kaggle_train_imr_lora_sup21k.sh
```

## Notes

- If your Kaggle dataset is mounted somewhere else, set `INPUT_DATA_PATH` to the folder that contains the inner `imagenet-r` class-folder directory.
- Outputs are written to `/kaggle/working/hrm-pet-output` by default.
- Use `GPUS=1` for a single Kaggle GPU.

