#!/usr/bin/env bash
set -euo pipefail

SEED="${SEED:-42}"
INPUT_DATA_PATH="${INPUT_DATA_PATH:-/kaggle/input/datasets/my1nonly/imagenet-r}"
DATA_PATH="${DATA_PATH:-/kaggle/working/datasets/imagenet-r}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working/hrm-pet-output}"
GPUS="${GPUS:-1}"

if [ ! -d "${DATA_PATH}" ]; then
  if [ ! -d "${INPUT_DATA_PATH}" ]; then
    echo "ImageNet-R input not found: ${INPUT_DATA_PATH}" >&2
    echo "Set INPUT_DATA_PATH to the Kaggle input folder that contains imagenet-r." >&2
    exit 1
  fi
  mkdir -p "$(dirname "${DATA_PATH}")"
  cp -r "${INPUT_DATA_PATH}" "${DATA_PATH}"
fi

mkdir -p "${OUTPUT_ROOT}"

python -m torch.distributed.run \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT:-29500}" \
  main.py \
  imr_hideprompt_5e \
  --model vit_base_patch16_224 \
  --original_model vit_base_patch16_224 \
  --batch-size "${TII_BATCH_SIZE:-64}" \
  --ca_storage_efficient_method covariance \
  --epochs "${TII_EPOCHS:-20}" \
  --data-path "${DATA_PATH}" \
  --lr 0.0005 \
  --ca_lr 0.005 \
  --crct_epochs 30 \
  --seed "${SEED}" \
  --train_inference_task_only \
  --output_dir "${OUTPUT_ROOT}/imr_vit_multi_centroid_mlp_2_seed${SEED}"

python -m torch.distributed.run \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT_LORA:-29513}" \
  main.py \
  imr_lora \
  --model vit_base_patch16_224 \
  --original_model vit_base_patch16_224 \
  --batch-size "${LORA_BATCH_SIZE:-24}" \
  --epochs "${LORA_EPOCHS:-50}" \
  --data-path "${DATA_PATH}" \
  --ca_lr 0.005 \
  --crct_epochs 30 \
  --seed "${SEED}" \
  --lr 0.03 \
  --con 0.2 \
  --lora_rank 5 \
  --En gen \
  --tau -10 \
  --K 5 \
  --sched cosine \
  --dataset Split-Imagenet-R \
  --lora_momentum 0.4 \
  --lora_type hide \
  --trained_original_model "${OUTPUT_ROOT}/imr_vit_multi_centroid_mlp_2_seed${SEED}" \
  --output_dir "${OUTPUT_ROOT}/test_imr_sup21k_lora_pe_seed${SEED}"
