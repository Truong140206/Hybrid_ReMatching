#!/usr/bin/env bash
set -euo pipefail

SEED="${SEED:-42}"
INPUT_DATA_PATH="${INPUT_DATA_PATH:-/kaggle/input/datasets/my1nonly/imagenet-r}"
DATA_PATH="${DATA_PATH:-/kaggle/working/datasets/imagenet-r}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working/hrm-pet-output}"
GPUS="${GPUS:-1}"
NUM_TASKS="${NUM_TASKS:-10}"
CRCT_EPOCHS="${CRCT_EPOCHS:-30}"
SKIP_TII_IF_COMPLETE="${SKIP_TII_IF_COMPLETE:-1}"
SKIP_LORA_IF_COMPLETE="${SKIP_LORA_IF_COMPLETE:-0}"

TII_OUTPUT="${OUTPUT_ROOT}/imr_vit_multi_centroid_mlp_2_seed${SEED}"
LORA_OUTPUT="${OUTPUT_ROOT}/test_imr_sup21k_lora_pe_seed${SEED}"

require_checkpoints() {
  local run_dir="$1"
  local label="$2"
  local missing=0

  for task_id in $(seq 1 "${NUM_TASKS}"); do
    local checkpoint_path="${run_dir}/checkpoint/task${task_id}_checkpoint.pth"
    if [ ! -s "${checkpoint_path}" ]; then
      echo "Missing ${label} checkpoint: ${checkpoint_path}" >&2
      missing=1
    fi
  done

  if [ "${missing}" -ne 0 ]; then
    echo "${label} did not produce all ${NUM_TASKS} checkpoints." >&2
    return 1
  fi

  echo "${label} checkpoints OK: ${run_dir}/checkpoint/task1..task${NUM_TASKS}_checkpoint.pth"
}

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

if [ "${SKIP_TII_IF_COMPLETE}" = "1" ] && require_checkpoints "${TII_OUTPUT}" "Stage 1 TII"; then
  echo "Skipping stage 1 because all TII checkpoints already exist."
else
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
    --crct_epochs "${CRCT_EPOCHS}" \
    --seed "${SEED}" \
    --train_inference_task_only \
    --output_dir "${TII_OUTPUT}"

  require_checkpoints "${TII_OUTPUT}" "Stage 1 TII"
fi

if [ "${SKIP_LORA_IF_COMPLETE}" = "1" ] && require_checkpoints "${LORA_OUTPUT}" "Stage 2 LoRA"; then
  echo "Skipping stage 2 because all LoRA checkpoints already exist."
else
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
    --crct_epochs "${CRCT_EPOCHS}" \
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
    --trained_original_model "${TII_OUTPUT}" \
    --output_dir "${LORA_OUTPUT}"

  require_checkpoints "${LORA_OUTPUT}" "Stage 2 LoRA"
fi
