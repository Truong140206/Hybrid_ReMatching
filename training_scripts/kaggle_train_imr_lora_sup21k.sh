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
CFS_ARGS=()
if [ "${CFS_SAMPLING:-0}" = "1" ]; then
  CFS_ARGS+=(--cfs_sampling)
  CFS_ARGS+=(--cfs_epochs "${CFS_EPOCHS:-50}")
  CFS_ARGS+=(--cfs_lr "${CFS_LR:-0.01}")
  CFS_ARGS+=(--cfs_hidden_dim "${CFS_HIDDEN_DIM:-512}")
  CFS_ARGS+=(--cfs_batch_size "${CFS_BATCH_SIZE:-256}")
  CFS_ARGS+=(--cfs_train_max_samples "${CFS_TRAIN_MAX_SAMPLES:-1024}")
  CFS_ARGS+=(--cfs_candidate_multiplier "${CFS_CANDIDATE_MULTIPLIER:-3}")
  CFS_ARGS+=(--cfs_tau "${CFS_TAU:-1.0}")
fi
SEMANTIC_ARGS=()
if [ "${SEMANTIC_DISTILL:-0}" = "1" ]; then
  SEMANTIC_ARGS+=(--semantic_distill)
  SEMANTIC_ARGS+=(--semantic_dim "${SEMANTIC_DIM:-512}")
  if [ -n "${SEMANTIC_CLASS_NAME_FILE:-}" ]; then
    SEMANTIC_ARGS+=(--semantic_class_name_file "${SEMANTIC_CLASS_NAME_FILE}")
  fi
  SEMANTIC_ARGS+=(--semantic_alpha "${SEMANTIC_ALPHA:-0.05}")
  SEMANTIC_ARGS+=(--semantic_floor "${SEMANTIC_FLOOR:-0.2}")
  SEMANTIC_ARGS+=(--semantic_sharpness "${SEMANTIC_SHARPNESS:-1.0}")
  SEMANTIC_ARGS+=(--semantic_top_k "${SEMANTIC_TOP_K:-5}")
  SEMANTIC_ARGS+=(--semantic_mode "${SEMANTIC_MODE:-adaptive_gate}")
  if [ "${SEMANTIC_PROJECTION:-0}" = "1" ]; then
    SEMANTIC_ARGS+=(--semantic_projection)
    SEMANTIC_ARGS+=(--semantic_projection_ratio "${SEMANTIC_PROJECTION_RATIO:-0.25}")
    SEMANTIC_ARGS+=(--semantic_projection_top_k "${SEMANTIC_PROJECTION_TOP_K:-5}")
    SEMANTIC_ARGS+=(--semantic_projection_strength "${SEMANTIC_PROJECTION_STRENGTH:-1.0}")
  fi
fi
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
    --output_dir "${TII_OUTPUT}" \
    "${CFS_ARGS[@]}" \
    "${SEMANTIC_ARGS[@]}"

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
    --output_dir "${LORA_OUTPUT}" \
    "${CFS_ARGS[@]}" \
    "${SEMANTIC_ARGS[@]}"

  require_checkpoints "${LORA_OUTPUT}" "Stage 2 LoRA"
fi
