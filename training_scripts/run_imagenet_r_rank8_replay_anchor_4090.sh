#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
if [[ "${MODE}" != "smoke" && "${MODE}" != "full" && "${MODE}" != "baseline" ]]; then
  echo "Usage: $0 [smoke|full|baseline]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
SEED="${SEED:-42}"
GPUS="${GPUS:-1}"

if [[ -n "${DATA_PATH:-}" ]]; then
  IMR_DATA_PATH="${DATA_PATH}"
elif [[ -d "${DATASETS_ROOT}/imagenet-r/imagenet-r" || -f "${DATASETS_ROOT}/imagenet-r/imagenet-r.tar" ]]; then
  IMR_DATA_PATH="${DATASETS_ROOT}/imagenet-r"
else
  IMR_DATA_PATH="${DATASETS_ROOT}"
fi

TII_DIR="${TII_DIR:-${OUTPUT_ROOT}/imr_tii_original_10tasks_seed${SEED}}"
LORA_EPOCHS="${LORA_EPOCHS:-50}"
CRCT_EPOCHS="${CRCT_EPOCHS:-30}"
MAX_TRAIN_TASKS="${MAX_TRAIN_TASKS:-0}"
RUN_NAME="${RUN_NAME_OVERRIDE:-imr_lora_rank8_cfs_pmi_anchor_w005_seed${SEED}}"
ANCHOR_ENABLED=1

if [[ "${MODE}" == "smoke" ]]; then
  LORA_EPOCHS="${SMOKE_LORA_EPOCHS:-1}"
  CRCT_EPOCHS="${SMOKE_CRCT_EPOCHS:-1}"
  MAX_TRAIN_TASKS="${SMOKE_TASKS:-2}"
  RUN_NAME="${RUN_NAME_OVERRIDE:-imr_rank8_replay_anchor_smoke_seed${SEED}}"
elif [[ "${MODE}" == "baseline" ]]; then
  ANCHOR_ENABLED=0
  RUN_NAME="${RUN_NAME_OVERRIDE:-imr_lora_rank8_baseline_10tasks_seed${SEED}}"
fi

OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_PATH="${OUTPUT_ROOT}/${RUN_NAME}.log"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -d "${IMR_DATA_PATH}/imagenet-r" && ! -f "${IMR_DATA_PATH}/imagenet-r.tar" ]]; then
  echo "ImageNet-R was not found under: ${IMR_DATA_PATH}" >&2
  echo "Set DATA_PATH to the directory containing imagenet-r/ or imagenet-r.tar." >&2
  exit 2
fi

for task_id in $(seq 1 10); do
  checkpoint="${TII_DIR}/checkpoint/task${task_id}_checkpoint.pth"
  if [[ ! -s "${checkpoint}" ]]; then
    echo "Missing TII checkpoint: ${checkpoint}" >&2
    exit 3
  fi
done

if [[ -d "${OUTPUT_DIR}/checkpoint" ]] && compgen -G "${OUTPUT_DIR}/checkpoint/task*_checkpoint.pth" > /dev/null; then
  echo "Refusing to overwrite checkpoints in: ${OUTPUT_DIR}" >&2
  echo "Set RUN_NAME_OVERRIDE to a new name before rerunning." >&2
  exit 4
fi
if [[ -s "${LOG_PATH}" ]]; then
  echo "Refusing to overwrite existing log: ${LOG_PATH}" >&2
  echo "Set RUN_NAME_OVERRIDE to a new name before rerunning." >&2
  exit 4
fi

METHOD_ARGS=()
if [[ "${ANCHOR_ENABLED}" == "1" ]]; then
  LAYER_STEPS="${INVERSION_LAYER_STEPS:-200}"
  FULL_STEPS="${INVERSION_FULL_STEPS:-600}"
  IMAGES_PER_CLASS="${REPLAY_IMAGES_PER_CLASS:-5}"
  CFS_EPOCHS="${CFS_EPOCHS:-200}"
  if [[ "${MODE}" == "smoke" ]]; then
    LAYER_STEPS="${SMOKE_INVERSION_LAYER_STEPS:-2}"
    FULL_STEPS="${SMOKE_INVERSION_FULL_STEPS:-3}"
    IMAGES_PER_CLASS="${SMOKE_REPLAY_IMAGES_PER_CLASS:-2}"
    CFS_EPOCHS="${SMOKE_CFS_EPOCHS:-1}"
  fi
  METHOD_ARGS=(
    --ca_storage_efficient_method variance
    --cfs_sampling
    --cfs_epochs "${CFS_EPOCHS}"
    --cfs_lr "${CFS_LR:-0.01}"
    --cfs_candidate_multiplier "${CFS_CANDIDATE_MULTIPLIER:-3}"
    --cfs_train_max_samples "${CFS_TRAIN_MAX_SAMPLES:-1024}"
    --cfs_paper_style
    --cfs_selection_ratio "${CFS_SELECTION_RATIO:-0.5}"
    --cfs_selection_steps "${CFS_SELECTION_STEPS:-5}"
    --replay_anchor_ctird
    --replay_anchor_cache_dir "${REPLAY_CACHE_DIR:-${OUTPUT_DIR}/replay_anchor_cache}"
    --replay_anchor_weight "${REPLAY_ANCHOR_WEIGHT:-0.05}"
    --replay_anchor_batch_size "${REPLAY_ANCHOR_BATCH_SIZE:-20}"
    --replay_anchor_temperature "${REPLAY_ANCHOR_TEMPERATURE:-1.0}"
    --replay_anchor_teacher_confidence "${REPLAY_TEACHER_CONFIDENCE:-0.2}"
    --replay_anchor_images_per_class "${IMAGES_PER_CLASS}"
    --replay_inversion_candidate_multiplier "${INVERSION_CANDIDATE_MULTIPLIER:-2}"
    --replay_inversion_split_block "${INVERSION_SPLIT_BLOCK:-5}"
    --replay_inversion_layer_steps "${LAYER_STEPS}"
    --replay_inversion_full_steps "${FULL_STEPS}"
    --replay_inversion_layer_lr "${INVERSION_LAYER_LR:-0.1}"
    --replay_inversion_full_lr "${INVERSION_FULL_LR:-0.01}"
    --replay_inversion_class_weight "${INVERSION_CLASS_WEIGHT:-0.1}"
    --replay_inversion_tv_weight "${INVERSION_TV_WEIGHT:-0.0005}"
  )
fi

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

echo "Mode: ${MODE}"
echo "Dataset: ${IMR_DATA_PATH}"
echo "TII checkpoints: ${TII_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "LoRA rank: 8"
if [[ "${ANCHOR_ENABLED}" == "1" ]]; then
  echo "Replay-Anchored CTIRD: weight=${REPLAY_ANCHOR_WEIGHT:-0.05}, images/class=${IMAGES_PER_CLASS}, inversion=${LAYER_STEPS}+${FULL_STEPS} steps"
  echo "Replay cache: ${REPLAY_CACHE_DIR:-${OUTPUT_DIR}/replay_anchor_cache}"
else
  echo "Rank-8 baseline: original HRM-PET training without CFS-PMI replay"
fi

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT:-29563}" \
  main.py \
  imr_lora \
  --model vit_base_patch16_224 \
  --original_model vit_base_patch16_224 \
  --batch-size "${LORA_BATCH_SIZE:-24}" \
  --epochs "${LORA_EPOCHS}" \
  --data-path "${IMR_DATA_PATH}" \
  --ca_lr "${CA_LR:-0.005}" \
  --crct_epochs "${CRCT_EPOCHS}" \
  --seed "${SEED}" \
  --lr 0.03 \
  --con 0.2 \
  --lora_rank 8 \
  --En gen \
  --tau -10 \
  --K 5 \
  --sched cosine \
  --dataset Split-Imagenet-R \
  --lora_momentum 0.4 \
  --lora_type hide \
  --trained_original_model "${TII_DIR}" \
  --num_tasks 10 \
  --max_train_tasks "${MAX_TRAIN_TASKS}" \
  "${METHOD_ARGS[@]}" \
  --output_dir "${OUTPUT_DIR}" \
  2>&1 | tee "${LOG_PATH}"

EXPECTED_TASKS=10
if [[ "${MAX_TRAIN_TASKS}" -gt 0 ]]; then
  EXPECTED_TASKS="${MAX_TRAIN_TASKS}"
fi
for task_id in $(seq 1 "${EXPECTED_TASKS}"); do
  checkpoint="${OUTPUT_DIR}/checkpoint/task${task_id}_checkpoint.pth"
  if [[ ! -s "${checkpoint}" ]]; then
    echo "Missing checkpoint after training: ${checkpoint}" >&2
    exit 5
  fi
done

if [[ "${ANCHOR_ENABLED}" == "1" && "${EXPECTED_TASKS}" -gt 1 ]]; then
  EXPECTED_CACHED_CLASSES=$(( (EXPECTED_TASKS - 1) * 20 ))
  ACTIVE_CACHE_DIR="${REPLAY_CACHE_DIR:-${OUTPUT_DIR}/replay_anchor_cache}"
  CACHED_CLASSES=$(find "${ACTIVE_CACHE_DIR}" -name 'class_*.pth' -type f | wc -l)
  if [[ "${CACHED_CLASSES}" -lt "${EXPECTED_CACHED_CLASSES}" ]]; then
    echo "Incomplete replay cache: expected ${EXPECTED_CACHED_CLASSES} classes, found ${CACHED_CLASSES}" >&2
    exit 6
  fi
fi

echo "Final metrics:"
grep "Average accuracy till task${EXPECTED_TASKS}" "${LOG_PATH}" | tail -n 1 || true
