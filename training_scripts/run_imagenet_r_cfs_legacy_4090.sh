#!/usr/bin/env bash
set -euo pipefail

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
RUN_NAME="${RUN_NAME_OVERRIDE:-imr_lora_cfs_legacy_ctird_10tasks_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_PATH="${OUTPUT_ROOT}/${RUN_NAME}.log"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 1
fi

for task_id in $(seq 1 10); do
  checkpoint="${TII_DIR}/checkpoint/task${task_id}_checkpoint.pth"
  if [[ ! -s "${checkpoint}" ]]; then
    echo "Missing TII checkpoint: ${checkpoint}" >&2
    exit 2
  fi
done

if [[ -d "${OUTPUT_DIR}/checkpoint" ]] && compgen -G "${OUTPUT_DIR}/checkpoint/task*_checkpoint.pth" > /dev/null; then
  echo "Refusing to overwrite checkpoints in: ${OUTPUT_DIR}" >&2
  echo "Set RUN_NAME_OVERRIDE to a new name before rerunning." >&2
  exit 3
fi

if [[ -s "${LOG_PATH}" ]]; then
  echo "Refusing to overwrite existing log: ${LOG_PATH}" >&2
  echo "Set RUN_NAME_OVERRIDE to a new name before rerunning." >&2
  exit 3
fi

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

# Keep the original legacy CTIRD selection. Only the replay/correction branch
# changes: CFS diversity, boundary samples, and full replay.
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT:-29534}" \
  main.py \
  imr_lora \
  --model vit_base_patch16_224 \
  --original_model vit_base_patch16_224 \
  --batch-size "${LORA_BATCH_SIZE:-24}" \
  --epochs "${LORA_EPOCHS:-50}" \
  --data-path "${IMR_DATA_PATH}" \
  --ca_lr "${CA_LR:-0.005}" \
  --crct_epochs "${CRCT_EPOCHS:-30}" \
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
  --trained_original_model "${TII_DIR}" \
  --num_tasks 10 \
  --cfs_sampling \
  --cfs_epochs "${CFS_EPOCHS:-20}" \
  --cfs_train_max_samples "${CFS_TRAIN_MAX_SAMPLES:-1024}" \
  --cfs_candidate_multiplier "${CFS_CANDIDATE_MULTIPLIER:-2}" \
  --cfs_boundary_replay \
  --cfs_boundary_ratio "${CFS_BOUNDARY_RATIO:-0.25}" \
  --cfs_boundary_multiplier "${CFS_BOUNDARY_MULTIPLIER:-3}" \
  --cfs_boundary_density_quantile "${CFS_BOUNDARY_DENSITY_QUANTILE:-0.9}" \
  --crct_use_all_samples \
  --output_dir "${OUTPUT_DIR}" \
  2>&1 | tee "${LOG_PATH}"

for task_id in $(seq 1 10); do
  checkpoint="${OUTPUT_DIR}/checkpoint/task${task_id}_checkpoint.pth"
  if [[ ! -s "${checkpoint}" ]]; then
    echo "Missing CFS checkpoint after training: ${checkpoint}" >&2
    exit 4
  fi
done

echo "Final metrics:"
grep "Average accuracy till task10" "${LOG_PATH}" | tail -n 1 || true
