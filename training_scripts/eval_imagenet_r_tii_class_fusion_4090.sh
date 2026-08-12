#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
RUN_DIR="${1:-}"
TII_CLASS_WEIGHT="${2:-0.1}"
TII_CLASS_TEMPERATURE="${3:-1.0}"
MAX_CALIBRATION_WEIGHT="${4:-0.5}"
TII_PRIOR_WEIGHT="${5:-0.3}"
LOGIT_TEMPERATURE="${6:-1.0}"
TII_DIR="${TII_DIR:-${OUTPUT_ROOT}/imr_tii_original_10tasks_seed42}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: $0 RUN_DIR [tii_class_weight] [tii_class_temperature] [max_calibration] [tii_prior] [logit_temperature]" >&2
  exit 64
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ -n "${DATA_PATH:-}" ]]; then
  IMR_DATA_PATH="${DATA_PATH}"
elif [[ -d "${DATASETS_ROOT}/imagenet-r/imagenet-r" || -f "${DATASETS_ROOT}/imagenet-r/imagenet-r.tar" ]]; then
  IMR_DATA_PATH="${DATASETS_ROOT}/imagenet-r"
else
  IMR_DATA_PATH="${DATASETS_ROOT}"
fi

for task_id in $(seq 1 10); do
  if [[ ! -s "${RUN_DIR}/checkpoint/task${task_id}_checkpoint.pth" ]]; then
    echo "Missing LoRA checkpoint: ${RUN_DIR}/checkpoint/task${task_id}_checkpoint.pth" >&2
    exit 2
  fi
  if [[ ! -s "${TII_DIR}/checkpoint/task${task_id}_checkpoint.pth" ]]; then
    echo "Missing TII checkpoint: ${TII_DIR}/checkpoint/task${task_id}_checkpoint.pth" >&2
    exit 2
  fi
done

tag_value() {
  printf '%s' "$1" | tr '.' 'p'
}

RUN_BASENAME="$(basename "${RUN_DIR}")"
CLASS_TAG="$(tag_value "${TII_CLASS_WEIGHT}")"
CLASS_TEMP_TAG="$(tag_value "${TII_CLASS_TEMPERATURE}")"
MAX_TAG="$(tag_value "${MAX_CALIBRATION_WEIGHT}")"
PRIOR_TAG="$(tag_value "${TII_PRIOR_WEIGHT}")"
TEMP_TAG="$(tag_value "${LOGIT_TEMPERATURE}")"
LOG_PATH="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_exhaustive_tiiclass_w${CLASS_TAG}_ct${CLASS_TEMP_TAG}_m${MAX_TAG}_p${PRIOR_TAG}_t${TEMP_TAG}.log"
if [[ -s "${LOG_PATH}" ]]; then
  if [[ "${SKIP_EXISTING:-0}" == "1" ]] && grep -q "Average accuracy till task10" "${LOG_PATH}"; then
    echo "Skipping completed TII-class fusion setting: ${LOG_PATH}"
    exit 0
  fi
  echo "Refusing to overwrite existing evaluation log: ${LOG_PATH}" >&2
  exit 3
fi

cd "${REPO_ROOT}"
echo "Task-max-preserving TII class fusion"
echo "class_weight=${TII_CLASS_WEIGHT}, class_temperature=${TII_CLASS_TEMPERATURE}"
echo "max_calibration=${MAX_CALIBRATION_WEIGHT}, TII prior=${TII_PRIOR_WEIGHT}, LoRA temperature=${LOGIT_TEMPERATURE}"

START_TIME="$(date +%s)"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT:-29567}" \
  main.py \
  imr_lora \
  --model vit_base_patch16_224 \
  --original_model vit_base_patch16_224 \
  --batch-size "${EVAL_BATCH_SIZE:-24}" \
  --epochs 1 \
  --data-path "${IMR_DATA_PATH}" \
  --seed 42 \
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
  --exhaustive_rematching \
  --exhaustive_tii_prior_weight "${TII_PRIOR_WEIGHT}" \
  --exhaustive_logit_temperature "${LOGIT_TEMPERATURE}" \
  --exhaustive_max_calibration_weight "${MAX_CALIBRATION_WEIGHT}" \
  --exhaustive_tii_class_weight "${TII_CLASS_WEIGHT}" \
  --exhaustive_tii_class_temperature "${TII_CLASS_TEMPERATURE}" \
  --eval \
  --output_dir "${RUN_DIR}" \
  2>&1 | tee "${LOG_PATH}"

END_TIME="$(date +%s)"
printf 'TII-class fusion evaluation wall time seconds: %s\n' "$((END_TIME - START_TIME))" | tee -a "${LOG_PATH}"
echo "Final task-max-preserving TII class-fusion metrics:"
grep "Average accuracy till task10" "${LOG_PATH}" | tail -n 1 || true