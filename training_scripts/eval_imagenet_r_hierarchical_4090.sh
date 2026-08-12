#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
RUN_DIR="${1:-}"
TII_WEIGHT="${2:-0.3}"
MAX_WEIGHT="${3:-1.0}"
MARGIN_WEIGHT="${4:-0.5}"
ENTROPY_WEIGHT="${5:-0.5}"
CLASS_TEMPERATURE="${6:-1.0}"
TASK_TEMPERATURE="${7:-1.0}"
TII_DIR="${TII_DIR:-${OUTPUT_ROOT}/imr_tii_original_10tasks_seed42}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: $0 RUN_DIR [tii_weight] [max_weight] [margin_weight] [entropy_weight] [class_temperature] [task_temperature]" >&2
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
LOG_PATH="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_hierarchical_tii$(tag_value "${TII_WEIGHT}")_max$(tag_value "${MAX_WEIGHT}")_margin$(tag_value "${MARGIN_WEIGHT}")_entropy$(tag_value "${ENTROPY_WEIGHT}")_ct$(tag_value "${CLASS_TEMPERATURE}")_tt$(tag_value "${TASK_TEMPERATURE}").log"
if [[ -s "${LOG_PATH}" ]]; then
  if [[ "${SKIP_EXISTING:-0}" == "1" ]] && grep -q "Average accuracy till task10" "${LOG_PATH}"; then
    echo "Skipping completed hierarchical setting: ${LOG_PATH}"
    exit 0
  fi
  echo "Refusing to overwrite existing evaluation log: ${LOG_PATH}" >&2
  exit 3
fi

cd "${REPO_ROOT}"
echo "Hierarchical exhaustive rematching"
echo "P(class|x) = P(task|x) * P(class|task,x)"
echo "weights: TII=${TII_WEIGHT}, max=${MAX_WEIGHT}, margin=${MARGIN_WEIGHT}, entropy=${ENTROPY_WEIGHT}"
echo "temperatures: class=${CLASS_TEMPERATURE}, task=${TASK_TEMPERATURE}"

START_TIME="$(date +%s)"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT:-29563}" \
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
  --hierarchical_rematching \
  --hierarchical_tii_weight "${TII_WEIGHT}" \
  --hierarchical_max_weight "${MAX_WEIGHT}" \
  --hierarchical_margin_weight "${MARGIN_WEIGHT}" \
  --hierarchical_entropy_weight "${ENTROPY_WEIGHT}" \
  --hierarchical_class_temperature "${CLASS_TEMPERATURE}" \
  --hierarchical_task_temperature "${TASK_TEMPERATURE}" \
  --eval \
  --output_dir "${RUN_DIR}" \
  2>&1 | tee "${LOG_PATH}"

END_TIME="$(date +%s)"
printf 'Hierarchical evaluation wall time seconds: %s\n' "$((END_TIME - START_TIME))" | tee -a "${LOG_PATH}"
echo "Final hierarchical-rematching metrics:"
grep "Average accuracy till task10" "${LOG_PATH}" | tail -n 1 || true