#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
RUN_DIR="${1:-}"
FALLBACK_FRACTION="${2:-0.2}"
TII_PRIOR_WEIGHT="${3:-0.3}"
LOGIT_TEMPERATURE="${4:-1.0}"
TII_DIR="${TII_DIR:-${OUTPUT_ROOT}/imr_tii_original_10tasks_seed42}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: $0 RUN_DIR [fallback_fraction] [tii_prior] [temperature]" >&2
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
FRACTION_TAG="$(tag_value "${FALLBACK_FRACTION}")"
PRIOR_TAG="$(tag_value "${TII_PRIOR_WEIGHT}")"
TEMP_TAG="$(tag_value "${LOGIT_TEMPERATURE}")"
LOG_PATH="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_budgeted_v2_f${FRACTION_TAG}_p${PRIOR_TAG}_t${TEMP_TAG}.log"
if [[ -s "${LOG_PATH}" ]]; then
  echo "Refusing to overwrite existing evaluation log: ${LOG_PATH}" >&2
  exit 3
fi

cd "${REPO_ROOT}"
echo "Confidence-gated exhaustive fallback"
echo "Fallback fraction=${FALLBACK_FRACTION}, TII prior=${TII_PRIOR_WEIGHT}, temperature=${LOGIT_TEMPERATURE}"
echo "Original routing is preserved; disagreement and low-margin samples receive exhaustive rematching."

START_TIME="$(date +%s)"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT:-29561}" \
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
  --budgeted_rematching \
  --budgeted_fallback_fraction "${FALLBACK_FRACTION}" \
  --budgeted_tii_prior_weight "${TII_PRIOR_WEIGHT}" \
  --budgeted_logit_temperature "${LOGIT_TEMPERATURE}" \
  --budgeted_disagreement_weight "${DISAGREEMENT_WEIGHT:-2.0}" \
  --budgeted_classifier_weight "${CLASSIFIER_WEIGHT:-0.5}" \
  --eval \
  --output_dir "${RUN_DIR}" \
  2>&1 | tee "${LOG_PATH}"

END_TIME="$(date +%s)"
printf 'Budgeted evaluation wall time seconds: %s\n' "$((END_TIME - START_TIME))" | tee -a "${LOG_PATH}"
echo "Final budgeted-rematching metrics:"
grep "Average accuracy till task10" "${LOG_PATH}" | tail -n 1 || true
