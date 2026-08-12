#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
RUN_DIR="${1:-}"
CANDIDATE_TASKS="${2:-3}"
MODE="${3:-fixed}"
TII_PRIOR_WEIGHT="${4:-0.3}"
LOGIT_TEMPERATURE="${5:-1.0}"
CONFIDENT_MARGIN="${6:-1.0}"
AMBIGUOUS_MARGIN="${7:-0.35}"
CANDIDATE_SOURCE="${SELECTIVE_SOURCE:-tii}"
CASCADE_WEIGHT="${CASCADE_WEIGHT:-0.5}"
EXCLUDED_LOGIT_MARGIN="${EXCLUDED_LOGIT_MARGIN:-20.0}"
TII_DIR="${TII_DIR:-${OUTPUT_ROOT}/imr_tii_original_10tasks_seed42}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: $0 RUN_DIR [candidate_tasks] [fixed|adaptive] [tii_prior] [temperature] [confident_margin] [ambiguous_margin]" >&2
  exit 64
fi
if [[ "${MODE}" != "fixed" && "${MODE}" != "adaptive" ]]; then
  echo "Mode must be fixed or adaptive, got: ${MODE}" >&2
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
PRIOR_TAG="$(tag_value "${TII_PRIOR_WEIGHT}")"
TEMP_TAG="$(tag_value "${LOGIT_TEMPERATURE}")"
MARGIN_TAG="$(tag_value "${CONFIDENT_MARGIN}")_$(tag_value "${AMBIGUOUS_MARGIN}")"
SOURCE_TAG="${CANDIDATE_SOURCE}_w$(tag_value "${CASCADE_WEIGHT}")"
LOG_PATH="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_selective_k${CANDIDATE_TASKS}_${MODE}_${SOURCE_TAG}_p${PRIOR_TAG}_t${TEMP_TAG}_m${MARGIN_TAG}.log"

if [[ -s "${LOG_PATH}" ]]; then
  echo "Refusing to overwrite existing evaluation log: ${LOG_PATH}" >&2
  exit 3
fi

ADAPTIVE_ARGS=()
if [[ "${MODE}" == "adaptive" ]]; then
  ADAPTIVE_ARGS+=(--selective_adaptive)
fi

cd "${REPO_ROOT}"
echo "Selective Rematching: mode=${MODE}, max LoRAs=${CANDIDATE_TASKS}, source=${CANDIDATE_SOURCE}"
echo "TII prior=${TII_PRIOR_WEIGHT}, cascade weight=${CASCADE_WEIGHT}, temperature=${LOGIT_TEMPERATURE}, margins=${CONFIDENT_MARGIN}/${AMBIGUOUS_MARGIN}"
echo "Run: ${RUN_DIR}"

START_TIME="$(date +%s)"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT:-29557}" \
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
  --selective_rematching \
  --selective_candidate_tasks "${CANDIDATE_TASKS}" \
  --selective_tii_prior_weight "${TII_PRIOR_WEIGHT}" \
  --selective_logit_temperature "${LOGIT_TEMPERATURE}" \
  --selective_confident_margin "${CONFIDENT_MARGIN}" \
  --selective_ambiguous_margin "${AMBIGUOUS_MARGIN}" \
  --selective_candidate_source "${CANDIDATE_SOURCE}" \
  --selective_cascade_weight "${CASCADE_WEIGHT}" \
  --selective_excluded_logit_margin "${EXCLUDED_LOGIT_MARGIN}" \
  "${ADAPTIVE_ARGS[@]}" \
  --eval \
  --output_dir "${RUN_DIR}" \
  2>&1 | tee "${LOG_PATH}"

END_TIME="$(date +%s)"
printf 'Selective evaluation wall time seconds: %s\n' "$((END_TIME - START_TIME))" | tee -a "${LOG_PATH}"
echo "Final selective-rematching metrics:"
grep "Average accuracy till task10" "${LOG_PATH}" | tail -n 1 || true
