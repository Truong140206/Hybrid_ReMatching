#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
RUN_DIR="${1:-}"
TOP_K="${2:-3}"
SAMPLES_PER_CLASS="${3:-8}"
TII_DIR="${TII_DIR:-${OUTPUT_ROOT}/imr_tii_original_10tasks_seed42}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: $0 RUN_DIR [top_k] [teacher_samples_per_class]" >&2
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

RUN_BASENAME="$(basename "${RUN_DIR}")"
LOG_PATH="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_distilled_topk${TOP_K}_s${SAMPLES_PER_CLASS}.log"
if [[ -s "${LOG_PATH}" ]]; then
  echo "Refusing to overwrite existing evaluation log: ${LOG_PATH}" >&2
  exit 3
fi

cd "${REPO_ROOT}"
echo "Exhaustive-teacher distilled top-${TOP_K} router"
echo "Teacher samples per class: ${SAMPLES_PER_CLASS}"
echo "Exhaustive labels training images only; test inference runs at most ${TOP_K} LoRAs."

START_TIME="$(date +%s)"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT:-29559}" \
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
  --selective_candidate_source router \
  --selective_candidate_tasks "${TOP_K}" \
  --selective_tii_prior_weight 0.3 \
  --selective_logit_temperature 1.0 \
  --selective_excluded_logit_margin 20.0 \
  --distilled_router_rematching \
  --distilled_router_samples_per_class "${SAMPLES_PER_CLASS}" \
  --distilled_router_teacher_weight "${TEACHER_WEIGHT:-0.3}" \
  --distilled_router_teacher_prior 0.3 \
  --distilled_router_epochs "${ROUTER_EPOCHS:-50}" \
  --distilled_router_patience "${ROUTER_PATIENCE:-8}" \
  --eval \
  --output_dir "${RUN_DIR}" \
  2>&1 | tee "${LOG_PATH}"

END_TIME="$(date +%s)"
printf 'Distilled top-k evaluation wall time seconds: %s\n' "$((END_TIME - START_TIME))" | tee -a "${LOG_PATH}"
echo "Final distilled top-k metrics:"
grep "Average accuracy till task10" "${LOG_PATH}" | tail -n 1 || true
