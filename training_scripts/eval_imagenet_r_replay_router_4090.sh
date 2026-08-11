#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
RUN_DIR="${1:-}"
ROUTER_EPOCHS="${2:-50}"
HIDDEN_DIM="${3:-256}"
MIN_VALIDATION_GAIN="${4:-0.25}"
TII_DIR="${TII_DIR:-${OUTPUT_ROOT}/imr_tii_original_10tasks_seed42}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: $0 RUN_DIR [router_epochs] [hidden_dim] [min_validation_gain]" >&2
  exit 64
fi

if [[ -n "${DATA_PATH:-}" ]]; then
  IMR_DATA_PATH="${DATA_PATH}"
elif [[ -d "${DATASETS_ROOT}/imagenet-r/imagenet-r" || -f "${DATASETS_ROOT}/imagenet-r/imagenet-r.tar" ]]; then
  IMR_DATA_PATH="${DATASETS_ROOT}/imagenet-r"
else
  IMR_DATA_PATH="${DATASETS_ROOT}"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 1
fi

for task_id in $(seq 1 10); do
  run_checkpoint="${RUN_DIR}/checkpoint/task${task_id}_checkpoint.pth"
  tii_checkpoint="${TII_DIR}/checkpoint/task${task_id}_checkpoint.pth"
  if [[ ! -s "${run_checkpoint}" ]]; then
    echo "Missing LoRA checkpoint: ${run_checkpoint}" >&2
    exit 2
  fi
  if [[ ! -s "${tii_checkpoint}" ]]; then
    echo "Missing TII checkpoint: ${tii_checkpoint}" >&2
    exit 2
  fi
done

tag_value() {
  printf '%s' "$1" | tr '.' 'p'
}

RUN_BASENAME="$(basename "${RUN_DIR}")"
GAIN_TAG="$(tag_value "${MIN_VALIDATION_GAIN}")"
LOG_PATH="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_replayrouter_e${ROUTER_EPOCHS}_h${HIDDEN_DIM}_g${GAIN_TAG}.log"

if [[ -s "${LOG_PATH}" ]]; then
  echo "Refusing to overwrite existing evaluation log: ${LOG_PATH}" >&2
  echo "Change one router parameter or move the existing log first." >&2
  exit 3
fi

cd "${REPO_ROOT}"

echo "Validated replay task router: nonlinear 10-way routing with automatic TII fallback."
echo "Run: ${RUN_DIR}"
echo "epochs=${ROUTER_EPOCHS}, hidden_dim=${HIDDEN_DIM}, min_validation_gain=${MIN_VALIDATION_GAIN}"

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT:-29551}" \
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
  --replay_task_router \
  --replay_router_epochs "${ROUTER_EPOCHS}" \
  --replay_router_lr "${ROUTER_LR:-0.001}" \
  --replay_router_weight_decay "${ROUTER_WEIGHT_DECAY:-0.01}" \
  --replay_router_hidden_dim "${HIDDEN_DIM}" \
  --replay_router_dropout "${ROUTER_DROPOUT:-0.1}" \
  --replay_router_patience "${ROUTER_PATIENCE:-8}" \
  --replay_router_validation_ratio "${VALIDATION_RATIO:-0.25}" \
  --replay_router_min_validation_gain "${MIN_VALIDATION_GAIN}" \
  --replay_router_samples_per_class "${MEMORY_SAMPLES_PER_CLASS:-48}" \
  --eval \
  --output_dir "${RUN_DIR}" \
  2>&1 | tee "${LOG_PATH}"

echo "Final validated-router evaluation metrics:"
grep "Average accuracy till task10" "${LOG_PATH}" | tail -n 1 || true
