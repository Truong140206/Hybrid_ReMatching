#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
RUN_DIR="${1:-}"
STEPS="${2:-200}"
OLD_MARGIN_WEIGHT="${3:-0.25}"
REGULARIZATION="${4:-0.01}"
TII_DIR="${TII_DIR:-${OUTPUT_ROOT}/imr_tii_original_10tasks_seed42}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: $0 RUN_DIR [steps] [old_margin_weight] [regularization]" >&2
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
MARGIN_TAG="$(tag_value "${OLD_MARGIN_WEIGHT}")"
REG_TAG="$(tag_value "${REGULARIZATION}")"
LOG_PATH="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_replaycal_s${STEPS}_m${MARGIN_TAG}_r${REG_TAG}.log"

if [[ -s "${LOG_PATH}" ]]; then
  echo "Refusing to overwrite existing evaluation log: ${LOG_PATH}" >&2
  echo "Change one calibration parameter or move the existing log first." >&2
  exit 3
fi

cd "${REPO_ROOT}"

echo "Replay-calibrated task logit alignment uses only feature memory stored in each checkpoint."
echo "Run: ${RUN_DIR}"
echo "steps=${STEPS}, old_margin_weight=${OLD_MARGIN_WEIGHT}, regularization=${REGULARIZATION}"

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT:-29549}" \
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
  --replay_logit_calibration \
  --replay_calibration_steps "${STEPS}" \
  --replay_calibration_lr "${CALIBRATION_LR:-0.05}" \
  --replay_calibration_max_scale "${MAX_SCALE:-1.5}" \
  --replay_calibration_max_bias "${MAX_BIAS:-1.0}" \
  --replay_calibration_regularization "${REGULARIZATION}" \
  --replay_calibration_old_margin_weight "${OLD_MARGIN_WEIGHT}" \
  --replay_calibration_old_tolerance "${OLD_TOLERANCE:-0.0}" \
  --replay_calibration_min_gain "${MIN_MEMORY_GAIN:-0.0}" \
  --replay_calibration_samples_per_class "${MEMORY_SAMPLES_PER_CLASS:-48}" \
  --eval \
  --output_dir "${RUN_DIR}" \
  2>&1 | tee "${LOG_PATH}"

echo "Final replay-calibrated evaluation metrics:"
grep "Average accuracy till task10" "${LOG_PATH}" | tail -n 1 || true
