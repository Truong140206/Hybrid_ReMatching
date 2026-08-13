#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
RUN_DIR="${1:-}"
SAMPLES_PER_CLASS="${2:-12}"
TARGET_PRECISION="${3:-0.995}"
EXCLUDED_LOGIT_MARGIN="${EXCLUDED_LOGIT_MARGIN:-8.0}"
STAGE2_MIN_SAMPLES_PER_CLASS="${STAGE2_MIN_SAMPLES_PER_CLASS:-4}"
STAGE2_CONTEXT_RATIO="${STAGE2_CONTEXT_RATIO:-0.25}"
STAGE2_TARGET_PRECISION="${STAGE2_TARGET_PRECISION:-1.0}"
STAGE2_LOSS_TOLERANCE="${STAGE2_LOSS_TOLERANCE:-0.0}"
LORA_BATCH_RANKS="${LORA_BATCH_RANKS:-2}"
TII_DIR="${TII_DIR:-${OUTPUT_ROOT}/imr_tii_original_10tasks_seed42}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: $0 RUN_DIR [samples_per_class] [target_precision]" >&2
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
  [[ -s "${RUN_DIR}/checkpoint/task${task_id}_checkpoint.pth" ]] || {
    echo "Missing LoRA checkpoint: ${RUN_DIR}/checkpoint/task${task_id}_checkpoint.pth" >&2
    exit 2
  }
  [[ -s "${TII_DIR}/checkpoint/task${task_id}_checkpoint.pth" ]] || {
    echo "Missing TII checkpoint: ${TII_DIR}/checkpoint/task${task_id}_checkpoint.pth" >&2
    exit 2
  }
done

tag_value() {
  printf '%s' "$1" | tr '.' 'p'
}

RUN_BASENAME="$(basename "${RUN_DIR}")"
PRECISION_TAG="$(tag_value "${TARGET_PRECISION}")"
MARGIN_TAG="$(tag_value "${EXCLUDED_LOGIT_MARGIN}")"
CONTEXT_TAG="$(tag_value "${STAGE2_CONTEXT_RATIO}")"
STAGE2_PRECISION_TAG="$(tag_value "${STAGE2_TARGET_PRECISION}")"
LOG_PATH="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_calibrated_progressive_batch${LORA_BATCH_RANKS}_smooth05_s${SAMPLES_PER_CLASS}_q${PRECISION_TAG}_q2${STAGE2_PRECISION_TAG}_c${CONTEXT_TAG}_m${MARGIN_TAG}.log"
if [[ -s "${LOG_PATH}" ]]; then
  echo "Refusing to overwrite existing evaluation log: ${LOG_PATH}" >&2
  exit 3
fi

cd "${REPO_ROOT}"
echo "Train-calibrated progressive rematching: 2 -> 4 -> all seen LoRAs"
echo "Calibration uses only train images; test images never tune the gates."
echo "Run=${RUN_DIR}; samples/class=${SAMPLES_PER_CLASS}; target precision=${TARGET_PRECISION}"
echo "Stage-2 calibration is conditioned on Stage-1 rejects; excluded-logit margin=${EXCLUDED_LOGIT_MARGIN}"
echo "Stage-2 boundary context=${STAGE2_CONTEXT_RATIO}; target precision=${STAGE2_TARGET_PRECISION}; loss tolerance=${STAGE2_LOSS_TOLERANCE}"
echo "Rank-preserving uncertainty smoothing is applied only to early exits; exhaustive fallbacks are unchanged."
echo "Adjacent LoRA ranks per GPU forward=${LORA_BATCH_RANKS}; gate boundaries remain 2 -> 4 -> all."

START_TIME="$(date +%s)"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT:-29571}" \
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
  --calibrated_progressive_rematching \
  --progressive_gate_samples_per_class "${SAMPLES_PER_CLASS}" \
  --progressive_gate_target_precision "${TARGET_PRECISION}" \
  --progressive_gate_stage2_min_samples_per_class "${STAGE2_MIN_SAMPLES_PER_CLASS}" \
  --progressive_gate_stage2_context_ratio "${STAGE2_CONTEXT_RATIO}" \
  --progressive_gate_stage2_target_precision "${STAGE2_TARGET_PRECISION}" \
  --progressive_gate_stage2_loss_tolerance "${STAGE2_LOSS_TOLERANCE}" \
  --progressive_gate_min_coverage 0.02 \
  --progressive_gate_validation_ratio 0.25 \
  --progressive_gate_epochs 60 \
  --progressive_gate_patience 10 \
  --progressive_tii_prior_weight 0.3 \
  --progressive_logit_temperature 1.0 \
  --progressive_excluded_logit_margin "${EXCLUDED_LOGIT_MARGIN}" \
  --progressive_uncertainty_smoothing \
  --progressive_smoothing_strength 0.5 \
  --progressive_smoothing_max 0.05 \
  --progressive_lora_batch_ranks "${LORA_BATCH_RANKS}" \
  --eval \
  --output_dir "${RUN_DIR}" \
  2>&1 | tee "${LOG_PATH}"

END_TIME="$(date +%s)"
printf 'Calibrated progressive wall time seconds: %s\n' "$((END_TIME - START_TIME))" | tee -a "${LOG_PATH}"
echo "Final calibrated-progressive metrics:"
grep "Average accuracy till task10" "${LOG_PATH}" | tail -n 1 || true
