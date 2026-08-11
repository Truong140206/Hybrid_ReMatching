#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
RUN_DIR="${1:-}"
CANDIDATE_TASKS="${2:-2}"
PROTOTYPE_TEMPERATURE="${3:-0.07}"
CLASSIFIER_WEIGHT="${4:-0.5}"
TASK_PRIOR_WEIGHT="${5:-0.25}"
TII_DIR="${TII_DIR:-${OUTPUT_ROOT}/imr_tii_original_10tasks_seed42}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: $0 RUN_DIR [candidate_tasks] [prototype_temperature] [classifier_weight] [task_prior_weight]" >&2
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
TEMP_TAG="$(tag_value "${PROTOTYPE_TEMPERATURE}")"
CLASSIFIER_TAG="$(tag_value "${CLASSIFIER_WEIGHT}")"
PRIOR_TAG="$(tag_value "${TASK_PRIOR_WEIGHT}")"
LOG_PATH="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_proto_k${CANDIDATE_TASKS}_t${TEMP_TAG}_c${CLASSIFIER_TAG}_p${PRIOR_TAG}.log"

if [[ -s "${LOG_PATH}" ]]; then
  echo "Refusing to overwrite existing evaluation log: ${LOG_PATH}" >&2
  echo "Change one evaluation parameter or move the existing log first." >&2
  exit 3
fi

cd "${REPO_ROOT}"

echo "Prototype evaluation reconstructs feature memory from training images when the checkpoint does not contain it."
echo "Run: ${RUN_DIR}"
echo "Candidates=${CANDIDATE_TASKS}, temperature=${PROTOTYPE_TEMPERATURE}, classifier_weight=${CLASSIFIER_WEIGHT}, prior_weight=${TASK_PRIOR_WEIGHT}"

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT:-29543}" \
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
  --crct_real_feature_replay \
  --crct_real_memory_per_class "${REAL_MEMORY_PER_CLASS:-48}" \
  --crct_real_outlier_quantile "${REAL_OUTLIER_QUANTILE:-0.9}" \
  --crct_real_diversity_weight "${REAL_DIVERSITY_WEIGHT:-0.7}" \
  --prototype_rematching \
  --prototype_candidate_tasks "${CANDIDATE_TASKS}" \
  --prototype_temperature "${PROTOTYPE_TEMPERATURE}" \
  --prototype_classifier_weight "${CLASSIFIER_WEIGHT}" \
  --prototype_task_prior_weight "${TASK_PRIOR_WEIGHT}" \
  --prototype_task_prior_temperature "${TASK_PRIOR_TEMPERATURE:-1.0}" \
  --eval \
  --output_dir "${RUN_DIR}" \
  2>&1 | tee "${LOG_PATH}"

echo "Final prototype evaluation metrics:"
grep "Average accuracy till task10" "${LOG_PATH}" | tail -n 1 || true
