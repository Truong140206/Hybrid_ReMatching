#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
RUN_DIR="${1:-}"
PROFILE="${2:-conservative}"
TII_DIR="${TII_DIR:-${OUTPUT_ROOT}/imr_tii_original_10tasks_seed42}"
LORA_RANK="${LORA_RANK:-8}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: $0 RUN_DIR [conservative|balanced|aggressive]" >&2
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
  "${PYTHON_BIN}" "${REPO_ROOT}/tools/audit_exemplar_free_checkpoint.py" \
    "${RUN_DIR}/checkpoint/task${task_id}_checkpoint.pth"
done

case "${PROFILE}" in
  conservative)
    STAGE1_TII=1.25; STAGE1_ADAPTER=1.00; STAGE1_CLASS=0.75
    STAGE2_TII=0.60; STAGE2_ADAPTER=0.75; STAGE2_CLASS=0.50
    ;;
  balanced)
    STAGE1_TII=1.00; STAGE1_ADAPTER=0.75; STAGE1_CLASS=0.50
    STAGE2_TII=0.35; STAGE2_ADAPTER=0.50; STAGE2_CLASS=0.25
    ;;
  aggressive)
    STAGE1_TII=0.75; STAGE1_ADAPTER=0.50; STAGE1_CLASS=0.25
    STAGE2_TII=0.15; STAGE2_ADAPTER=0.25; STAGE2_CLASS=0.10
    ;;
  *)
    echo "Unknown profile: ${PROFILE}" >&2
    exit 64
    ;;
esac

RUN_BASENAME="$(basename "${RUN_DIR}")"
LOG_PATH="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_progressive_${PROFILE}_p0p3_t1p0.log"
if [[ -s "${LOG_PATH}" ]]; then
  echo "Refusing to overwrite existing evaluation log: ${LOG_PATH}" >&2
  exit 3
fi

cd "${REPO_ROOT}"
echo "Progressive exhaustive rematching: 2 -> 4 -> all seen LoRAs"
echo "Profile=${PROFILE}; run=${RUN_DIR}"
echo "Stage 1 thresholds: TII=${STAGE1_TII}, adapter=${STAGE1_ADAPTER}, class=${STAGE1_CLASS}"
echo "Stage 2 thresholds: TII=${STAGE2_TII}, adapter=${STAGE2_ADAPTER}, class=${STAGE2_CLASS}"

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
  --lora_rank "${LORA_RANK}" \
  --En gen \
  --tau -10 \
  --K 5 \
  --sched cosine \
  --dataset Split-Imagenet-R \
  --lora_momentum 0.4 \
  --lora_type hide \
  --trained_original_model "${TII_DIR}" \
  --num_tasks 10 \
  --strict_exemplar_free \
  --progressive_rematching \
  --progressive_initial_candidates 2 \
  --progressive_intermediate_candidates 4 \
  --progressive_tii_prior_weight 0.3 \
  --progressive_logit_temperature 1.0 \
  --progressive_stage1_tii_margin "${STAGE1_TII}" \
  --progressive_stage1_adapter_margin "${STAGE1_ADAPTER}" \
  --progressive_stage1_class_margin "${STAGE1_CLASS}" \
  --progressive_stage2_tii_margin "${STAGE2_TII}" \
  --progressive_stage2_adapter_margin "${STAGE2_ADAPTER}" \
  --progressive_stage2_class_margin "${STAGE2_CLASS}" \
  --eval \
  --output_dir "${RUN_DIR}" \
  2>&1 | tee "${LOG_PATH}"

END_TIME="$(date +%s)"
printf 'Progressive evaluation wall time seconds: %s\n' "$((END_TIME - START_TIME))" | tee -a "${LOG_PATH}"
echo "Final progressive-rematching metrics:"
grep "Average accuracy till task10" "${LOG_PATH}" | tail -n 1 || true
