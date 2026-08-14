#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
SEED="${SEED:-42}"

if [[ -n "${DATA_PATH:-}" ]]; then
  IMR_DATA_PATH="${DATA_PATH}"
elif [[ -d "${DATASETS_ROOT}/imagenet-r/imagenet-r" || -f "${DATASETS_ROOT}/imagenet-r/imagenet-r.tar" ]]; then
  IMR_DATA_PATH="${DATASETS_ROOT}/imagenet-r"
else
  IMR_DATA_PATH="${DATASETS_ROOT}"
fi

RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/imr_lora_rank8_baseline_10tasks_seed${SEED}}"
TII_DIR="${TII_DIR:-${OUTPUT_ROOT}/imr_tii_original_10tasks_seed${SEED}}"
LOG_PATH="${LOG_PATH:-${OUTPUT_ROOT}/imr_rank8_cfs_pmi_diagnostic_task1_seed${SEED}.log}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -d "${IMR_DATA_PATH}/imagenet-r" && ! -f "${IMR_DATA_PATH}/imagenet-r.tar" ]]; then
  echo "ImageNet-R was not found under: ${IMR_DATA_PATH}" >&2
  exit 2
fi
if [[ ! -s "${RUN_DIR}/checkpoint/task1_checkpoint.pth" ]]; then
  echo "Missing baseline checkpoint: ${RUN_DIR}/checkpoint/task1_checkpoint.pth" >&2
  exit 3
fi
if [[ -s "${LOG_PATH}" ]]; then
  echo "Refusing to overwrite diagnostic log: ${LOG_PATH}" >&2
  echo "Move the old log or set LOG_PATH to a new path." >&2
  exit 4
fi

cd "${REPO_ROOT}"

echo "Checkpoint: ${RUN_DIR}/checkpoint/task1_checkpoint.pth"
echo "Dataset: ${IMR_DATA_PATH}"
echo "Diagnostic only: no training and no checkpoint changes"
echo "Comparison: Gaussian targets versus paper-style CFS targets"
echo "Reachability test: partial inversion at the ViT patch-token boundary"

START_TIME="$(date +%s)"

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT:-29571}" \
  main.py \
  imr_lora \
  --model vit_base_patch16_224 \
  --original_model vit_base_patch16_224 \
  --batch-size "${BATCH_SIZE:-24}" \
  --epochs 1 \
  --data-path "${IMR_DATA_PATH}" \
  --seed "${SEED}" \
  --lr 0.03 \
  --con 0.2 \
  --lora_rank 8 \
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
  --cfs_pmi_diagnostic \
  --cfs_pmi_diag_task 1 \
  --cfs_pmi_diag_classes "${DIAG_CLASSES:-2}" \
  --cfs_pmi_diag_targets_per_class "${TARGETS_PER_CLASS:-4}" \
  --cfs_pmi_diag_real_samples_per_class "${REAL_SAMPLES_PER_CLASS:-64}" \
  --cfs_pmi_diag_cfs_epochs "${CFS_EPOCHS:-50}" \
  --cfs_pmi_diag_split_block "${SPLIT_BLOCK:-5}" \
  --cfs_pmi_diag_layer_steps "${LAYER_STEPS:-20}" \
  --cfs_pmi_diag_full_steps "${FULL_STEPS:-40}" \
  --eval \
  --output_dir "${RUN_DIR}" \
  2>&1 | tee "${LOG_PATH}"

END_TIME="$(date +%s)"
echo "CFS-PMI diagnostic wall time seconds: $((END_TIME - START_TIME))" | tee -a "${LOG_PATH}"
grep "CFS_PMI_DIAGNOSTIC=" "${LOG_PATH}" | tail -n 1
