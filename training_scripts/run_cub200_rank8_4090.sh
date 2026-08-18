#!/usr/bin/env bash
set -euo pipefail

# Train the strict rank-8 HRM-PET pipeline on Split-CUB200 (no CFS/replay),
# reusing the ImageNet-R configs (same 200-class / 10-task / 20-per-task
# structure) with --dataset Split-CUB200. Modes: check | tii | baseline.
# SMOKE=1 trains at 1 epoch under a separate run name.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
DATA_PATH="${DATA_PATH:-${DATASETS_ROOT}}"
SEED="${SEED:-42}"
GPUS="${GPUS:-1}"
MODE="${1:-check}"
SMOKE="${SMOKE:-0}"
LORA_RANK="${LORA_RANK:-8}"

TII_DIR="${TII_DIR:-${OUTPUT_ROOT}/cub200_tii_original_10tasks_seed${SEED}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then echo "Python not found: ${PYTHON_BIN}" >&2; exit 1; fi
mkdir -p "${OUTPUT_ROOT}" "${DATA_PATH}"

require_checkpoints() {
  local run_dir="$1"; local count="$2"; local label="$3"; local task_id
  for task_id in $(seq 1 "${count}"); do
    [[ -s "${run_dir}/checkpoint/task${task_id}_checkpoint.pth" ]] || { echo "Missing ${label} checkpoint task${task_id}" >&2; return 1; }
  done
  echo "${label} checkpoints OK: task1..task${count}"
}
refuse_overwrite() {
  local d="$1"; local l="$2"
  if [[ -d "${d}/checkpoint" ]] && compgen -G "${d}/checkpoint/task*_checkpoint.pth" > /dev/null; then
    echo "Refusing to overwrite checkpoints in: ${d}" >&2; exit 2; fi
  [[ -s "${l}" ]] && { echo "Refusing to overwrite log: ${l}" >&2; exit 2; } || true
}

run_tii() {
  local run_name="cub200_tii_original_10tasks_seed${SEED}"
  local epochs="${TII_EPOCHS:-20}"; local crct_epochs="${TII_CRCT_EPOCHS:-30}"
  if [[ "${SMOKE}" == "1" ]]; then run_name="cub200_tii_smoke_seed${SEED}"; epochs=1; crct_epochs=1; fi
  local tii_dir="${OUTPUT_ROOT}/${run_name}"; local log_path="${OUTPUT_ROOT}/${run_name}.log"
  if [[ "${SMOKE}" != "1" ]] && require_checkpoints "${TII_DIR}" 10 "TII"; then echo "TII complete."; return 0; fi
  refuse_overwrite "${tii_dir}" "${log_path}"; mkdir -p "${tii_dir}"; cd "${REPO_ROOT}"
  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node="${GPUS}" --master_port="${MASTER_PORT:-29550}" \
    main.py imr_hideprompt_5e \
    --model vit_base_patch16_224 --original_model vit_base_patch16_224 \
    --batch-size "${TII_BATCH_SIZE:-64}" --ca_storage_efficient_method covariance \
    --epochs "${epochs}" --data-path "${DATA_PATH}" --lr 0.0005 --ca_lr 0.005 \
    --crct_epochs "${crct_epochs}" --seed "${SEED}" --num_tasks 10 \
    --dataset Split-CUB200 --train_inference_task_only \
    --output_dir "${tii_dir}" 2>&1 | tee "${log_path}"
  echo "TII training done: ${tii_dir}"
}

run_baseline() {
  local run_name="cub200_lora_rank${LORA_RANK}_baseline_10tasks_seed${SEED}"
  local epochs="${LORA_EPOCHS:-50}"; local crct_epochs="${CRCT_EPOCHS:-30}"; local max_tasks="${MAX_TRAIN_TASKS:-0}"
  if [[ "${SMOKE}" == "1" ]]; then run_name="cub200_lora_rank${LORA_RANK}_baseline_smoke_seed${SEED}"; epochs=1; crct_epochs=1; fi
  run_name="${RUN_NAME_OVERRIDE:-${run_name}}"
  local output_dir="${OUTPUT_ROOT}/${run_name}"; local log_path="${OUTPUT_ROOT}/${run_name}.log"
  require_checkpoints "${TII_DIR}" 10 "TII" || { echo "Run '$0 tii' first." >&2; exit 4; }
  refuse_overwrite "${output_dir}" "${log_path}"; mkdir -p "${output_dir}"; cd "${REPO_ROOT}"
  echo "Rank-8 CUB-200 baseline: original HRM-PET training without CFS-PMI replay"
  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node="${GPUS}" --master_port="${MASTER_PORT:-29551}" \
    main.py imr_lora \
    --model vit_base_patch16_224 --original_model vit_base_patch16_224 \
    --batch-size "${LORA_BATCH_SIZE:-24}" --epochs "${epochs}" --data-path "${DATA_PATH}" \
    --ca_lr "${CA_LR:-0.005}" --crct_epochs "${crct_epochs}" --seed "${SEED}" \
    --lr 0.03 --con 0.2 --lora_rank "${LORA_RANK}" --En gen --tau -10 --K 5 --sched cosine \
    --dataset Split-CUB200 --lora_momentum 0.4 --lora_type hide \
    --trained_original_model "${TII_DIR}" --num_tasks 10 --max_train_tasks "${max_tasks}" \
    --output_dir "${output_dir}" 2>&1 | tee "${log_path}"
  local expected=10; [[ "${max_tasks}" -gt 0 ]] && expected="${max_tasks}"
  require_checkpoints "${output_dir}" "${expected}" "baseline LoRA"
  echo "Final metrics:"; grep "Average accuracy till task${expected}" "${log_path}" | tail -n 1 || true
}

case "${MODE}" in
  check)
    echo "Data path (CUB-200 auto-downloads here): ${DATA_PATH}"
    require_checkpoints "${TII_DIR}" 10 "TII" && echo "TII ready; run '$0 baseline'." || echo "Train TII first with '$0 tii'." ;;
  tii) run_tii ;;
  baseline) run_baseline ;;
  *) echo "Usage: $0 [check|tii|baseline]" >&2; exit 64 ;;
esac
