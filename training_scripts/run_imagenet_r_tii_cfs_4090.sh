#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
SEED="${SEED:-42}"
GPUS="${GPUS:-1}"
MODE="${1:-pilot}"
CFS_SELECTION_RATIO="${CFS_SELECTION_RATIO:-0.5}"

if [[ -n "${DATA_PATH:-}" ]]; then
  IMR_DATA_PATH="${DATA_PATH}"
elif [[ -d "${DATASETS_ROOT}/imagenet-r/imagenet-r" || -f "${DATASETS_ROOT}/imagenet-r/imagenet-r.tar" ]]; then
  IMR_DATA_PATH="${DATASETS_ROOT}/imagenet-r"
else
  IMR_DATA_PATH="${DATASETS_ROOT}"
fi

BASELINE_LOG="${BASELINE_LOG:-${OUTPUT_ROOT}/imr_tii_original_10tasks_seed${SEED}.log}"

case "${MODE}" in
  pilot)
    TASK_COUNT=3
    DEFAULT_RUN_NAME="imr_tii_cfs_paper_pilot3_seed${SEED}"
    ;;
  full)
    TASK_COUNT=10
    DEFAULT_RUN_NAME="imr_tii_cfs_paper_10tasks_seed${SEED}"
    ;;
  *)
    echo "Usage: $0 [pilot|full]" >&2
    exit 64
    ;;
esac

RUN_NAME="${RUN_NAME_OVERRIDE:-${DEFAULT_RUN_NAME}}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_PATH="${OUTPUT_ROOT}/${RUN_NAME}.log"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -d "${IMR_DATA_PATH}/imagenet-r" && ! -f "${IMR_DATA_PATH}/imagenet-r.tar" ]]; then
  echo "ImageNet-R was not found under: ${IMR_DATA_PATH}" >&2
  exit 1
fi
if [[ ! -s "${BASELINE_LOG}" ]]; then
  echo "Original TII baseline log not found: ${BASELINE_LOG}" >&2
  exit 1
fi
if [[ -d "${OUTPUT_DIR}/checkpoint" ]] && compgen -G "${OUTPUT_DIR}/checkpoint/task*_checkpoint.pth" > /dev/null; then
  echo "Refusing to overwrite checkpoints in: ${OUTPUT_DIR}" >&2
  echo "Set RUN_NAME_OVERRIDE to a new name." >&2
  exit 2
fi
if [[ -s "${LOG_PATH}" ]]; then
  echo "Refusing to overwrite existing log: ${LOG_PATH}" >&2
  echo "Set RUN_NAME_OVERRIDE to a new name." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

echo "Running CFS in TII task inference, not in LoRA CRCT."
echo "Mode: ${MODE}; tasks: ${TASK_COUNT}; output: ${OUTPUT_DIR}"
echo "CFS selection ratio: ${CFS_SELECTION_RATIO}"

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT:-29545}" \
  main.py \
  imr_hideprompt_5e \
  --model vit_base_patch16_224 \
  --original_model vit_base_patch16_224 \
  --batch-size 64 \
  --ca_storage_efficient_method covariance \
  --epochs 20 \
  --data-path "${IMR_DATA_PATH}" \
  --lr 0.0005 \
  --ca_lr 0.005 \
  --crct_epochs 30 \
  --seed "${SEED}" \
  --num_tasks 10 \
  --max_train_tasks "${TASK_COUNT}" \
  --train_inference_task_only \
  --strict_exemplar_free \
  --cfs_sampling \
  --cfs_epochs 200 \
  --cfs_train_max_samples 1024 \
  --cfs_candidate_multiplier 3 \
  --cfs_paper_style \
  --cfs_selection_ratio "${CFS_SELECTION_RATIO}" \
  --cfs_selection_steps 5 \
  --output_dir "${OUTPUT_DIR}" \
  2>&1 | tee "${LOG_PATH}"

for task_id in $(seq 1 "${TASK_COUNT}"); do
  checkpoint="${OUTPUT_DIR}/checkpoint/task${task_id}_checkpoint.pth"
  if [[ ! -s "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 3
  fi
  "${PYTHON_BIN}" tools/audit_exemplar_free_checkpoint.py "${checkpoint}"
done

FINAL_ROW=$(grep "Average accuracy till task${TASK_COUNT}" "${LOG_PATH}" | tail -n 1 || true)
if [[ -z "${FINAL_ROW}" ]]; then
  echo "Final task-${TASK_COUNT} metrics were not found in ${LOG_PATH}" >&2
  exit 4
fi
echo "Final CFS-TII metrics:"
echo "${FINAL_ROW}"

if [[ "${MODE}" == "pilot" ]]; then
  BASELINE_ROW=$(grep "Average accuracy till task3" "${BASELINE_LOG}" | tail -n 1 || true)
  if [[ -z "${BASELINE_ROW}" ]]; then
    echo "Task-3 baseline metrics were not found in ${BASELINE_LOG}" >&2
    exit 5
  fi
  export BASELINE_ROW FINAL_ROW
  "${PYTHON_BIN}" - <<'PY'
import os
import re


def metric(row, name):
    match = re.search(rf"{re.escape(name)}:\s*(-?[0-9.]+)", row)
    if not match:
        raise RuntimeError(f"Missing {name} in: {row}")
    return float(match.group(1))


baseline = os.environ['BASELINE_ROW']
candidate = os.environ['FINAL_ROW']
higher = ['Acc@1', 'Acc@5', 'Backward']
lower = ['Loss', 'Forgetting']
deltas = {
    name: metric(candidate, name) - metric(baseline, name)
    for name in higher + lower
}
checks = {
    **{name: deltas[name] >= 0.0 for name in higher},
    **{name: deltas[name] <= 0.0 for name in lower},
}

print('Pilot reference:', baseline)
print('Pilot candidate:', candidate)
print('Strict five-metric comparison:')
for name in higher + lower:
    direction = 'higher' if name in higher else 'lower'
    status = 'PASS' if checks[name] else 'FAIL'
    print(f'  {name:10s} delta={deltas[name]:+.4f} '
          f'({direction} is better): {status}')
print('PILOT_GATE=' + ('PASS' if all(checks.values()) else 'FAIL'))
PY
fi
