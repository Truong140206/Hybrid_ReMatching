#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-pilot}"
if [[ "${MODE}" != "pilot" && "${MODE}" != "full" ]]; then
  echo "Usage: $0 [pilot|full]" >&2
  exit 64
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
SEED="${SEED:-42}"
GPUS="${GPUS:-1}"

if [[ -n "${DATA_PATH:-}" ]]; then
  IMR_DATA_PATH="${DATA_PATH}"
elif [[ -d "${DATASETS_ROOT}/imagenet-r/imagenet-r" || -f "${DATASETS_ROOT}/imagenet-r/imagenet-r.tar" ]]; then
  IMR_DATA_PATH="${DATASETS_ROOT}/imagenet-r"
else
  IMR_DATA_PATH="${DATASETS_ROOT}"
fi

TII_DIR="${TII_DIR:-${OUTPUT_ROOT}/imr_tii_original_10tasks_seed${SEED}}"
BASELINE_LOG="${BASELINE_LOG:-${OUTPUT_ROOT}/imr_lora_rank8_baseline_10tasks_seed${SEED}.log}"
MAX_TRAIN_TASKS=10
RUN_NAME="${RUN_NAME_OVERRIDE:-imr_rank8_aligned_ctird_mean_10tasks_seed${SEED}}"

if [[ "${MODE}" == "pilot" ]]; then
  MAX_TRAIN_TASKS="${PILOT_TASKS:-3}"
  RUN_NAME="${RUN_NAME_OVERRIDE:-imr_rank8_aligned_ctird_mean_pilot${MAX_TRAIN_TASKS}_seed${SEED}}"
fi

OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_PATH="${OUTPUT_ROOT}/${RUN_NAME}.log"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -d "${IMR_DATA_PATH}/imagenet-r" && ! -f "${IMR_DATA_PATH}/imagenet-r.tar" ]]; then
  echo "ImageNet-R was not found under: ${IMR_DATA_PATH}" >&2
  exit 2
fi
for task_id in $(seq 1 10); do
  checkpoint="${TII_DIR}/checkpoint/task${task_id}_checkpoint.pth"
  if [[ ! -s "${checkpoint}" ]]; then
    echo "Missing TII checkpoint: ${checkpoint}" >&2
    exit 3
  fi
done
if [[ -d "${OUTPUT_DIR}/checkpoint" ]] && compgen -G "${OUTPUT_DIR}/checkpoint/task*_checkpoint.pth" > /dev/null; then
  echo "Refusing to overwrite checkpoints in: ${OUTPUT_DIR}" >&2
  echo "Set RUN_NAME_OVERRIDE to a new name." >&2
  exit 4
fi
if [[ -s "${LOG_PATH}" ]]; then
  echo "Refusing to overwrite existing log: ${LOG_PATH}" >&2
  echo "Set RUN_NAME_OVERRIDE to a new name." >&2
  exit 4
fi
if [[ "${MODE}" == "pilot" ]]; then
  if [[ ! -s "${BASELINE_LOG}" ]]; then
    echo "Baseline log was not found: ${BASELINE_LOG}" >&2
    exit 6
  fi
  if ! grep -q "Average accuracy till task${MAX_TRAIN_TASKS}" "${BASELINE_LOG}"; then
    echo "Baseline log has no comparable task-${MAX_TRAIN_TASKS} row." >&2
    exit 6
  fi
fi

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

echo "Mode: ${MODE}"
echo "Dataset: ${IMR_DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Method: rank-8 HRM-PET with online batch-aligned CTIRD"
echo "Only method change: aligned top-K CTIRD with mean rank reduction"
echo "Online CTIRD reduction: ${CTIRD_ONLINE_REDUCTION:-mean}"
echo "CFS/semantic/prototype/exhaustive: disabled"

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT:-29571}" \
  main.py \
  imr_lora \
  --model vit_base_patch16_224 \
  --original_model vit_base_patch16_224 \
  --batch-size "${LORA_BATCH_SIZE:-24}" \
  --epochs "${LORA_EPOCHS:-50}" \
  --data-path "${IMR_DATA_PATH}" \
  --ca_lr "${CA_LR:-0.005}" \
  --crct_epochs "${CRCT_EPOCHS:-30}" \
  --seed "${SEED}" \
  --lr 0.03 \
  --con "${CTIRD_WEIGHT:-0.2}" \
  --lora_rank 8 \
  --En gen \
  --tau -10 \
  --K "${CTIRD_TOP_K:-5}" \
  --sched cosine \
  --dataset Split-Imagenet-R \
  --lora_momentum 0.4 \
  --lora_type hide \
  --trained_original_model "${TII_DIR}" \
  --num_tasks 10 \
  --max_train_tasks "${MAX_TRAIN_TASKS}" \
  --ctird_online_aligned \
  --ctird_online_temperature "${CTIRD_ONLINE_TEMPERATURE:-1.0}" \
  --ctird_online_ranks_per_batch "${CTIRD_RANKS_PER_BATCH:-1}" \
  --ctird_online_reduction "${CTIRD_ONLINE_REDUCTION:-mean}" \
  --output_dir "${OUTPUT_DIR}" \
  2>&1 | tee "${LOG_PATH}"

for task_id in $(seq 1 "${MAX_TRAIN_TASKS}"); do
  checkpoint="${OUTPUT_DIR}/checkpoint/task${task_id}_checkpoint.pth"
  if [[ ! -s "${checkpoint}" ]]; then
    echo "Missing checkpoint after training: ${checkpoint}" >&2
    exit 5
  fi
done

FINAL_ROW=$(grep "Average accuracy till task${MAX_TRAIN_TASKS}" "${LOG_PATH}" | tail -n 1 || true)
echo "Final metrics:"
echo "${FINAL_ROW}"

if [[ "${MODE}" == "pilot" ]]; then
  if [[ ! -s "${BASELINE_LOG}" ]]; then
    echo "Pilot completed, but baseline log was not found: ${BASELINE_LOG}" >&2
    exit 6
  fi
  BASELINE_ROW=$(grep "Average accuracy till task${MAX_TRAIN_TASKS}" "${BASELINE_LOG}" | tail -n 1 || true)
  if [[ -z "${BASELINE_ROW}" || -z "${FINAL_ROW}" ]]; then
    echo "Could not find comparable task-${MAX_TRAIN_TASKS} rows." >&2
    exit 6
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
higher = ['Acc@task', 'Acc@1', 'Acc@5', 'Backward']
lower = ['Loss', 'Forgetting']
deltas = {name: metric(candidate, name) - metric(baseline, name)
          for name in higher + lower}
checks = {**{name: deltas[name] >= 0.0 for name in higher},
          **{name: deltas[name] <= 0.0 for name in lower}}

print('Pilot baseline :', baseline)
print('Pilot candidate:', candidate)
print('Diagnostic multi-metric comparison:')
for name in higher + lower:
    direction = 'higher' if name in higher else 'lower'
    status = 'PASS' if checks[name] else 'FAIL'
    print(f'  {name:10s} delta={deltas[name]:+.4f} '
          f'({direction} is better): {status}')
print('PILOT_GATE=' + ('PASS' if all(checks.values()) else 'FAIL'))
PY
fi
