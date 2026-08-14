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
CANDIDATE_DIR="${1:?pass the candidate TII checkpoint directory}"
TASK_COUNT="${2:-3}"
BASELINE_DIR="${BASELINE_DIR:-${OUTPUT_ROOT}/imr_tii_original_10tasks_seed${SEED}}"
EVAL_TAG="${EVAL_TAG:-taskacc_v1}"

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
if [[ ! -d "${IMR_DATA_PATH}/imagenet-r" && ! -f "${IMR_DATA_PATH}/imagenet-r.tar" ]]; then
  echo "ImageNet-R was not found under: ${IMR_DATA_PATH}" >&2
  exit 1
fi

require_checkpoints() {
  local directory="$1"
  local task_id
  for task_id in $(seq 1 "${TASK_COUNT}"); do
    if [[ ! -s "${directory}/checkpoint/task${task_id}_checkpoint.pth" ]]; then
      echo "Missing checkpoint: ${directory}/checkpoint/task${task_id}_checkpoint.pth" >&2
      exit 2
    fi
  done
}

eval_checkpoint_sequence() {
  local label="$1"
  local directory="$2"
  local log_path="$3"

  if [[ -s "${log_path}" ]]; then
    echo "Refusing to overwrite evaluation log: ${log_path}" >&2
    echo "Set EVAL_TAG to a new value." >&2
    exit 3
  fi

  echo "Evaluating ${label}: ${directory}"
  cd "${REPO_ROOT}"
  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node="${GPUS}" \
    --master_port="$4" \
    main.py \
    imr_hideprompt_5e \
    --model vit_base_patch16_224 \
    --original_model vit_base_patch16_224 \
    --batch-size 64 \
    --data-path "${IMR_DATA_PATH}" \
    --seed "${SEED}" \
    --num_tasks 10 \
    --max_train_tasks "${TASK_COUNT}" \
    --train_inference_task_only \
    --strict_exemplar_free \
    --eval \
    --output_dir "${directory}" \
    2>&1 | tee "${log_path}"
}

require_checkpoints "${BASELINE_DIR}"
require_checkpoints "${CANDIDATE_DIR}"

BASELINE_LOG="${OUTPUT_ROOT}/$(basename "${BASELINE_DIR}")_eval_${EVAL_TAG}_task${TASK_COUNT}.log"
CANDIDATE_LOG="${OUTPUT_ROOT}/$(basename "${CANDIDATE_DIR}")_eval_${EVAL_TAG}_task${TASK_COUNT}.log"

eval_checkpoint_sequence baseline "${BASELINE_DIR}" "${BASELINE_LOG}" "${MASTER_PORT_BASELINE:-29546}"
eval_checkpoint_sequence candidate "${CANDIDATE_DIR}" "${CANDIDATE_LOG}" "${MASTER_PORT_CANDIDATE:-29547}"

BASELINE_ROW=$(grep "Average accuracy till task${TASK_COUNT}" "${BASELINE_LOG}" | tail -n 1 || true)
CANDIDATE_ROW=$(grep "Average accuracy till task${TASK_COUNT}" "${CANDIDATE_LOG}" | tail -n 1 || true)
if [[ -z "${BASELINE_ROW}" || -z "${CANDIDATE_ROW}" ]]; then
  echo "Could not find comparable task-${TASK_COUNT} rows." >&2
  exit 4
fi

export BASELINE_ROW CANDIDATE_ROW
"${PYTHON_BIN}" - <<'PY'
import os
import re


def metric(row, name):
    match = re.search(rf"{re.escape(name)}:\s*(-?[0-9.]+)", row)
    if not match:
        raise RuntimeError(f"Missing {name} in: {row}")
    return float(match.group(1))


baseline = os.environ['BASELINE_ROW']
candidate = os.environ['CANDIDATE_ROW']
higher = ['Acc@task', 'Acc@1', 'Acc@5', 'Backward']
lower = ['Loss', 'Forgetting']
deltas = {
    name: metric(candidate, name) - metric(baseline, name)
    for name in higher + lower
}
checks = {
    **{name: deltas[name] >= 0.0 for name in higher},
    **{name: deltas[name] <= 0.0 for name in lower},
}
routing_metrics = ['Acc@task', 'Acc@1', 'Backward', 'Loss', 'Forgetting']
routing_pass = all(checks[name] for name in routing_metrics)
strict_pass = all(checks.values())

print('TII baseline :', baseline)
print('TII candidate:', candidate)
print('Task-routing evaluation:')
for name in higher + lower:
    direction = 'higher' if name in higher else 'lower'
    status = 'PASS' if checks[name] else 'FAIL'
    print(f'  {name:10s} delta={deltas[name]:+.4f} '
          f'({direction} is better): {status}')
print('TII_ROUTING_GATE=' + ('PASS' if routing_pass else 'FAIL'))
print('STRICT_ALL_METRIC_GATE=' + ('PASS' if strict_pass else 'FAIL'))
PY
