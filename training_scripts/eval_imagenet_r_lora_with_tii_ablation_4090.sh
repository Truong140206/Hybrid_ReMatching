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
LORA_DIR="${LORA_DIR:-${OUTPUT_ROOT}/imr_lora_rank8_baseline_10tasks_seed${SEED}}"
BASELINE_TII_DIR="${BASELINE_TII_DIR:-${OUTPUT_ROOT}/imr_tii_original_10tasks_seed${SEED}}"
CANDIDATE_TII_DIR="${1:?pass the candidate TII checkpoint directory}"
TASK_COUNT="${2:-3}"
EVAL_TAG="${EVAL_TAG:-tii_cfs_e2e_v1}"

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
  local label="$2"
  local task_id
  for task_id in $(seq 1 "${TASK_COUNT}"); do
    if [[ ! -s "${directory}/checkpoint/task${task_id}_checkpoint.pth" ]]; then
      echo "Missing ${label} checkpoint: ${directory}/checkpoint/task${task_id}_checkpoint.pth" >&2
      exit 2
    fi
  done
}

eval_variant() {
  local label="$1"
  local tii_dir="$2"
  local log_path="$3"
  local port="$4"

  if [[ -s "${log_path}" ]]; then
    echo "Refusing to overwrite evaluation log: ${log_path}" >&2
    echo "Set EVAL_TAG to a new value." >&2
    exit 3
  fi

  echo "Evaluating ${label} TII with fixed LoRA checkpoints."
  cd "${REPO_ROOT}"
  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node="${GPUS}" \
    --master_port="${port}" \
    main.py \
    imr_lora \
    --model vit_base_patch16_224 \
    --original_model vit_base_patch16_224 \
    --batch-size 24 \
    --data-path "${IMR_DATA_PATH}" \
    --seed "${SEED}" \
    --num_tasks 10 \
    --max_train_tasks "${TASK_COUNT}" \
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
    --trained_original_model "${tii_dir}" \
    --strict_exemplar_free \
    --eval \
    --output_dir "${LORA_DIR}" \
    2>&1 | tee "${log_path}"
}

require_checkpoints "${LORA_DIR}" "LoRA"
require_checkpoints "${BASELINE_TII_DIR}" "baseline TII"
require_checkpoints "${CANDIDATE_TII_DIR}" "candidate TII"

BASELINE_LOG="${OUTPUT_ROOT}/$(basename "${LORA_DIR}")_${EVAL_TAG}_baseline_task${TASK_COUNT}.log"
CANDIDATE_LOG="${OUTPUT_ROOT}/$(basename "${LORA_DIR}")_${EVAL_TAG}_candidate_task${TASK_COUNT}.log"

eval_variant baseline "${BASELINE_TII_DIR}" "${BASELINE_LOG}" "${MASTER_PORT_BASELINE:-29548}"
eval_variant candidate "${CANDIDATE_TII_DIR}" "${CANDIDATE_LOG}" "${MASTER_PORT_CANDIDATE:-29549}"

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

print('Fixed-LoRA baseline TII :', baseline)
print('Fixed-LoRA CFS TII      :', candidate)
print('End-to-end TII replacement comparison:')
for name in higher + lower:
    direction = 'higher' if name in higher else 'lower'
    status = 'PASS' if checks[name] else 'FAIL'
    print(f'  {name:10s} delta={deltas[name]:+.4f} '
          f'({direction} is better): {status}')
print('END_TO_END_GATE=' + ('PASS' if all(checks.values()) else 'FAIL'))
PY
