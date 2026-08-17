#!/usr/bin/env bash
set -euo pipefail

# Re-run the default HRM-PET (DRM+CRM) evaluation with per-sample cost logging.
# Accuracy is deterministic and must match the frozen conventional log; the only
# addition is LoRA/sample and ForwardCalls/sample for a fair cost table.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
RUN_DIR="${1:-}"
LORA_RANK="${LORA_RANK:-}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: $0 RUN_DIR" >&2
  exit 64
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 1
fi

RUN_BASENAME="$(basename "${RUN_DIR}")"
SEED="${SEED:-}"
if [[ -z "${SEED}" && "${RUN_BASENAME}" =~ seed([0-9]+)$ ]]; then
  SEED="${BASH_REMATCH[1]}"
fi
SEED="${SEED:-42}"
TII_DIR="${TII_DIR:-${OUTPUT_ROOT}/imr_tii_original_10tasks_seed${SEED}}"
TRAIN_LOG="${TRAIN_LOG:-${OUTPUT_ROOT}/${RUN_BASENAME}.log}"
CONVENTIONAL_LOG="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_conventional.log"
LOG_PATH="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_conventional_cost.log"

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
if [[ ! -s "${TRAIN_LOG}" ]]; then
  echo "Training log required for protocol audit: ${TRAIN_LOG}" >&2
  exit 2
fi
if [[ -s "${LOG_PATH}" ]]; then
  echo "Refusing to overwrite completed cost log: ${LOG_PATH}" >&2
  exit 3
fi

cd "${REPO_ROOT}"
CHECKPOINT_RANK="$(PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -c '
import sys
import torch
from protocols import validate_exemplar_free_training_log
validate_exemplar_free_training_log(sys.argv[1])
checkpoint = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
state = checkpoint.get("model", checkpoint)
matches = [value for key, value in state.items()
           if key.endswith("lora_layer.k_lora_A")]
if len(matches) != 1:
    raise SystemExit(f"Expected one lora_layer.k_lora_A tensor, found {len(matches)}")
print(int(matches[0].shape[-1]))
' "${TRAIN_LOG}" "${RUN_DIR}/checkpoint/task1_checkpoint.pth")"
if [[ -z "${LORA_RANK}" ]]; then
  LORA_RANK="${CHECKPOINT_RANK}"
fi

echo "Conventional HRM-PET (DRM+CRM) cost measurement"
echo "Run=${RUN_DIR}; seed=${SEED}; rank=${LORA_RANK}"

START_TIME="$(date +%s)"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT:-29592}" \
  main.py \
  imr_lora \
  --model vit_base_patch16_224 \
  --original_model vit_base_patch16_224 \
  --batch-size "${EVAL_BATCH_SIZE:-24}" \
  --epochs 1 \
  --data-path "${IMR_DATA_PATH}" \
  --seed "${SEED}" \
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
  --report_conventional_cost \
  --strict_exemplar_free \
  --eval \
  --output_dir "${RUN_DIR}" \
  2>&1 | tee "${LOG_PATH}"

END_TIME="$(date +%s)"
printf 'Conventional cost measurement wall time seconds: %s\n' "$((END_TIME - START_TIME))" | tee -a "${LOG_PATH}"
echo "Final conventional cost metrics:"
grep "Average accuracy till task10" "${LOG_PATH}" | tail -n 1 || true

if [[ -s "${CONVENTIONAL_LOG}" ]]; then
  "${PYTHON_BIN}" - "${CONVENTIONAL_LOG}" "${LOG_PATH}" <<'PY' | tee -a "${LOG_PATH}"
import re
import sys

def final_row(path):
    with open(path, 'r', encoding='utf-8') as handle:
        rows = [line.strip() for line in handle
                if 'Average accuracy till task10' in line]
    return rows[-1] if rows else ''

def metric(row, name):
    match = re.search(rf'{re.escape(name)}:\s*(-?[0-9]+(?:\.[0-9]+)?)', row)
    return float(match.group(1)) if match else float('nan')

frozen = final_row(sys.argv[1])
cost = final_row(sys.argv[2])
same = all(
    abs(metric(frozen, name) - metric(cost, name)) <= 1e-3
    for name in ('Acc@task', 'Acc@1', 'Acc@5', 'Loss'))
print('Accuracy matches frozen conventional log: ' + ('PASS' if same else 'FAIL'))
print('HRM-PET LoRA/sample: {:.4f}'.format(metric(cost, 'LoRA/sample')))
print('HRM-PET ForwardCalls/sample: {:.4f}'.format(
    metric(cost, 'ForwardCalls/sample')))
print('CONVENTIONAL_COST_GATE=' + ('PASS' if same else 'FAIL'))
PY
fi
