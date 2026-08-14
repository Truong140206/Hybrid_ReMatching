#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
RUN_DIR="${1:-}"
TII_DIR="${TII_DIR:-${OUTPUT_ROOT}/imr_tii_original_10tasks_seed42}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: $0 RUN_DIR" >&2
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

RUN_BASENAME="$(basename "${RUN_DIR}")"
LOG_PATH="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_prediction_proposal_i2_p2_c5.log"
if [[ -s "${LOG_PATH}" ]]; then
  echo "Refusing to overwrite existing evaluation log: ${LOG_PATH}" >&2
  exit 3
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m pytest -q \
  tests/test_prediction_proposal_audit.py \
  tests/test_progressive_oracle_audit.py \
  tests/test_exemplar_free_protocol.py

echo "Operational prediction-induced task proposal rematching"
echo "TII top-2 + two post-LoRA class-prediction proposals; no exhaustive fallback"
echo "Fixed deployment budget: at most 4 LoRAs/sample in 2 vectorized model calls"

START_TIME="$(date +%s)"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT:-29587}" \
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
  --prediction_proposal_rematching \
  --prediction_proposal_initial_count 2 \
  --prediction_proposal_count 2 \
  --prediction_proposal_top_classes 5 \
  --progressive_tii_prior_weight 0.3 \
  --progressive_logit_temperature 1.0 \
  --strict_exemplar_free \
  --eval \
  --output_dir "${RUN_DIR}" \
  2>&1 | tee "${LOG_PATH}"

END_TIME="$(date +%s)"
printf 'Prediction-proposal evaluation wall time seconds: %s\n' "$((END_TIME - START_TIME))" | tee -a "${LOG_PATH}"
echo "Final operational prediction-proposal metrics:"
grep "Average accuracy till task10" "${LOG_PATH}" | tail -n 1 || true

"${PYTHON_BIN}" - "${LOG_PATH}" <<'PY'
import re
import sys

BASELINE = {
    'Acc@task': 77.5854,
    'Acc@1': 74.0477,
    'Acc@5': 86.4646,
    'Loss': 1.2230,
    'Forgetting': 3.3264,
    'Backward': -2.9319,
}
EXHAUSTIVE = {
    'Acc@task': 80.6549,
    'Acc@1': 75.1798,
    'Acc@5': 88.5327,
    'Loss': 1.0809,
    'Forgetting': 2.8848,
    'Backward': -2.8449,
}

with open(sys.argv[1], 'r', encoding='utf-8') as handle:
    rows = [line.strip() for line in handle if 'Average accuracy till task10' in line]
if not rows:
    raise SystemExit('OPERATIONAL_PROPOSAL_GATE=FAIL (missing task-10 metrics)')
row = rows[-1]

def metric(name):
    match = re.search(rf'{re.escape(name)}:\s*(-?[0-9]+(?:\.[0-9]+)?)', row)
    if not match:
        raise SystemExit(f'OPERATIONAL_PROPOSAL_GATE=FAIL (missing {name})')
    return float(match.group(1))

higher = ('Acc@task', 'Acc@1', 'Acc@5', 'Backward')
lower = ('Loss', 'Forgetting')
baseline_checks = {}
print('Comparison with conventional baseline:')
for name in higher:
    value = metric(name)
    delta = value - BASELINE[name]
    baseline_checks[name] = delta >= 0.0
    print(f'  {name} delta={delta:+.4f}: {"PASS" if delta >= 0.0 else "FAIL"}')
for name in lower:
    value = metric(name)
    delta = value - BASELINE[name]
    baseline_checks[name] = delta <= 0.0
    print(f'  {name} delta={delta:+.4f}: {"PASS" if delta <= 0.0 else "FAIL"}')

print('Retention relative to exhaustive:')
for name in (*higher, *lower):
    delta = metric(name) - EXHAUSTIVE[name]
    print(f'  {name} delta={delta:+.4f}')

cost_ok = metric('LoRA/sample') <= 4.0001
call_ok = metric('ForwardCalls/sample') <= 2.0001
print(f'  LoRA/sample <= 4: {"PASS" if cost_ok else "FAIL"}')
print(f'  ForwardCalls/sample <= 2: {"PASS" if call_ok else "FAIL"}')
print('BASELINE_ALL_METRIC_GATE=' + (
    'PASS' if all(baseline_checks.values()) else 'FAIL'))
print('OPERATIONAL_PROPOSAL_EFFICIENCY_GATE=' + (
    'PASS' if cost_ok and call_ok else 'FAIL'))
PY