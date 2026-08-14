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
TII_PRIOR_WEIGHT="0.3"
LOGIT_TEMPERATURE="1.0"
INITIAL_COUNT="2"
PROPOSAL_COUNT="2"
TOP_CLASSES="5"

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
LOG_PATH="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_prediction_proposal_oracle_i2_p2_c5.log"
if [[ -s "${LOG_PATH}" ]]; then
  echo "Refusing to overwrite existing audit log: ${LOG_PATH}" >&2
  exit 3
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m pytest -q \
  tests/test_prediction_proposal_audit.py \
  tests/test_progressive_oracle_audit.py \
  tests/test_exemplar_free_protocol.py

echo "Prediction-induced task proposal oracle audit"
echo "Fixed design: TII top-2 + at most 2 tasks proposed by top-5 class predictions"
echo "This audit runs all LoRAs only to reveal the exhaustive winner; deployment cost under audit is 4 LoRAs/sample."
echo "Predeclared gate: recall >=95%, exact agreement >=95%, gain over TII top-4 >=5 points, cost <=4."

START_TIME="$(date +%s)"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT:-29583}" \
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
  --progressive_oracle_audit \
  --progressive_prediction_proposal_audit \
  --prediction_proposal_initial_count "${INITIAL_COUNT}" \
  --prediction_proposal_count "${PROPOSAL_COUNT}" \
  --prediction_proposal_top_classes "${TOP_CLASSES}" \
  --progressive_tii_prior_weight "${TII_PRIOR_WEIGHT}" \
  --progressive_logit_temperature "${LOGIT_TEMPERATURE}" \
  --strict_exemplar_free \
  --eval \
  --output_dir "${RUN_DIR}" \
  2>&1 | tee "${LOG_PATH}"

END_TIME="$(date +%s)"
printf 'Prediction-proposal oracle audit wall time seconds: %s\n' "$((END_TIME - START_TIME))" | tee -a "${LOG_PATH}"
echo "Final prediction-proposal oracle metrics:"
grep "Average accuracy till task10" "${LOG_PATH}" | tail -n 1 || true

"${PYTHON_BIN}" - "${LOG_PATH}" <<'PY'
import re
import sys

log_path = sys.argv[1]
with open(log_path, 'r', encoding='utf-8') as handle:
    rows = [line.strip() for line in handle if 'Average accuracy till task10' in line]
if not rows:
    raise SystemExit('PREDICTION_PROPOSAL_GATE=FAIL (missing task-10 metrics)')
row = rows[-1]

def metric(name):
    match = re.search(rf'{re.escape(name)}:\s*(-?[0-9]+(?:\.[0-9]+)?)', row)
    if not match:
        raise SystemExit(f'PREDICTION_PROPOSAL_GATE=FAIL (missing {name})')
    return float(match.group(1))

recall = metric('ProposalWinnerRecall')
exact = metric('ProposalExactAgreement')
cost = metric('ProposalLoRA/sample')
tii4 = metric('WinnerRecall@4')
gain = recall - tii4
checks = {
    'winner_recall_at_least_95': recall >= 95.0,
    'exact_agreement_at_least_95': exact >= 95.0,
    'gain_over_tii_top4_at_least_5': gain >= 5.0,
    'lora_per_sample_at_most_4': cost <= 4.0001,
}
print(f'Proposal gain over TII top-4: {gain:+.4f} points')
for name, passed in checks.items():
    print(f'{name}: {"PASS" if passed else "FAIL"}')
print('PREDICTION_PROPOSAL_GATE=' + ('PASS' if all(checks.values()) else 'FAIL'))
PY