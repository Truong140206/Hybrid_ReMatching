#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-imr_lora_rank8_baseline_10tasks_seed${SEED}}"
RUN_DIR="${1:-${OUTPUT_ROOT}/${RUN_NAME}}"
TII_DIR="${TII_DIR:-${OUTPUT_ROOT}/imr_tii_original_10tasks_seed${SEED}}"
TRAIN_LOG="${OUTPUT_ROOT}/$(basename "${RUN_DIR}").log"
LOG_PATH="${OUTPUT_ROOT}/$(basename "${RUN_DIR}")_eval_prediction_closure_oracle_i2_c5_strict.log"
INITIAL_COUNT="2"
TOP_CLASSES="5"
TII_PRIOR_WEIGHT="0.3"
LOGIT_TEMPERATURE="1.0"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 1
fi
for task_id in $(seq 1 10); do
  [[ -s "${RUN_DIR}/checkpoint/task${task_id}_checkpoint.pth" ]] || {
    echo "Missing strict LoRA checkpoint: ${RUN_DIR}/checkpoint/task${task_id}_checkpoint.pth" >&2
    exit 2
  }
  [[ -s "${TII_DIR}/checkpoint/task${task_id}_checkpoint.pth" ]] || {
    echo "Missing TII checkpoint: ${TII_DIR}/checkpoint/task${task_id}_checkpoint.pth" >&2
    exit 2
  }
done
[[ -s "${TRAIN_LOG}" ]] || {
  echo "Missing training log: ${TRAIN_LOG}" >&2
  exit 2
}
if [[ -s "${LOG_PATH}" ]]; then
  echo "Refusing to overwrite existing audit log: ${LOG_PATH}" >&2
  exit 3
fi

if [[ -n "${DATA_PATH:-}" ]]; then
  IMR_DATA_PATH="${DATA_PATH}"
elif [[ -d "${DATASETS_ROOT}/imagenet-r/imagenet-r" || -f "${DATASETS_ROOT}/imagenet-r/imagenet-r.tar" ]]; then
  IMR_DATA_PATH="${DATASETS_ROOT}/imagenet-r"
else
  IMR_DATA_PATH="${DATASETS_ROOT}"
fi

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -c '
import sys
import torch
from protocols import validate_exemplar_free_training_log
validate_exemplar_free_training_log(sys.argv[1])
checkpoint = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
if checkpoint.get("real_feature_memory"):
    raise SystemExit("STRICT_CHECKPOINT_AUDIT=FAIL (real_feature_memory present)")
saved_args = checkpoint.get("args")
rank = getattr(saved_args, "lora_rank", None)
if rank != 8:
    raise SystemExit(f"STRICT_CHECKPOINT_AUDIT=FAIL (expected LoRA rank 8, got {rank})")
print("STRICT_CHECKPOINT_AUDIT=PASS")
' "${TRAIN_LOG}" "${RUN_DIR}/checkpoint/task10_checkpoint.pth"

"${PYTHON_BIN}" -m pytest -q \
  tests/test_prediction_proposal_audit.py \
  tests/test_progressive_oracle_audit.py \
  tests/test_exemplar_free_protocol.py

echo "Prediction-closure oracle audit"
echo "Locked design: TII top-2, then add all unseen task owners from each newly evaluated LoRA's top-5 predictions until closure."
echo "No confidence threshold, learned gate, old samples, stored per-example features, or label-conditioned routing."
echo "This audit runs all LoRAs only to reveal the exhaustive reference; it does not deploy the closure router."
echo "Predeclared gate: winner/exact >=99.5%, top-5 coverage >=99%, LoRA/sample <=7, calls/sample <=3, full-scan <=20%."

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
  --progressive_oracle_audit \
  --progressive_prediction_closure_audit \
  --prediction_proposal_initial_count "${INITIAL_COUNT}" \
  --prediction_proposal_top_classes "${TOP_CLASSES}" \
  --progressive_tii_prior_weight "${TII_PRIOR_WEIGHT}" \
  --progressive_logit_temperature "${LOGIT_TEMPERATURE}" \
  --strict_exemplar_free \
  --eval \
  --output_dir "${RUN_DIR}" \
  2>&1 | tee "${LOG_PATH}"

END_TIME="$(date +%s)"
printf 'Prediction-closure oracle audit wall time seconds: %s\n' "$((END_TIME - START_TIME))" | tee -a "${LOG_PATH}"
echo "Final prediction-closure oracle metrics:"
grep "Average accuracy till task10" "${LOG_PATH}" | tail -n 1 || true

"${PYTHON_BIN}" - "${LOG_PATH}" <<'PY'
import re
import sys

log_path = sys.argv[1]
with open(log_path, 'r', encoding='utf-8') as handle:
    rows = [line.strip() for line in handle if 'Average accuracy till task10' in line]
if not rows:
    raise SystemExit('PREDICTION_CLOSURE_GATE=FAIL (missing task-10 metrics)')
row = rows[-1]

def metric(name):
    match = re.search(rf'{re.escape(name)}:\s*(-?[0-9]+(?:\.[0-9]+)?)', row)
    if not match:
        raise SystemExit(f'PREDICTION_CLOSURE_GATE=FAIL (missing {name})')
    return float(match.group(1))

checks = {
    'winner_recall_at_least_99p5': metric('ClosureWinnerRecall') >= 99.5,
    'exact_agreement_at_least_99p5': metric('ClosureExactAgreement') >= 99.5,
    'top5_coverage_at_least_99': metric('ClosureTop5Coverage') >= 99.0,
    'lora_per_sample_at_most_7': metric('ClosureLoRA/sample') <= 7.0001,
    'calls_per_sample_at_most_3': metric('ClosureCalls/sample') <= 3.0001,
    'full_scan_rate_at_most_20': metric('ClosureFullScanRate') <= 20.0,
}
for name, passed in checks.items():
    print(f'{name}: {"PASS" if passed else "FAIL"}')
print('PREDICTION_CLOSURE_GATE=' + ('PASS' if all(checks.values()) else 'FAIL'))
PY
