#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
SEED="${SEED:-43}"
if [[ "${SEED}" == "42" ]]; then
  echo "Seed 42 is closed for development; use seed 43." >&2
  exit 64
fi

RUN_NAME="${RUN_NAME:-imr_lora_rank8_baseline_10tasks_seed${SEED}}"
RUN_DIR="${1:-${OUTPUT_ROOT}/${RUN_NAME}}"
TII_DIR="${TII_DIR:-${OUTPUT_ROOT}/imr_tii_original_10tasks_seed${SEED}}"
TRAIN_LOG="${OUTPUT_ROOT}/$(basename "${RUN_DIR}").log"
CONVENTIONAL_LOG="${OUTPUT_ROOT}/$(basename "${RUN_DIR}")_eval_conventional.log"
OWNER_LOG="${OUTPUT_ROOT}/$(basename "${RUN_DIR}")_eval_owner_mass_consensus_cap5_i2_c5_dev.log"
STAGE_DRIFT_REPORT="${OUTPUT_ROOT}/imagenet_r_seed${SEED}_stage_drift.md"
LOG_PATH="${OUTPUT_ROOT}/$(basename "${RUN_DIR}")_eval_owner_consensus_hedge_i2_c5_dev.log"

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
for required_path in   "${TRAIN_LOG}"   "${CONVENTIONAL_LOG}"   "${OWNER_LOG}"   "${STAGE_DRIFT_REPORT}"; do
  [[ -s "${required_path}" ]] || {
    echo "Missing required reference: ${required_path}" >&2
    exit 2
  }
done
grep -q 'Dominant measured signal: ROUTING_SELECTION_PENALTY'   "${STAGE_DRIFT_REPORT}" || {
  echo "Stage-drift report did not identify routing as dominant." >&2
  exit 2
}
if [[ -s "${LOG_PATH}" ]]; then
  echo "Refusing to overwrite owner-consensus hedge audit log: ${LOG_PATH}" >&2
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
rank = getattr(checkpoint.get("args"), "lora_rank", None)
if rank != 8:
    raise SystemExit(
        f"STRICT_CHECKPOINT_AUDIT=FAIL (expected LoRA rank 8, got {rank})")
print("STRICT_CHECKPOINT_AUDIT=PASS")
' "${TRAIN_LOG}" "${RUN_DIR}/checkpoint/task10_checkpoint.pth"

"${PYTHON_BIN}" -m pytest -q \
  tests/test_prediction_proposal_audit.py \
  tests/test_progressive_oracle_audit.py \
  tests/test_exemplar_free_protocol.py

echo "Owner-consensus hedge: equal probability mixture only on owner-mass reroutes."
echo "The 1/2 + 1/2 prior is fixed by exchange symmetry, not fitted on seed 43."
echo "All non-rerouted rows remain bitwise equal to owner-mass; no new adapter or call."
echo "Locked gate: all six metrics beat conventional; LoRA/calls equal locked owner-mass and remain below 5/4."

START_TIME="$(date +%s)"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT:-29623}" \
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
  --progressive_prediction_owner_consensus_hedge_audit \
  --prediction_proposal_initial_count 2 \
  --prediction_proposal_top_classes 5 \
  --progressive_tii_prior_weight 0.3 \
  --progressive_logit_temperature 1.0 \
  --strict_exemplar_free \
  --eval \
  --output_dir "${RUN_DIR}" \
  2>&1 | tee "${LOG_PATH}"

END_TIME="$(date +%s)"
printf 'Owner-consensus hedge audit wall time seconds: %s\n' \
  "$((END_TIME - START_TIME))" | tee -a "${LOG_PATH}"

"${PYTHON_BIN}" - \
  "${CONVENTIONAL_LOG}" \
  "${OWNER_LOG}" \
  "${LOG_PATH}" <<'PY' | tee -a "${LOG_PATH}"
import re
import sys


def final_row(path):
    with open(path, 'r', encoding='utf-8') as handle:
        rows = [line.strip() for line in handle
                if 'Average accuracy till task10' in line]
    if not rows:
        raise SystemExit(
            f'OWNER_HEDGE_ALL_METRIC_GATE=FAIL (missing task-10 row in {path})')
    return rows[-1]


def metric(row, name):
    match = re.search(
        rf'{re.escape(name)}:\s*(-?[0-9]+(?:\.[0-9]+)?)', row)
    if not match:
        raise SystemExit(
            f'OWNER_HEDGE_ALL_METRIC_GATE=FAIL (missing {name})')
    return float(match.group(1))


baseline = final_row(sys.argv[1])
owner = final_row(sys.argv[2])
proposal = final_row(sys.argv[3])
checks = {}

print('Comparison with development-seed conventional baseline:')
for name in ('Acc@task', 'Acc@1', 'Acc@5', 'Backward'):
    delta = metric(proposal, name) - metric(baseline, name)
    checks[f'baseline_{name}'] = delta > 0.0
    print(f'  {name} delta={delta:+.4f}: '
          f'{"PASS" if checks[f"baseline_{name}"] else "FAIL"}')
for name in ('Loss', 'Forgetting'):
    delta = metric(proposal, name) - metric(baseline, name)
    checks[f'baseline_{name}'] = delta < 0.0
    print(f'  {name} delta={delta:+.4f}: '
          f'{"PASS" if checks[f"baseline_{name}"] else "FAIL"}')

print('Comparison with locked owner-mass:')
for name in ('Acc@task', 'Acc@1', 'Acc@5', 'Backward', 'Loss', 'Forgetting'):
    delta = metric(proposal, name) - metric(owner, name)
    print(f'  {name} delta={delta:+.4f}')

cost = metric(proposal, 'OwnerHedgeLoRA/sample')
calls = metric(proposal, 'OwnerHedgeCalls/sample')
reference_cost = metric(owner, 'OwnerMassLoRA/sample')
reference_calls = metric(owner, 'OwnerMassCalls/sample')
checks['lora_equal_owner'] = abs(cost - reference_cost) <= 0.0001
checks['calls_equal_owner'] = abs(calls - reference_calls) <= 0.0001
checks['lora_below_5'] = cost < 5.0
checks['calls_below_4'] = calls < 4.0
print('Locked efficiency checks:')
for name in (
        'lora_equal_owner', 'calls_equal_owner',
        'lora_below_5', 'calls_below_4'):
    print(f'  {name}: {"PASS" if checks[name] else "FAIL"}')

print('Diagnostic only:')
for name in (
        'OwnerHedgeWinnerRecall',
        'OwnerHedgeExactAgreement',
        'OwnerHedgeRate',
        'OwnerHedgeOwnerAgreement',
        'OwnerHedgeConsensusAgreement'):
    print(f'  {name}: {metric(proposal, name):.4f}')

print('OWNER_HEDGE_ALL_METRIC_GATE=' +
      ('PASS' if all(checks.values()) else 'FAIL'))
PY