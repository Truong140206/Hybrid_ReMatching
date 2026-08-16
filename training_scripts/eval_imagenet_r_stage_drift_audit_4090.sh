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
REFERENCE_LOG="${OUTPUT_ROOT}/$(basename "${RUN_DIR}")_eval_consensus_cap5_closure_i2_c5_dev.log"
LOG_PATH="${OUTPUT_ROOT}/$(basename "${RUN_DIR}")_eval_stage_drift_consensus_cap5_i2_c5_dev.log"
REPORT_PATH="${OUTPUT_ROOT}/imagenet_r_seed${SEED}_stage_drift.md"

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
for required_path in "${TRAIN_LOG}" "${REFERENCE_LOG}"; do
  [[ -s "${required_path}" ]] || {
    echo "Missing required reference: ${required_path}" >&2
    exit 2
  }
done
if [[ -s "${LOG_PATH}" || -s "${REPORT_PATH}" ]]; then
  echo "Refusing to overwrite stage-drift output." >&2
  echo "Existing log/report: ${LOG_PATH} ${REPORT_PATH}" >&2
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
  tests/test_progressive_oracle_audit.py \
  tests/test_analyze_imagenet_r_stage_drift.py \
  tests/test_exemplar_free_protocol.py

echo "Stage-drift audit: reuse exhaustive logits; no extra adapter evaluation."
echo "OwnLocal isolates within-task drift of the true adapter."
echo "OwnSeen adds competition from every seen class under the same true adapter."
echo "Consensus output remains the locked cap-5 method; operational LoRA cost is unchanged."
echo "The report ranks local drift, cross-task competition, and routing penalty."

START_TIME="$(date +%s)"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT:-29619}" \
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
  --progressive_prediction_consensus_closure_audit \
  --stage_drift_audit \
  --prediction_proposal_initial_count 2 \
  --prediction_proposal_top_classes 5 \
  --progressive_tii_prior_weight 0.3 \
  --progressive_logit_temperature 1.0 \
  --strict_exemplar_free \
  --eval \
  --output_dir "${RUN_DIR}" \
  2>&1 | tee "${LOG_PATH}"

END_TIME="$(date +%s)"
printf 'Stage-drift audit wall time seconds: %s\n' \
  "$((END_TIME - START_TIME))" | tee -a "${LOG_PATH}"

"${PYTHON_BIN}" - "${REFERENCE_LOG}" "${LOG_PATH}" <<'PY' | tee -a "${LOG_PATH}"
import re
import sys


def final_row(path):
    with open(path, 'r', encoding='utf-8') as handle:
        rows = [line.strip() for line in handle
                if 'Average accuracy till task10' in line]
    if not rows:
        raise SystemExit(f'Missing task-10 metrics in {path}')
    return rows[-1]


def metric(row, name):
    match = re.search(
        rf'{re.escape(name)}:\s*(-?[0-9]+(?:\.[0-9]+)?)', row)
    if not match:
        raise SystemExit(f'Missing {name}')
    return float(match.group(1))


reference = final_row(sys.argv[1])
audit = final_row(sys.argv[2])
names = (
    'Acc@task', 'Acc@1', 'Acc@5', 'Loss', 'Forgetting', 'Backward',
    'ConsensusClosureLoRA/sample', 'ConsensusClosureCalls/sample',
)
checks = {}
print('Equivalence with locked consensus-cap5 output:')
for name in names:
    delta = metric(audit, name) - metric(reference, name)
    checks[name] = abs(delta) <= 0.0001
    print(f'  {name} delta={delta:+.4f}: '
          f'{"PASS" if checks[name] else "FAIL"}')
print('STAGE_DRIFT_EQUIVALENCE_GATE=' +
      ('PASS' if all(checks.values()) else 'FAIL'))
if not all(checks.values()):
    raise SystemExit(1)
PY

"${PYTHON_BIN}" training_scripts/analyze_imagenet_r_stage_drift.py \
  --log "${LOG_PATH}" \
  --output "${REPORT_PATH}" | tee -a "${LOG_PATH}"

echo "STAGE_DRIFT_AUDIT=COMPLETE"