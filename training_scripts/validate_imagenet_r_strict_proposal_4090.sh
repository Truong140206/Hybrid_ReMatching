#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-run}"
if [[ "${MODE}" != "check" && "${MODE}" != "run" ]]; then
  echo "Usage: $0 [check|run]" >&2
  exit 64
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-imr_lora_rank8_baseline_10tasks_seed${SEED}}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
TRAIN_LOG="${OUTPUT_ROOT}/${RUN_NAME}.log"
CONVENTIONAL_LOG="${OUTPUT_ROOT}/${RUN_NAME}_eval_conventional.log"
EXHAUSTIVE_LOG="${OUTPUT_ROOT}/${RUN_NAME}_eval_vectorized_exhaustive_c4_p0p3_t1p0.log"
PROPOSAL_LOG="${OUTPUT_ROOT}/${RUN_NAME}_eval_prediction_proposal_i2_p3_c5_tiicomplete_strict.log"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 1
fi
for task_id in $(seq 1 10); do
  [[ -s "${RUN_DIR}/checkpoint/task${task_id}_checkpoint.pth" ]] || {
    echo "Missing strict baseline checkpoint: ${RUN_DIR}/checkpoint/task${task_id}_checkpoint.pth" >&2
    echo "Train it with: SEED=${SEED} bash training_scripts/run_imagenet_r_rank8_replay_anchor_4090.sh baseline" >&2
    exit 2
  }
done
[[ -s "${TRAIN_LOG}" ]] || {
  echo "Missing training log: ${TRAIN_LOG}" >&2
  exit 2
}

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -c '
import sys
import torch
from protocols import validate_exemplar_free_training_log
validate_exemplar_free_training_log(sys.argv[1])
checkpoint = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
if checkpoint.get("real_feature_memory"):
    raise SystemExit("STRICT_CHECKPOINT_AUDIT=FAIL (real_feature_memory present)")
print("STRICT_CHECKPOINT_AUDIT=PASS")
' "${TRAIN_LOG}" "${RUN_DIR}/checkpoint/task10_checkpoint.pth"

printf 'Seed: %s\nRun: %s\n' "${SEED}" "${RUN_DIR}"
for item in \
    "Conventional:${CONVENTIONAL_LOG}" \
    "Exhaustive:${EXHAUSTIVE_LOG}" \
    "Proposal:${PROPOSAL_LOG}"; do
  label="${item%%:*}"
  path="${item#*:}"
  if [[ -s "${path}" ]]; then
    echo "${label}: READY (${path})"
  else
    echo "${label}: MISSING (${path})"
  fi
done

if [[ "${MODE}" == "check" ]]; then
  exit 0
fi

"${PYTHON_BIN}" -m pytest -q \
  tests/test_exemplar_free_protocol.py \
  tests/test_prediction_proposal_audit.py \
  tests/test_progressive_oracle_audit.py \
  tests/test_summarize_imagenet_r_multiseed.py

if [[ ! -s "${CONVENTIONAL_LOG}" ]]; then
  SEED="${SEED}" bash training_scripts/eval_imagenet_r_conventional_4090.sh "${RUN_DIR}"
else
  echo "Conventional evaluation already complete; skipping."
fi

if [[ ! -s "${EXHAUSTIVE_LOG}" ]]; then
  SEED="${SEED}" bash training_scripts/eval_imagenet_r_vectorized_exhaustive_4090.sh \
    "${RUN_DIR}" 4 0.3 1.0
else
  echo "Vectorized exhaustive evaluation already complete; skipping."
fi

if [[ ! -s "${PROPOSAL_LOG}" ]]; then
  SEED="${SEED}" bash training_scripts/eval_imagenet_r_prediction_proposal5_completion_4090.sh \
    "${RUN_DIR}"
else
  echo "Strict proposal evaluation already complete; skipping."
fi

SUMMARY_PATH="${OUTPUT_ROOT}/imagenet_r_strict_seed${SEED}_summary.md"
"${PYTHON_BIN}" training_scripts/summarize_imagenet_r_multiseed.py \
  --output-root "${OUTPUT_ROOT}" \
  --seeds "${SEED}" \
  --output "${SUMMARY_PATH}"
