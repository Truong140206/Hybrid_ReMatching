#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-check}"
SEED="${2:-${SEED:-43}}"
if [[ "${MODE}" != "check" && "${MODE}" != "prepare" && "${MODE}" != "run" ]]; then
  echo "Usage: $0 [check|prepare|run] [seed]" >&2
  exit 64
fi
if [[ "${SEED}" == "42" ]]; then
  echo "Seed 42 is closed; use development seed 43 or a later holdout seed." >&2
  exit 64
fi
if ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "Seed must be a non-negative integer: ${SEED}" >&2
  exit 64
fi
if [[ "${SEED}" == "43" ]]; then
  SEED_ROLE="development"
else
  SEED_ROLE="holdout"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
RUN_NAME="imr_lora_rank8_baseline_10tasks_seed${SEED}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
TII_DIR="${OUTPUT_ROOT}/imr_tii_original_10tasks_seed${SEED}"
TII_LOG="${OUTPUT_ROOT}/imr_tii_original_10tasks_seed${SEED}.log"
TRAIN_LOG="${OUTPUT_ROOT}/${RUN_NAME}.log"
CONVENTIONAL_LOG="${OUTPUT_ROOT}/${RUN_NAME}_eval_conventional.log"
AUDIT_LOG="${OUTPUT_ROOT}/${RUN_NAME}_eval_prediction_closure_tii_tail_i2_c5_strict.log"
OPERATIONAL_LOG="${OUTPUT_ROOT}/${RUN_NAME}_eval_prediction_closure_tii_tail_operational_i2_c5_strict.log"

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
  echo "Set DATA_PATH to the directory containing imagenet-r/ or imagenet-r.tar." >&2
  exit 1
fi

checkpoints_complete() {
  local directory="$1"
  local task_id
  for task_id in $(seq 1 10); do
    [[ -s "${directory}/checkpoint/task${task_id}_checkpoint.pth" ]] || return 1
  done
}

stage_status() {
  local label="$1"
  local path="$2"
  if [[ -s "${path}" ]]; then
    echo "${label}: READY (${path})"
  else
    echo "${label}: MISSING (${path})"
  fi
}

printf 'Seed: %s (%s)\n' "${SEED}" "${SEED_ROLE}"
printf 'Locked method: closure i2/c5, prior 0.3, temperature 1.0, top-1-safe TII tail\n'
printf 'Run directory: %s\n' "${RUN_DIR}"
if checkpoints_complete "${TII_DIR}"; then
  echo "TII checkpoints: READY"
else
  echo "TII checkpoints: MISSING"
fi
if checkpoints_complete "${RUN_DIR}"; then
  echo "Rank-8 baseline checkpoints: READY"
else
  echo "Rank-8 baseline checkpoints: MISSING"
fi
stage_status "TII training log" "${TII_LOG}"
stage_status "Rank-8 training log" "${TRAIN_LOG}"
stage_status "Conventional evaluation" "${CONVENTIONAL_LOG}"
stage_status "Closure-tail audit" "${AUDIT_LOG}"
stage_status "Closure-tail operational" "${OPERATIONAL_LOG}"

if [[ "${MODE}" == "check" ]]; then
  exit 0
fi

cd "${REPO_ROOT}"
echo "Current-best reference is frozen; routing development is allowed only when SEED_ROLE=development."

if ! checkpoints_complete "${TII_DIR}"; then
  echo "Stage 1/5: training TII for seed ${SEED}."
  SEED="${SEED}" MASTER_PORT="${TII_MASTER_PORT:-29610}" \
    bash training_scripts/run_imagenet_r_4090.sh tii
else
  echo "Stage 1/5: TII already complete; skipping."
fi

if ! checkpoints_complete "${RUN_DIR}"; then
  echo "Stage 2/5: training strict rank-8 conventional baseline for seed ${SEED}."
  SEED="${SEED}" MASTER_PORT="${LORA_MASTER_PORT:-29611}" \
    bash training_scripts/run_imagenet_r_rank8_replay_anchor_4090.sh baseline
else
  echo "Stage 2/5: rank-8 baseline already complete; skipping."
fi

if [[ ! -s "${CONVENTIONAL_LOG}" ]]; then
  echo "Stage 3/5: conventional evaluation."
  SEED="${SEED}" MASTER_PORT="${CONVENTIONAL_MASTER_PORT:-29612}" \
    bash training_scripts/eval_imagenet_r_conventional_4090.sh "${RUN_DIR}"
else
  echo "Stage 3/5: conventional evaluation already complete; skipping."
fi

if [[ "${MODE}" == "prepare" ]]; then
  echo "DEVELOPMENT_ASSETS_READY"
  exit 0
fi

if [[ ! -s "${AUDIT_LOG}" ]]; then
  echo "Stage 4/5: locked closure + TII-tail audit."
  SEED="${SEED}" MASTER_PORT="${AUDIT_MASTER_PORT:-29613}" \
    bash training_scripts/eval_imagenet_r_prediction_closure_tii_tail_4090.sh \
      "${RUN_DIR}"
else
  echo "Stage 4/5: closure-tail audit already complete; skipping."
fi

if ! grep -q '^CLOSURE_TII_TAIL_ALL_METRIC_GATE=PASS$' "${AUDIT_LOG}"; then
  echo "SEED_CLOSURE_REFERENCE_GATE=FAIL"
  echo "The locked method did not pass on seed ${SEED}; operational confirmation is not run." >&2
  exit 10
fi

if [[ ! -s "${OPERATIONAL_LOG}" ]]; then
  echo "Stage 5/5: operational closure + TII-tail confirmation."
  SEED="${SEED}" MASTER_PORT="${OPERATIONAL_MASTER_PORT:-29614}" \
    bash training_scripts/eval_imagenet_r_prediction_closure_tii_tail_operational_4090.sh \
      "${RUN_DIR}"
else
  echo "Stage 5/5: operational evaluation already complete; skipping."
fi

if ! grep -q '^OPERATIONAL_CLOSURE_TII_TAIL_GATE=PASS$' "${OPERATIONAL_LOG}"; then
  echo "SEED_CLOSURE_REFERENCE_GATE=FAIL"
  echo "Operational closure did not pass on seed ${SEED}." >&2
  exit 11
fi

echo "Final seed conventional metrics:"
grep 'Average accuracy till task10' "${CONVENTIONAL_LOG}" | tail -n 1
echo "Final seed operational metrics:"
grep 'Average accuracy till task10' "${OPERATIONAL_LOG}" | tail -n 1
echo "SEED_CLOSURE_REFERENCE_GATE=PASS"
