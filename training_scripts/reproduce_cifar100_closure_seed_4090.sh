#!/usr/bin/env bash
set -euo pipefail

# One-command CIFAR-100 pipeline for the closure+tail method:
#   TII -> rank-8 baseline (no CFS) -> conventional HRM-PET -> exhaustive -> closure.
# SMOKE=1 runs the whole pipeline at 1 epoch (10 tasks) to validate scripts/config
# end-to-end (~25 min) before the multi-hour full run. Default seed 42.

MODE="${1:-run}"
SEED="${SEED:-42}"
SMOKE="${SMOKE:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"

if [[ "${SMOKE}" == "1" ]]; then
  TII_DIR="${OUTPUT_ROOT}/cifar100_tii_smoke_seed${SEED}"
  RUN_DIR="${OUTPUT_ROOT}/cifar100_lora_rank8_baseline_smoke_seed${SEED}"
else
  TII_DIR="${OUTPUT_ROOT}/cifar100_tii_original_10tasks_seed${SEED}"
  RUN_DIR="${OUTPUT_ROOT}/cifar100_lora_rank8_baseline_10tasks_seed${SEED}"
fi
export TII_DIR

checkpoints_complete() { local d="$1"; local t; for t in $(seq 1 10); do [[ -s "${d}/checkpoint/task${t}_checkpoint.pth" ]] || return 1; done; }

echo "Seed: ${SEED}; smoke=${SMOKE}"
echo "TII dir: ${TII_DIR}"
echo "Baseline dir: ${RUN_DIR}"
checkpoints_complete "${TII_DIR}" && echo "TII: READY" || echo "TII: MISSING"
checkpoints_complete "${RUN_DIR}" && echo "Baseline: READY" || echo "Baseline: MISSING"
[[ "${MODE}" == "check" ]] && exit 0

cd "${REPO_ROOT}"

if ! checkpoints_complete "${TII_DIR}"; then
  echo "Stage 1/5: train CIFAR-100 TII"
  SEED="${SEED}" SMOKE="${SMOKE}" bash training_scripts/run_cifar100_rank8_4090.sh tii
fi
if ! checkpoints_complete "${RUN_DIR}"; then
  echo "Stage 2/5: train CIFAR-100 rank-8 baseline (no CFS)"
  SEED="${SEED}" SMOKE="${SMOKE}" bash training_scripts/run_cifar100_rank8_4090.sh baseline
fi

CONV_LOG="${OUTPUT_ROOT}/$(basename "${RUN_DIR}")_eval_conventional.log"
EXH_LOG="${OUTPUT_ROOT}/$(basename "${RUN_DIR}")_eval_vectorized_exhaustive_c4_p0p3_t1p0.log"
CLO_LOG="${OUTPUT_ROOT}/$(basename "${RUN_DIR}")_eval_prediction_closure_tii_tail_i2_c5_strict.log"

if [[ ! -s "${CONV_LOG}" ]]; then
  echo "Stage 3/5: conventional HRM-PET evaluation"
  SEED="${SEED}" bash training_scripts/eval_cifar100_conventional_4090.sh "${RUN_DIR}"
fi
if [[ ! -s "${EXH_LOG}" ]]; then
  echo "Stage 4/5: exhaustive evaluation"
  SEED="${SEED}" bash training_scripts/eval_cifar100_exhaustive_4090.sh "${RUN_DIR}" 4 0.3 1.0
fi
if [[ ! -s "${CLO_LOG}" ]]; then
  echo "Stage 5/5: closure + TII tail evaluation"
  SEED="${SEED}" bash training_scripts/eval_cifar100_closure_tii_tail_4090.sh "${RUN_DIR}"
fi

echo "==================== CIFAR-100 summary (seed ${SEED}) ===================="
for label in conventional exhaustive closure; do
  case "${label}" in
    conventional) LOG="${CONV_LOG}" ;;
    exhaustive) LOG="${EXH_LOG}" ;;
    closure) LOG="${CLO_LOG}" ;;
  esac
  printf '%-12s ' "${label}:"
  grep "Average accuracy till task10" "${LOG}" 2>/dev/null | tail -n 1 || echo "(missing)"
done
grep -E "CLOSURE_TII_TAIL_ALL_METRIC_GATE" "${CLO_LOG}" 2>/dev/null | tail -n 1 || true
