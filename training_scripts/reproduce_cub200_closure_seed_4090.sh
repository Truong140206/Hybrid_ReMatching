#!/usr/bin/env bash
set -euo pipefail

# One-command CUB-200 pipeline: TII -> rank-8 baseline (no CFS) -> conventional
# HRM-PET -> exhaustive -> closure+tail. SMOKE=1 runs at 1 epoch (~fast) to
# validate scripts/config end-to-end. Default seed 42.

MODE="${1:-run}"; SEED="${SEED:-42}"; SMOKE="${SMOKE:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"

if [[ "${SMOKE}" == "1" ]]; then
  TII_DIR="${OUTPUT_ROOT}/cub200_tii_smoke_seed${SEED}"
  RUN_DIR="${OUTPUT_ROOT}/cub200_lora_rank8_baseline_smoke_seed${SEED}"
else
  TII_DIR="${OUTPUT_ROOT}/cub200_tii_original_10tasks_seed${SEED}"
  RUN_DIR="${OUTPUT_ROOT}/cub200_lora_rank8_baseline_10tasks_seed${SEED}"
fi
export TII_DIR

complete() { local d="$1"; local t; for t in $(seq 1 10); do [[ -s "${d}/checkpoint/task${t}_checkpoint.pth" ]] || return 1; done; }

echo "Seed: ${SEED}; smoke=${SMOKE}"
echo "TII dir: ${TII_DIR}"; echo "Baseline dir: ${RUN_DIR}"
complete "${TII_DIR}" && echo "TII: READY" || echo "TII: MISSING"
complete "${RUN_DIR}" && echo "Baseline: READY" || echo "Baseline: MISSING"
[[ "${MODE}" == "check" ]] && exit 0
cd "${REPO_ROOT}"

if ! complete "${TII_DIR}"; then
  echo "Stage 1/5: train CUB-200 TII"
  SEED="${SEED}" SMOKE="${SMOKE}" bash training_scripts/run_cub200_rank8_4090.sh tii
fi
if ! complete "${RUN_DIR}"; then
  echo "Stage 2/5: train CUB-200 rank-8 baseline (no CFS)"
  SEED="${SEED}" SMOKE="${SMOKE}" bash training_scripts/run_cub200_rank8_4090.sh baseline
fi

CONV_LOG="${OUTPUT_ROOT}/$(basename "${RUN_DIR}")_eval_conventional.log"
EXH_LOG="${OUTPUT_ROOT}/$(basename "${RUN_DIR}")_eval_vectorized_exhaustive_c4_p0p3_t1p0.log"
CLO_LOG="${OUTPUT_ROOT}/$(basename "${RUN_DIR}")_eval_prediction_closure_tii_tail_i2_c5_strict.log"

[[ -s "${CONV_LOG}" ]] || { echo "Stage 3/5: conventional HRM-PET"; SEED="${SEED}" bash training_scripts/eval_cub200_conventional_4090.sh "${RUN_DIR}"; }
[[ -s "${EXH_LOG}" ]] || { echo "Stage 4/5: exhaustive"; SEED="${SEED}" bash training_scripts/eval_cub200_exhaustive_4090.sh "${RUN_DIR}" 4 0.3 1.0; }
[[ -s "${CLO_LOG}" ]] || { echo "Stage 5/5: closure + TII tail"; SEED="${SEED}" bash training_scripts/eval_cub200_closure_tii_tail_4090.sh "${RUN_DIR}"; }

echo "==================== CUB-200 summary (seed ${SEED}) ===================="
for label in conventional exhaustive closure; do
  case "${label}" in conventional) LOG="${CONV_LOG}";; exhaustive) LOG="${EXH_LOG}";; closure) LOG="${CLO_LOG}";; esac
  printf '%-12s ' "${label}:"; grep "Average accuracy till task10" "${LOG}" 2>/dev/null | tail -n 1 || echo "(missing)"
done
grep -E "CLOSURE_TII_TAIL_ALL_METRIC_GATE" "${CLO_LOG}" 2>/dev/null | tail -n 1 || true
