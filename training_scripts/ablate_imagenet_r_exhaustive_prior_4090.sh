#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
RUN_DIR="${1:-}"
LOGIT_TEMPERATURE="${2:-1.0}"
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))

if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: $0 RUN_DIR [temperature] [prior ...]" >&2
  exit 64
fi

if [[ "$#" -gt 0 ]]; then
  PRIOR_WEIGHTS=("$@")
else
  PRIOR_WEIGHTS=(0.0 0.05 0.1 0.2)
fi

tag_value() {
  printf '%s' "$1" | tr '.' 'p'
}

RUN_BASENAME="$(basename "${RUN_DIR}")"
TEMP_TAG="$(tag_value "${LOGIT_TEMPERATURE}")"

for prior in "${PRIOR_WEIGHTS[@]}"; do
  PRIOR_TAG="$(tag_value "${prior}")"
  LOG_PATH="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_exhaustive_p${PRIOR_TAG}_t${TEMP_TAG}.log"
  if [[ -s "${LOG_PATH}" ]] && grep -q "Average accuracy till task10" "${LOG_PATH}"; then
    echo "Skipping completed prior=${prior}: ${LOG_PATH}"
    continue
  fi
  echo "Running exhaustive ablation: prior=${prior}, temperature=${LOGIT_TEMPERATURE}"
  MASTER_PORT="${MASTER_PORT:-29555}" \
    bash "${SCRIPT_DIR}/eval_imagenet_r_exhaustive_rematching_4090.sh" \
      "${RUN_DIR}" "${prior}" "${LOGIT_TEMPERATURE}"
done

SUMMARY_PATH="${OUTPUT_ROOT}/${RUN_BASENAME}_exhaustive_prior_ablation_t${TEMP_TAG}.md"
"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_exhaustive_ablation.py" \
  "${OUTPUT_ROOT}" "${RUN_BASENAME}" "${LOGIT_TEMPERATURE}" \
  "${PRIOR_WEIGHTS[@]}" | tee "${SUMMARY_PATH}"

echo "Ablation summary: ${SUMMARY_PATH}"
