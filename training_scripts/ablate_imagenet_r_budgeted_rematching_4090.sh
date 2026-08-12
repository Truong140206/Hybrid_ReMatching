#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${1:-}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: $0 RUN_DIR" >&2
  exit 64
fi

echo "Budgeted ablation 1/3: 10% fallback"
bash "${SCRIPT_DIR}/eval_imagenet_r_budgeted_rematching_4090.sh" \
  "${RUN_DIR}" 0.1 0.3 1.0

echo "Budgeted ablation 2/3: 20% fallback"
bash "${SCRIPT_DIR}/eval_imagenet_r_budgeted_rematching_4090.sh" \
  "${RUN_DIR}" 0.2 0.3 1.0

echo "Budgeted ablation 3/3: 30% fallback"
bash "${SCRIPT_DIR}/eval_imagenet_r_budgeted_rematching_4090.sh" \
  "${RUN_DIR}" 0.3 0.3 1.0

echo "Budgeted fallback ablation complete. Final task-10 rows:"
RUN_BASENAME="$(basename "${RUN_DIR}")"
grep "Average accuracy till task10" \
  "$(dirname "${RUN_DIR}")/${RUN_BASENAME}"_eval_budgeted_*.log \
  | tail -n 3
