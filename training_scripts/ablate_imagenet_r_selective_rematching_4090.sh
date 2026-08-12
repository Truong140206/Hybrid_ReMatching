#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${1:-}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: $0 RUN_DIR" >&2
  exit 64
fi

echo "Ablation 1/3: fixed top-2"
bash "${SCRIPT_DIR}/eval_imagenet_r_selective_rematching_4090.sh" \
  "${RUN_DIR}" 2 fixed 0.3 1.0

echo "Ablation 2/3: fixed top-3"
bash "${SCRIPT_DIR}/eval_imagenet_r_selective_rematching_4090.sh" \
  "${RUN_DIR}" 3 fixed 0.3 1.0

echo "Ablation 3/3: adaptive top-3"
bash "${SCRIPT_DIR}/eval_imagenet_r_selective_rematching_4090.sh" \
  "${RUN_DIR}" 3 adaptive 0.3 1.0 1.0 0.35

echo "Selective-rematching ablation complete. Final task-10 rows:"
RUN_BASENAME="$(basename "${RUN_DIR}")"
grep "Average accuracy till task10" \
  "$(dirname "${RUN_DIR}")/${RUN_BASENAME}"_eval_selective_*.log \
  | tail -n 3
