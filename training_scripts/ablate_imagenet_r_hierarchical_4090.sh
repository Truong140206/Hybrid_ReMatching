#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
RUN_DIR="${1:-}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: $0 RUN_DIR" >&2
  exit 64
fi

echo "Hierarchical ablation 1/3: max evidence only"
SKIP_EXISTING=1 bash "${SCRIPT_DIR}/eval_imagenet_r_hierarchical_4090.sh" \
  "${RUN_DIR}" 0.3 1.0 0.0 0.0 1.0 1.0

echo "Hierarchical ablation 2/3: balanced confidence"
SKIP_EXISTING=1 bash "${SCRIPT_DIR}/eval_imagenet_r_hierarchical_4090.sh" \
  "${RUN_DIR}" 0.3 1.0 0.5 0.5 1.0 1.0

echo "Hierarchical ablation 3/3: balanced confidence with sharper task posterior"
SKIP_EXISTING=1 bash "${SCRIPT_DIR}/eval_imagenet_r_hierarchical_4090.sh" \
  "${RUN_DIR}" 0.3 1.0 0.5 0.5 1.0 0.5

echo "Hierarchical ablation complete. Final task-10 rows:"
RUN_BASENAME="$(basename "${RUN_DIR}")"
grep "Average accuracy till task10" \
  "$(dirname "${RUN_DIR}")/${RUN_BASENAME}"_eval_hierarchical_*.log \
  | tail -n 3