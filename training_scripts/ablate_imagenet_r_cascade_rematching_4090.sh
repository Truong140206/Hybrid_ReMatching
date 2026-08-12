#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${1:-}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: $0 RUN_DIR" >&2
  exit 64
fi

echo "Cascade ablation 1/3: top-3, equal TII/LoRA evidence"
SELECTIVE_SOURCE=cascade CASCADE_WEIGHT=1.0 \
bash "${SCRIPT_DIR}/eval_imagenet_r_selective_rematching_4090.sh" \
  "${RUN_DIR}" 3 fixed 0.3 1.0

echo "Cascade ablation 2/3: top-3, conservative LoRA evidence"
SELECTIVE_SOURCE=cascade CASCADE_WEIGHT=0.5 \
bash "${SCRIPT_DIR}/eval_imagenet_r_selective_rematching_4090.sh" \
  "${RUN_DIR}" 3 fixed 0.3 1.0

echo "Cascade ablation 3/3: top-4, conservative LoRA evidence"
SELECTIVE_SOURCE=cascade CASCADE_WEIGHT=0.5 \
bash "${SCRIPT_DIR}/eval_imagenet_r_selective_rematching_4090.sh" \
  "${RUN_DIR}" 4 fixed 0.3 1.0

echo "Cascade ablation complete. Final task-10 rows:"
RUN_BASENAME="$(basename "${RUN_DIR}")"
grep "Average accuracy till task10" \
  "$(dirname "${RUN_DIR}")/${RUN_BASENAME}"_eval_selective_*_cascade_*.log \
  | tail -n 3
