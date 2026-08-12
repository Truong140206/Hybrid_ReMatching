#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${1:-}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: $0 RUN_DIR" >&2
  exit 64
fi

WEIGHTS=(0.1 0.25 0.5)
for index in "${!WEIGHTS[@]}"; do
  weight="${WEIGHTS[${index}]}"
  echo "TII-class fusion ablation $((index + 1))/${#WEIGHTS[@]}: weight=${weight}"
  SKIP_EXISTING=1 bash "${SCRIPT_DIR}/eval_imagenet_r_tii_class_fusion_4090.sh" \
    "${RUN_DIR}" "${weight}" 1.0 0.5 0.3 1.0
done

echo "TII-class fusion ablation complete. Final task-10 rows:"
RUN_BASENAME="$(basename "${RUN_DIR}")"
grep "Average accuracy till task10" \
  "$(dirname "${RUN_DIR}")/${RUN_BASENAME}"_eval_exhaustive_tiiclass_*.log \
  | tail -n 3