#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${1:-}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: $0 RUN_DIR" >&2
  exit 64
fi

WEIGHTS=(0.05 0.10 0.20)
for index in "${!WEIGHTS[@]}"; do
  weight="${WEIGHTS[${index}]}"
  echo "Local-prototype ablation $((index + 1))/${#WEIGHTS[@]}: weight=${weight}"
  SKIP_EXISTING=1 bash "${SCRIPT_DIR}/eval_imagenet_r_local_prototype_fusion_4090.sh" \
    "${RUN_DIR}" "${weight}" 0.07 0.5 0.3 1.0
done

echo "Local-prototype ablation complete. Final task-10 rows:"
RUN_BASENAME="$(basename "${RUN_DIR}")"
grep "Average accuracy till task10" \
  "$(dirname "${RUN_DIR}")/${RUN_BASENAME}"_eval_exhaustive_localproto_*.log \
  | tail -n 3