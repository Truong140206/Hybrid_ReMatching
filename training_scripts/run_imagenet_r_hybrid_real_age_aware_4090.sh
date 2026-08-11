#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
SEED="${SEED:-42}"
CRCT_EPOCHS="${CRCT_EPOCHS:-30}"

if [[ "${MODE}" == "smoke" ]]; then
  DEFAULT_RUN_NAME="imr_hybrid_real_ageaware_smoke_seed${SEED}"
else
  DEFAULT_RUN_NAME="imr_lora_hybrid_real_ageaware_crct${CRCT_EPOCHS}_old035_new010_seed${SEED}"
fi
export RUN_NAME_OVERRIDE="${RUN_NAME_OVERRIDE:-${DEFAULT_RUN_NAME}}"

echo "Class-age-aware real replay: old classes=0.35, current classes=0.10"
bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_imagenet_r_hybrid_real_replay_4090.sh" "${MODE}"
