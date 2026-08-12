#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
SEED="${SEED:-42}"
CRCT_EPOCHS="${CRCT_EPOCHS:-30}"

if [[ "${MODE}" == "smoke" ]]; then
  DEFAULT_RUN_NAME="imr_hybrid_semantic_ctird_smoke_seed${SEED}"
else
  DEFAULT_RUN_NAME="imr_lora_hybrid_real_cfs_semantic_ctird_a003_k3_crct${CRCT_EPOCHS}_seed${SEED}"
fi

export RUN_NAME_OVERRIDE="${RUN_NAME_OVERRIDE:-${DEFAULT_RUN_NAME}}"
export SEMANTIC_DISTILL=1
export SEMANTIC_MODE="${SEMANTIC_MODE:-topk_mix}"
export SEMANTIC_TOP_K="${SEMANTIC_TOP_K:-3}"
export SEMANTIC_ALPHA="${SEMANTIC_ALPHA:-0.03}"

echo "Hybrid Real+CFS with CLIP semantic-aware CTIRD"
echo "Semantic relation target: top-k=${SEMANTIC_TOP_K}, alpha=${SEMANTIC_ALPHA}"
bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_imagenet_r_hybrid_real_age_aware_4090.sh" "${MODE}"
