#!/usr/bin/env bash
set -uo pipefail

# Roll the hybrid (HRM-PET LoRA as first-session adaptation + routing-free
# second-order head) across every dataset that has trained checkpoints.
# Run directories are discovered by glob, so no paths need to be passed in.
#
# Usage: bash training_scripts/rollout_hybrid_4090.sh [SEED]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
SEED="${1:-${SEED:-42}}"

pick() {
  # Newest directory holding a full 10-task checkpoint set.
  local pattern="$1" d
  for d in $(ls -dt ${pattern} 2>/dev/null); do
    [[ -s "${d}/checkpoint/task10_checkpoint.pth" ]] && { printf '%s' "${d}"; return 0; }
  done
  return 1
}

run_dataset() {
  local key="$1" dataset="$2"
  local run_dir tii_dir
  run_dir="$(pick "${OUTPUT_ROOT}/*${key}*baseline*seed${SEED}")" || {
    echo "[${dataset}] no complete baseline run for seed ${SEED}; skipping"; return 0; }
  tii_dir="$(pick "${OUTPUT_ROOT}/*${key}*tii*seed${SEED}")" || {
    echo "[${dataset}] no complete TII run for seed ${SEED}; skipping"; return 0; }
  echo "==================== ${dataset} (seed ${SEED}) ===================="
  echo "  baseline: $(basename "${run_dir}")"
  echo "  TII:      $(basename "${tii_dir}")"
  DATASET="${dataset}" TII_DIR="${tii_dir}" SEED="${SEED}" CALIBRATE="${CALIBRATE:-1}" \
    bash "${SCRIPT_DIR}/eval_rp_head_any_4090.sh" "${run_dir}" 2>&1 \
    | grep -E "till task10|delta=|GATE|temperature|Refusing|Missing|Error"
}

run_dataset imagenetr Split-Imagenet-R
run_dataset cifar100  Split-CIFAR100
run_dataset cub200    Split-CUB200
echo "==================== rollout done ===================="
