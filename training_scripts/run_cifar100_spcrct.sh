#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATA_PATH="${DATA_PATH:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
TII_DIR="${TII_DIR:-${OUTPUT_ROOT}/cifar100_tii_10tasks_seed42}"
MODE="${1:-smoke}"
SEED="${SEED:-42}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -d "${TII_DIR}/checkpoint" ]]; then
  echo "TII checkpoint directory not found: ${TII_DIR}/checkpoint" >&2
  exit 1
fi

case "${MODE}" in
  smoke)
    NUM_TASKS=2
    EPOCHS=1
    CRCT_EPOCHS=1
    CFS_EPOCHS=1
    MASTER_PORT="${MASTER_PORT:-29525}"
    RUN_NAME="cifar100_spcrct_v4_trust_smoke_seed${SEED}"
    ;;
  full)
    NUM_TASKS=10
    EPOCHS=10
    CRCT_EPOCHS=3
    CFS_EPOCHS=20
    MASTER_PORT="${MASTER_PORT:-29526}"
    RUN_NAME="cifar100_spcrct_v4_adaptive_trust_seed${SEED}"
    ;;
  *)
    echo "Usage: $0 [smoke|full]" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME_OVERRIDE:-${RUN_NAME}}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_PATH="${OUTPUT_ROOT}/${RUN_NAME}.log"
if compgen -G "${OUTPUT_DIR}/checkpoint/task*_checkpoint.pth" > /dev/null; then
  echo "Refusing to overwrite existing checkpoints in: ${OUTPUT_DIR}" >&2
  echo "Set RUN_NAME_OVERRIDE to a new name before running again." >&2
  exit 3
fi

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT}" \
  main.py \
  cifar100_lora \
  --model vit_base_patch16_224 \
  --original_model vit_base_patch16_224 \
  --batch-size 24 \
  --epochs "${EPOCHS}" \
  --data-path "${DATA_PATH}" \
  --ca_lr 0.0045 \
  --crct_epochs "${CRCT_EPOCHS}" \
  --seed "${SEED}" \
  --lr 0.03 \
  --con 0.2 \
  --lora_rank 5 \
  --En gen \
  --tau -10 \
  --K 5 \
  --sched cosine \
  --dataset Split-CIFAR100 \
  --lora_momentum 0.4 \
  --lora_type hide \
  --trained_original_model "${TII_DIR}" \
  --num_tasks "${NUM_TASKS}" \
  --cfs_sampling \
  --cfs_epochs "${CFS_EPOCHS}" \
  --cfs_train_max_samples 1024 \
  --cfs_candidate_multiplier 2 \
  --cfs_boundary_replay \
  --cfs_boundary_ratio 0.25 \
  --cfs_boundary_multiplier 3 \
  --cfs_boundary_density_quantile 0.9 \
  --cfs_core_replay_ratio 0.25 \
  --cfs_core_multiplier 4 \
  --crct_use_all_samples \
  --ctird_task_selection task_energy \
  --ctird_task_temperature 0.1 \
  --ctird_task_weighting energy \
  --ctird_weight_temperature 1.0 \
  --ctird_weight_floor 0.2 \
  --crct_head_only \
  --crct_reliability_weighting \
  --crct_reliability_floor 0.5 \
  --crct_reliability_power 1.0 \
  --crct_reliability_preserve_class_mass \
  --crct_old_row_lr_scale 1.0 \
  --crct_adaptive_trust_region \
  --crct_trust_steps 10 \
  --crct_trust_samples_per_component 4 \
  --crct_trust_cov_scale 0.25 \
  --crct_trust_quantile 0.9 \
  --crct_trust_max_kl 0.02 \
  --crct_trust_max_conf_drop 0.02 \
  --crct_trust_max_margin_drop 0.10 \
  --output_dir "${OUTPUT_DIR}" \
  2>&1 | tee "${LOG_PATH}"

echo "Final metrics:"
grep "Average accuracy till task${NUM_TASKS}" "${LOG_PATH}" | tail -n 1 || true
