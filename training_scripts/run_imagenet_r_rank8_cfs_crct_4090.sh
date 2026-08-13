#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-pilot}"
if [[ "${MODE}" != "pilot" && "${MODE}" != "full" ]]; then
  echo "Usage: $0 [pilot|full]" >&2
  exit 64
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
SEED="${SEED:-42}"
GPUS="${GPUS:-1}"

if [[ -n "${DATA_PATH:-}" ]]; then
  IMR_DATA_PATH="${DATA_PATH}"
elif [[ -d "${DATASETS_ROOT}/imagenet-r/imagenet-r" || -f "${DATASETS_ROOT}/imagenet-r/imagenet-r.tar" ]]; then
  IMR_DATA_PATH="${DATASETS_ROOT}/imagenet-r"
else
  IMR_DATA_PATH="${DATASETS_ROOT}"
fi

TII_DIR="${TII_DIR:-${OUTPUT_ROOT}/imr_tii_original_10tasks_seed${SEED}}"
BASELINE_LOG="${BASELINE_LOG:-${OUTPUT_ROOT}/imr_lora_rank8_baseline_10tasks_seed${SEED}.log}"
LORA_EPOCHS="${LORA_EPOCHS:-50}"
CRCT_EPOCHS="${CRCT_EPOCHS:-30}"
MAX_TRAIN_TASKS=10
RUN_NAME="${RUN_NAME_OVERRIDE:-imr_rank8_cfs_oldonly_vgcrct_10tasks_seed${SEED}}"

if [[ "${MODE}" == "pilot" ]]; then
  MAX_TRAIN_TASKS="${PILOT_TASKS:-3}"
  RUN_NAME="${RUN_NAME_OVERRIDE:-imr_rank8_cfs_oldonly_vgcrct_pilot${MAX_TRAIN_TASKS}_seed${SEED}}"
fi

OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_PATH="${OUTPUT_ROOT}/${RUN_NAME}.log"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -d "${IMR_DATA_PATH}/imagenet-r" && ! -f "${IMR_DATA_PATH}/imagenet-r.tar" ]]; then
  echo "ImageNet-R was not found under: ${IMR_DATA_PATH}" >&2
  exit 2
fi
for task_id in $(seq 1 10); do
  checkpoint="${TII_DIR}/checkpoint/task${task_id}_checkpoint.pth"
  if [[ ! -s "${checkpoint}" ]]; then
    echo "Missing TII checkpoint: ${checkpoint}" >&2
    exit 3
  fi
done
if [[ -d "${OUTPUT_DIR}/checkpoint" ]] && compgen -G "${OUTPUT_DIR}/checkpoint/task*_checkpoint.pth" > /dev/null; then
  echo "Refusing to overwrite checkpoints in: ${OUTPUT_DIR}" >&2
  echo "Set RUN_NAME_OVERRIDE to a new name." >&2
  exit 4
fi
if [[ -s "${LOG_PATH}" ]]; then
  echo "Refusing to overwrite existing log: ${LOG_PATH}" >&2
  echo "Set RUN_NAME_OVERRIDE to a new name." >&2
  exit 4
fi

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

echo "Mode: ${MODE}"
echo "Dataset: ${IMR_DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Method: strict exemplar-free rank-8 LoRA + CFS-only validation-gated CRCT"
echo "Historical images/per-example real features: forbidden by runtime guard"
echo "Replay source: Gaussian statistics + CFS synthetic features only"
echo "CFS scope: old classes only; current-task classes use Gaussian replay"
echo "CRCT classifier interpolation cap: ${CRCT_VALIDATION_MAX_ALPHA:-1.0}"
echo "Semantic/prototype/exhaustive: disabled"

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT:-29567}" \
  main.py \
  imr_lora \
  --model vit_base_patch16_224 \
  --original_model vit_base_patch16_224 \
  --batch-size "${LORA_BATCH_SIZE:-24}" \
  --epochs "${LORA_EPOCHS}" \
  --data-path "${IMR_DATA_PATH}" \
  --ca_lr "${CA_LR:-0.005}" \
  --crct_epochs "${CRCT_EPOCHS}" \
  --seed "${SEED}" \
  --lr 0.03 \
  --con 0.2 \
  --lora_rank 8 \
  --En gen \
  --tau -10 \
  --K 5 \
  --sched cosine \
  --dataset Split-Imagenet-R \
  --lora_momentum 0.4 \
  --lora_type hide \
  --trained_original_model "${TII_DIR}" \
  --num_tasks 10 \
  --max_train_tasks "${MAX_TRAIN_TASKS}" \
  --strict_exemplar_free \
  --ca_storage_efficient_method variance \
  --cfs_sampling \
  --cfs_old_classes_only \
  --cfs_epochs "${CFS_EPOCHS:-50}" \
  --cfs_lr "${CFS_LR:-0.01}" \
  --cfs_train_max_samples "${CFS_TRAIN_MAX_SAMPLES:-1024}" \
  --cfs_candidate_multiplier "${CFS_CANDIDATE_MULTIPLIER:-3}" \
  --cfs_init_strategy mean \
  --cfs_paper_style \
  --cfs_selection_ratio "${CFS_SELECTION_RATIO:-0.5}" \
  --cfs_selection_steps "${CFS_SELECTION_STEPS:-5}" \
  --cfs_distribution_filter \
  --cfs_filter_multiplier "${CFS_FILTER_MULTIPLIER:-3}" \
  --cfs_boundary_replay \
  --cfs_boundary_ratio "${CFS_BOUNDARY_RATIO:-0.10}" \
  --cfs_boundary_multiplier "${CFS_BOUNDARY_MULTIPLIER:-3}" \
  --cfs_boundary_density_quantile "${CFS_BOUNDARY_DENSITY_QUANTILE:-0.85}" \
  --cfs_boundary_target_side \
  --cfs_core_replay_ratio "${CFS_CORE_REPLAY_RATIO:-0.40}" \
  --cfs_core_multiplier "${CFS_CORE_MULTIPLIER:-4}" \
  --crct_use_all_samples \
  --crct_balanced_batches \
  --crct_head_only \
  --crct_validation_gate \
  --crct_validation_steps "${CRCT_VALIDATION_STEPS:-20}" \
  --crct_validation_max_alpha "${CRCT_VALIDATION_MAX_ALPHA:-1.0}" \
  --crct_validation_samples_per_component "${CRCT_VALIDATION_SAMPLES:-16}" \
  --crct_validation_cov_scale "${CRCT_VALIDATION_COV_SCALE:-0.20}" \
  --crct_validation_repeats "${CRCT_VALIDATION_REPEATS:-3}" \
  --crct_validation_max_old_acc_drop "${CRCT_VALIDATION_MAX_OLD_DROP:-0.0}" \
  --output_dir "${OUTPUT_DIR}" \
  2>&1 | tee "${LOG_PATH}"

for task_id in $(seq 1 "${MAX_TRAIN_TASKS}"); do
  checkpoint="${OUTPUT_DIR}/checkpoint/task${task_id}_checkpoint.pth"
  if [[ ! -s "${checkpoint}" ]]; then
    echo "Missing checkpoint after training: ${checkpoint}" >&2
    exit 5
  fi
  "${PYTHON_BIN}" tools/audit_exemplar_free_checkpoint.py "${checkpoint}"
done

FINAL_ROW=$(grep "Average accuracy till task${MAX_TRAIN_TASKS}" "${LOG_PATH}" | tail -n 1 || true)
echo "Final metrics:"
echo "${FINAL_ROW}"

if [[ "${MODE}" == "pilot" ]]; then
  if [[ ! -s "${BASELINE_LOG}" ]]; then
    echo "Pilot completed, but baseline log was not found: ${BASELINE_LOG}" >&2
    exit 6
  fi
  BASELINE_ROW=$(grep "Average accuracy till task${MAX_TRAIN_TASKS}" "${BASELINE_LOG}" | tail -n 1 || true)
  if [[ -z "${BASELINE_ROW}" || -z "${FINAL_ROW}" ]]; then
    echo "Could not find comparable task-${MAX_TRAIN_TASKS} rows." >&2
    exit 6
  fi
  export BASELINE_ROW FINAL_ROW
  "${PYTHON_BIN}" - <<'PY'
import os
import re
import sys

def metric(row, name):
    match = re.search(rf"{re.escape(name)}:\s*(-?[0-9.]+)", row)
    if not match:
        raise RuntimeError(f"Missing {name} in: {row}")
    return float(match.group(1))

baseline = os.environ['BASELINE_ROW']
candidate = os.environ['FINAL_ROW']
higher_is_better = ['Acc@task', 'Acc@1', 'Acc@5', 'Backward']
lower_is_better = ['Loss', 'Forgetting']
deltas = {
    name: metric(candidate, name) - metric(baseline, name)
    for name in higher_is_better + lower_is_better
}
checks = {
    **{name: deltas[name] >= 0.0 for name in higher_is_better},
    **{name: deltas[name] <= 0.0 for name in lower_is_better},
}
passed = all(checks.values())
print('Pilot baseline :', baseline)
print('Pilot candidate:', candidate)
print('Strict multi-metric pilot gate:')
for name in higher_is_better + lower_is_better:
    direction = 'higher' if name in higher_is_better else 'lower'
    status = 'PASS' if checks[name] else 'FAIL'
    print(f'  {name:10s} delta={deltas[name]:+.4f} '
          f'({direction} is better): {status}')
print('PILOT_GATE=' + ('PASS' if passed else 'FAIL'))
sys.exit(0 if passed else 10)
PY
fi
