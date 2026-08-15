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
MAX_TRAIN_TASKS=10
RUN_NAME="${RUN_NAME_OVERRIDE:-imr_rank8_cfs_logit_calibration_10tasks_seed${SEED}}"
if [[ "${MODE}" == "pilot" ]]; then
  MAX_TRAIN_TASKS="${PILOT_TASKS:-3}"
  RUN_NAME="${RUN_NAME_OVERRIDE:-imr_rank8_cfs_logit_calibration_pilot${MAX_TRAIN_TASKS}_seed${SEED}}"
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

"${PYTHON_BIN}" -m pytest -q \
  tests/test_cfs_task_logit_calibration.py \
  tests/test_prediction_proposal_audit.py \
  tests/test_cfs_moment_matching.py \
  tests/test_exemplar_free_protocol.py

echo "Mode: ${MODE}"
echo "Dataset: ${IMR_DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Method: original rank-8 HRM-PET training + CFS task-logit calibration"
echo "CFS role: transient synthetic feature generation for task-wise scale/bias only"
echo "CRCT role: unchanged baseline Gaussian classifier correction"
echo "Stored calibration: scalar scale/bias and aggregate metrics only"
echo "Historical images/per-example real features: forbidden by strict audit"
echo "Evaluation: raw prediction proposal, TII top-2 + up to 4 proposals"
echo "Locked CFS epochs: ${CFS_EPOCHS:-50}; samples/class/split: ${CALIBRATION_SAMPLES:-48}"

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT:-29591}" \
  main.py \
  imr_lora \
  --model vit_base_patch16_224 \
  --original_model vit_base_patch16_224 \
  --batch-size "${LORA_BATCH_SIZE:-24}" \
  --epochs "${LORA_EPOCHS:-50}" \
  --data-path "${IMR_DATA_PATH}" \
  --ca_lr "${CA_LR:-0.005}" \
  --crct_epochs "${CRCT_EPOCHS:-30}" \
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
  --cfs_task_logit_calibration \
  --cfs_epochs "${CFS_EPOCHS:-50}" \
  --cfs_lr "${CFS_LR:-0.01}" \
  --cfs_batch_size "${CFS_BATCH_SIZE:-64}" \
  --cfs_train_max_samples "${CFS_TRAIN_MAX_SAMPLES:-1024}" \
  --cfs_candidate_multiplier "${CFS_CANDIDATE_MULTIPLIER:-3}" \
  --cfs_paper_style \
  --cfs_moment_match \
  --cfs_selection_ratio "${CFS_SELECTION_RATIO:-0.5}" \
  --cfs_selection_steps "${CFS_SELECTION_STEPS:-5}" \
  --cfs_logit_calibration_samples_per_class "${CALIBRATION_SAMPLES:-48}" \
  --cfs_logit_calibration_steps "${CALIBRATION_STEPS:-200}" \
  --cfs_logit_calibration_lr "${CALIBRATION_LR:-0.05}" \
  --cfs_logit_calibration_max_scale "${CALIBRATION_MAX_SCALE:-1.25}" \
  --cfs_logit_calibration_max_bias "${CALIBRATION_MAX_BIAS:-0.5}" \
  --cfs_logit_calibration_regularization "${CALIBRATION_REGULARIZATION:-0.05}" \
  --cfs_logit_calibration_old_margin_weight "${CALIBRATION_OLD_MARGIN_WEIGHT:-0.5}" \
  --cfs_logit_calibration_old_tolerance 0.0 \
  --cfs_logit_calibration_min_gain 0.0 \
  --cfs_logit_calibration_max_old_class_drop 0.0 \
  --prediction_proposal_rematching \
  --prediction_proposal_initial_count 2 \
  --prediction_proposal_count 4 \
  --prediction_proposal_top_classes 5 \
  --prediction_proposal_tii_completion \
  --progressive_tii_prior_weight 0.3 \
  --progressive_logit_temperature 1.0 \
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

UNCALIBRATED_ROW=""
if [[ "${MODE}" == "pilot" ]]; then
  UNCALIBRATED_LOG="${OUTPUT_ROOT}/${RUN_NAME}_uncalibrated_reference.log"
  if [[ -s "${UNCALIBRATED_LOG}" ]]; then
    echo "Refusing to overwrite reference log: ${UNCALIBRATED_LOG}" >&2
    exit 4
  fi
  echo "Evaluating the same checkpoint and proposal budget without CFS calibration"
  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port="${REFERENCE_MASTER_PORT:-29592}" \
    main.py \
    imr_lora \
    --model vit_base_patch16_224 \
    --original_model vit_base_patch16_224 \
    --batch-size "${LORA_BATCH_SIZE:-24}" \
    --epochs 1 \
    --data-path "${IMR_DATA_PATH}" \
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
    --prediction_proposal_rematching \
    --prediction_proposal_initial_count 2 \
    --prediction_proposal_count 4 \
    --prediction_proposal_top_classes 5 \
    --prediction_proposal_tii_completion \
    --progressive_tii_prior_weight 0.3 \
    --progressive_logit_temperature 1.0 \
    --eval \
    --output_dir "${OUTPUT_DIR}" \
    2>&1 | tee "${UNCALIBRATED_LOG}"
  UNCALIBRATED_ROW="$(grep "Average accuracy till task${MAX_TRAIN_TASKS}" \
    "${UNCALIBRATED_LOG}" | tail -n 1 || true)"
fi

FINAL_ROW="$(grep "Average accuracy till task${MAX_TRAIN_TASKS}" "${LOG_PATH}" | tail -n 1 || true)"
echo "Final CFS task-logit calibration metrics:"
echo "${FINAL_ROW}"
grep "CFS task-logit calibration:" "${LOG_PATH}" | tail -n "${MAX_TRAIN_TASKS}" || true

if [[ "${MODE}" == "pilot" ]]; then
  if [[ ! -s "${BASELINE_LOG}" ]]; then
    echo "Pilot completed, but baseline log was not found: ${BASELINE_LOG}" >&2
    exit 6
  fi
  BASELINE_ROW="$(grep "Average accuracy till task${MAX_TRAIN_TASKS}" "${BASELINE_LOG}" | tail -n 1 || true)"
  if [[ -z "${BASELINE_ROW}" || -z "${FINAL_ROW}" || -z "${UNCALIBRATED_ROW}" ]]; then
    echo "Could not find comparable task-${MAX_TRAIN_TASKS} rows." >&2
    exit 6
  fi
  export BASELINE_ROW FINAL_ROW UNCALIBRATED_ROW
  "${PYTHON_BIN}" - <<'PY'
import os
import re
import sys


def metric(row, name):
    match = re.search(rf'{re.escape(name)}:\s*(-?[0-9.]+)', row)
    if not match:
        raise RuntimeError(f'Missing {name} in: {row}')
    return float(match.group(1))


baseline = os.environ['BASELINE_ROW']
candidate = os.environ['FINAL_ROW']
reference = os.environ['UNCALIBRATED_ROW']
higher = ('Acc@task', 'Acc@1', 'Acc@5', 'Backward')
lower = ('Loss', 'Forgetting')
deltas = {
    name: metric(candidate, name) - metric(baseline, name)
    for name in higher + lower
}
checks = {
    **{name: deltas[name] >= 0.0 for name in higher},
    **{name: deltas[name] <= 0.0 for name in lower},
}
reference_deltas = {
    name: metric(candidate, name) - metric(reference, name)
    for name in higher + lower
}
reference_checks = {
    **{name: reference_deltas[name] >= 0.0 for name in higher},
    **{name: reference_deltas[name] <= 0.0 for name in lower},
}
strict_gain = any(
    reference_deltas[name] > 1e-6 for name in higher
) or any(
    reference_deltas[name] < -1e-6 for name in lower
)
print('Pilot baseline             :', baseline)
print('Same-checkpoint uncalibrated:', reference)
print('Pilot candidate            :', candidate)
print('Strict six-metric baseline gate:')
for name in higher + lower:
    direction = 'higher' if name in higher else 'lower'
    print(f'  {name:10s} delta={deltas[name]:+.4f} '
          f'({direction} is better): '
          f'{"PASS" if checks[name] else "FAIL"}')
print('CFS causal comparison against the same uncalibrated checkpoint:')
for name in higher + lower:
    direction = 'higher' if name in higher else 'lower'
    print(f'  {name:10s} delta={reference_deltas[name]:+.4f} '
          f'({direction} is better): '
          f'{"PASS" if reference_checks[name] else "FAIL"}')
causal_pass = all(reference_checks.values()) and strict_gain
print('CFS_STRICT_GAIN=' + ('PASS' if strict_gain else 'FAIL'))
print('CFS_CAUSAL_GATE=' + ('PASS' if causal_pass else 'FAIL'))
passed = all(checks.values()) and causal_pass
print('PILOT_GATE=' + ('PASS' if passed else 'FAIL'))
sys.exit(0 if passed else 10)
PY
fi
