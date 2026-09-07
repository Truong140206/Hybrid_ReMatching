#!/usr/bin/env bash
set -euo pipefail

# Conventional HRM-PET (TII + DRM + CRM) evaluation on any dataset and backbone.
#
# eval_imagenet_r_conventional_4090.sh pins imr_lora, vit_base_patch16_224,
# Split-Imagenet-R and ten tasks, so the baseline log exists for one backbone on
# two benchmarks. That is enough to state that the hybrid both learns each task
# better and retains more of it on those two cells, and not enough to state it
# anywhere else -- which is the weakest point in the current evidence, since the
# twenty-one-cell comparison is between two variants of our own method.
#
# The log name is unchanged, ${RUN_BASENAME}_eval_conventional.log, because
# eval_rp_head_any_4090.sh and tools/learning_accuracy.py both address it.
#
# Usage: DATASET=Split-CIFAR100 CONFIG=cifar100_lora TII_DIR=... $0 RUN_DIR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
DATA_PATH="${DATA_PATH:-${DATASETS_ROOT}}"
RUN_DIR="${1:-}"
DATASET="${DATASET:-}"
CONFIG="${CONFIG:-imr_lora}"
NUM_TASKS="${NUM_TASKS:-10}"
BACKBONE="${BACKBONE:-vit_base_patch16_224}"
LORA_RANK="${LORA_RANK:-}"

if [[ -z "${RUN_DIR}" ]]; then echo "Usage: DATASET=... TII_DIR=... $0 RUN_DIR" >&2; exit 64; fi
if [[ -z "${DATASET}" ]]; then echo "Set DATASET (e.g. Split-CIFAR100)" >&2; exit 64; fi
if [[ -z "${TII_DIR:-}" ]]; then echo "Set TII_DIR" >&2; exit 64; fi
if [[ ! -x "${PYTHON_BIN}" ]]; then echo "Python not found: ${PYTHON_BIN}" >&2; exit 1; fi

RUN_BASENAME="$(basename "${RUN_DIR}")"
SEED="${SEED:-}"
if [[ -z "${SEED}" && "${RUN_BASENAME}" =~ seed([0-9]+)$ ]]; then SEED="${BASH_REMATCH[1]}"; fi
SEED="${SEED:-42}"
TRAIN_LOG="${TRAIN_LOG:-${OUTPUT_ROOT}/${RUN_BASENAME}.log}"
LOG_PATH="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_conventional.log"

for task_id in $(seq 1 "${NUM_TASKS}"); do
  [[ -s "${RUN_DIR}/checkpoint/task${task_id}_checkpoint.pth" ]] || {
    echo "Missing LoRA checkpoint task${task_id}" >&2; exit 2; }
  [[ -s "${TII_DIR}/checkpoint/task${task_id}_checkpoint.pth" ]] || {
    echo "Missing TII checkpoint task${task_id}" >&2; exit 2; }
done
if [[ ! -s "${TRAIN_LOG}" ]]; then
  echo "Training log required for the protocol audit: ${TRAIN_LOG}" >&2; exit 2
fi
if [[ -s "${LOG_PATH}" ]]; then
  echo "Refusing to overwrite completed evaluation log: ${LOG_PATH}" >&2; exit 3
fi

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -c '
import sys
from protocols import validate_exemplar_free_training_log
validate_exemplar_free_training_log(sys.argv[1])
' "${TRAIN_LOG}"

CHECKPOINT_RANK="$("${PYTHON_BIN}" -c '
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if c.get("real_feature_memory"): raise SystemExit("STRICT_CHECKPOINT_AUDIT=FAIL (real_feature_memory present)")
s = c.get("model", c); m = [v for k, v in s.items() if k.endswith("lora_layer.k_lora_A")]
if len(m) != 1: raise SystemExit(f"Expected one k_lora_A, found {len(m)}")
print(int(m[0].shape[-1]))
' "${RUN_DIR}/checkpoint/task1_checkpoint.pth")"
if [[ -z "${LORA_RANK}" ]]; then
  LORA_RANK="${CHECKPOINT_RANK}"
elif [[ "${LORA_RANK}" != "${CHECKPOINT_RANK}" ]]; then
  echo "Requested LoRA rank ${LORA_RANK}, but checkpoint rank is ${CHECKPOINT_RANK}" >&2; exit 2
fi

echo "Conventional HRM-PET on ${DATASET}; backbone=${BACKBONE}; seed=${SEED}; rank=${LORA_RANK}"
START_TIME="$(date +%s)"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=1 --master_port="${MASTER_PORT:-29591}" \
  main.py "${CONFIG}" \
  --model "${BACKBONE}" --original_model "${BACKBONE}" \
  --batch-size "${EVAL_BATCH_SIZE:-24}" --epochs 1 --data-path "${DATA_PATH}" \
  --seed "${SEED}" --lr 0.03 --con 0.2 --lora_rank "${LORA_RANK}" \
  --En gen --tau -10 --K 5 --sched cosine --dataset "${DATASET}" \
  --lora_momentum 0.4 --lora_type hide --trained_original_model "${TII_DIR}" \
  --num_tasks "${NUM_TASKS}" \
  --strict_exemplar_free --eval --output_dir "${RUN_DIR}" 2>&1 | tee "${LOG_PATH}"

printf 'Conventional evaluation wall time seconds: %s\n' "$(( $(date +%s) - START_TIME ))" | tee -a "${LOG_PATH}"
FINAL_LINE="$(grep "Average accuracy till task${NUM_TASKS}" "${LOG_PATH}" | tail -n 1 || true)"
REFERENCE_LINE="$(grep "Average accuracy till task${NUM_TASKS}" "${TRAIN_LOG}" | tail -n 1 || true)"
echo "Final conventional metrics:"
echo "${FINAL_LINE}"

if [[ -z "${REFERENCE_LINE}" ]]; then
  echo "CONVENTIONAL_REPRODUCTION_GATE=SKIP (training log has no final row)"
  exit 0
fi
"${PYTHON_BIN}" - "${FINAL_LINE}" "${REFERENCE_LINE}" <<'PY'
import re
import sys

candidate, reference = sys.argv[1:]
metrics = ('Acc@task', 'Acc@1', 'Acc@5', 'Loss', 'Forgetting', 'Backward')
limits = {'Loss': 0.001}


def metric(row, name):
    match = re.search(re.escape(name) + r':\s*([-+0-9.]+)', row)
    if match is None:
        raise SystemExit(f'Missing {name}: {row}')
    return float(match.group(1))


passed = True
for name in metrics:
    delta = metric(candidate, name) - metric(reference, name)
    tolerance = limits.get(name, 0.01)
    ok = abs(delta) <= tolerance
    passed = passed and ok
    print(f'{name} delta={delta:+.6f}: {"PASS" if ok else "FAIL"}')
print('CONVENTIONAL_REPRODUCTION_GATE=' + ('PASS' if passed else 'FAIL'))
PY
