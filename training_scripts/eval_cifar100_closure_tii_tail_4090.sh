#!/usr/bin/env bash
set -euo pipefail

# closure + TII tail rematching audit on Split-CIFAR100 (locked i2/c5, prior 0.3,
# temperature 1.0, top-1-safe tail). Same method as the ImageNet-R result.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
DATA_PATH="${DATA_PATH:-${DATASETS_ROOT}}"
RUN_DIR="${1:-}"

if [[ -z "${RUN_DIR}" ]]; then echo "Usage: $0 RUN_DIR" >&2; exit 64; fi
if [[ ! -x "${PYTHON_BIN}" ]]; then echo "Python not found" >&2; exit 1; fi

RUN_BASENAME="$(basename "${RUN_DIR}")"
SEED="${SEED:-}"
if [[ -z "${SEED}" && "${RUN_BASENAME}" =~ seed([0-9]+)$ ]]; then SEED="${BASH_REMATCH[1]}"; fi
SEED="${SEED:-42}"
TII_DIR="${TII_DIR:-${OUTPUT_ROOT}/cifar100_tii_original_10tasks_seed${SEED}}"
TRAIN_LOG="${OUTPUT_ROOT}/${RUN_BASENAME}.log"
CONVENTIONAL_LOG="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_conventional.log"
LOG_PATH="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_prediction_closure_tii_tail_i2_c5_strict.log"

for task_id in $(seq 1 10); do
  [[ -s "${RUN_DIR}/checkpoint/task${task_id}_checkpoint.pth" ]] || { echo "Missing LoRA checkpoint task${task_id}" >&2; exit 2; }
  [[ -s "${TII_DIR}/checkpoint/task${task_id}_checkpoint.pth" ]] || { echo "Missing TII checkpoint task${task_id}" >&2; exit 2; }
done
[[ -s "${TRAIN_LOG}" ]] || { echo "Missing training log: ${TRAIN_LOG}" >&2; exit 2; }
if [[ -s "${LOG_PATH}" ]]; then echo "Refusing to overwrite: ${LOG_PATH}" >&2; exit 3; fi

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -c '
import sys, torch
from protocols import validate_exemplar_free_training_log
validate_exemplar_free_training_log(sys.argv[1])
checkpoint = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
if checkpoint.get("real_feature_memory"):
    raise SystemExit("STRICT_CHECKPOINT_AUDIT=FAIL (real_feature_memory present)")
state = checkpoint.get("model", checkpoint)
matches = [v for k, v in state.items() if k.endswith("lora_layer.k_lora_A")]
rank = int(matches[0].shape[-1]) if len(matches) == 1 else -1
if rank != 8:
    raise SystemExit(f"STRICT_CHECKPOINT_AUDIT=FAIL (expected LoRA rank 8, got {rank})")
print("STRICT_CHECKPOINT_AUDIT=PASS")
' "${TRAIN_LOG}" "${RUN_DIR}/checkpoint/task10_checkpoint.pth"

echo "closure + TII tail on Split-CIFAR100; locked i2/c5, prior 0.3, temperature 1.0"
START_TIME="$(date +%s)"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=1 --master_port="${MASTER_PORT:-29544}" \
  main.py cifar100_lora \
  --model vit_base_patch16_224 --original_model vit_base_patch16_224 \
  --batch-size "${EVAL_BATCH_SIZE:-24}" --epochs 1 --data-path "${DATA_PATH}" \
  --seed "${SEED}" --lr 0.03 --con 0.2 --lora_rank 8 \
  --En gen --tau -10 --K 5 --sched cosine --dataset Split-CIFAR100 \
  --lora_momentum 0.4 --lora_type hide --trained_original_model "${TII_DIR}" \
  --num_tasks 10 --progressive_oracle_audit --progressive_prediction_closure_audit \
  --progressive_prediction_closure_tii_tail_audit \
  --prediction_proposal_initial_count 2 --prediction_proposal_top_classes 5 \
  --progressive_tii_prior_weight 0.3 --progressive_logit_temperature 1.0 \
  --strict_exemplar_free --eval --output_dir "${RUN_DIR}" 2>&1 | tee "${LOG_PATH}"

printf 'Closure-TII-tail audit wall time seconds: %s\n' "$(( $(date +%s) - START_TIME ))" | tee -a "${LOG_PATH}"
echo "Final closure-TII-tail metrics:"
grep "Average accuracy till task10" "${LOG_PATH}" | tail -n 1 || true

if [[ -s "${CONVENTIONAL_LOG}" ]]; then
  "${PYTHON_BIN}" - "${CONVENTIONAL_LOG}" "${LOG_PATH}" <<'PY' | tee -a "${LOG_PATH}"
import re, sys
def final_row(path):
    with open(path, 'r', encoding='utf-8') as h:
        rows = [l.strip() for l in h if 'Average accuracy till task10' in l]
    if not rows: raise SystemExit('CLOSURE_TII_TAIL_ALL_METRIC_GATE=FAIL (missing task-10 row)')
    return rows[-1]
def metric(row, name):
    m = re.search(rf'{re.escape(name)}:\s*(-?[0-9]+(?:\.[0-9]+)?)', row)
    if not m: raise SystemExit(f'CLOSURE_TII_TAIL_ALL_METRIC_GATE=FAIL (missing {name})')
    return float(m.group(1))
base, prop = final_row(sys.argv[1]), final_row(sys.argv[2])
checks = {}
print('Comparison with conventional HRM-PET baseline:')
for name in ('Acc@task', 'Acc@1', 'Acc@5', 'Backward'):
    d = metric(prop, name) - metric(base, name); checks[name] = d > 0.0
    print(f'  {name} delta={d:+.4f}: {"PASS" if checks[name] else "FAIL"}')
for name in ('Loss', 'Forgetting'):
    d = metric(prop, name) - metric(base, name); checks[name] = d < 0.0
    print(f'  {name} delta={d:+.4f}: {"PASS" if checks[name] else "FAIL"}')
checks['ClosureWinnerRecall'] = metric(prop, 'ClosureWinnerRecall') >= 99.5
checks['ClosureLoRA/sample'] = metric(prop, 'ClosureLoRA/sample') <= 7.0001
for name in ('ClosureWinnerRecall', 'ClosureLoRA/sample'):
    print(f'  {name}: {"PASS" if checks[name] else "FAIL"}')
print('CLOSURE_TII_TAIL_ALL_METRIC_GATE=' + ('PASS' if all(checks.values()) else 'FAIL'))
PY
fi
