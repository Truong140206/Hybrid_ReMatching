#!/usr/bin/env bash
set -euo pipefail

# Hybrid evaluation on any dataset: HRM-PET's own LoRA supplies first-session
# adaptation, and a routing-free RanPAC-style second-order head classifies.
#
# On Split-CUB200 this beat conventional HRM-PET where every routing-based
# variant had failed: Acc@1 87.50 vs 86.53, Acc@task 94.02 vs 93.21, better
# Forgetting/Backward, at 1.0 LoRA/sample instead of 1.60.
#
# Usage:  DATASET=Split-CUB200 TII_DIR=... $0 RUN_DIR
# Knobs:  RP_DIM RP_LAMBDA RP_ACT RP_NORM RP_SOURCE RP_LORA_TASK CALIBRATE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
DATA_PATH="${DATA_PATH:-${DATASETS_ROOT}}"
RUN_DIR="${1:-}"
DATASET="${DATASET:-}"
RP_DIM="${RP_DIM:-10000}"
RP_ACT="${RP_ACT:-relu}"
RP_LAMBDA="${RP_LAMBDA:-10000}"
RP_SOURCE="${RP_SOURCE:-lora}"
RP_NORM="${RP_NORM:-none}"
RP_LORA_TASK="${RP_LORA_TASK:-0}"
CALIBRATE="${CALIBRATE:-1}"
NUM_TASKS="${NUM_TASKS:-10}"

if [[ -z "${RUN_DIR}" ]]; then echo "Usage: DATASET=... TII_DIR=... $0 RUN_DIR" >&2; exit 64; fi
if [[ -z "${DATASET}" ]]; then echo "Set DATASET (e.g. Split-CUB200)" >&2; exit 64; fi
if [[ -z "${TII_DIR:-}" ]]; then echo "Set TII_DIR" >&2; exit 64; fi
if [[ ! -x "${PYTHON_BIN}" ]]; then echo "Python not found" >&2; exit 1; fi
RUN_BASENAME="$(basename "${RUN_DIR}")"
SEED="${SEED:-}"; if [[ -z "${SEED}" && "${RUN_BASENAME}" =~ seed([0-9]+)$ ]]; then SEED="${BASH_REMATCH[1]}"; fi; SEED="${SEED:-42}"
CONVENTIONAL_LOG="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_conventional.log"

for task_id in $(seq 1 "${NUM_TASKS}"); do
  [[ -s "${RUN_DIR}/checkpoint/task${task_id}_checkpoint.pth" ]] || { echo "Missing LoRA checkpoint task${task_id}" >&2; exit 2; }
  [[ -s "${TII_DIR}/checkpoint/task${task_id}_checkpoint.pth" ]] || { echo "Missing TII checkpoint task${task_id}" >&2; exit 2; }
done

CHECKPOINT_RANK="$("${PYTHON_BIN}" -c '
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if c.get("real_feature_memory"): raise SystemExit("STRICT_CHECKPOINT_AUDIT=FAIL (real_feature_memory present)")
s = c.get("model", c); m = [v for k, v in s.items() if k.endswith("lora_layer.k_lora_A")]
if len(m) != 1: raise SystemExit(f"Expected one k_lora_A, found {len(m)}")
print(int(m[0].shape[-1]))
' "${RUN_DIR}/checkpoint/task1_checkpoint.pth")"

tag() { printf '%s' "$1" | tr '.' 'p'; }
CAL_FLAG=""; [[ "${CALIBRATE}" == "1" ]] && CAL_FLAG="--rp_calibrate"
LOG_PATH="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_rp_${RP_SOURCE}_d${RP_DIM}_${RP_ACT}_l$(tag "${RP_LAMBDA}")_n${RP_NORM}_t${RP_LORA_TASK}_c${CALIBRATE}.log"
[[ -s "${LOG_PATH}" ]] && { echo "Refusing to overwrite: ${LOG_PATH}" >&2; exit 3; } || true

cd "${REPO_ROOT}"
echo "Hybrid RP head on ${DATASET}; dim=${RP_DIM} lambda=${RP_LAMBDA} source=${RP_SOURCE} adapter=${RP_LORA_TASK} rank=${CHECKPOINT_RANK} calibrate=${CALIBRATE}"
START_TIME="$(date +%s)"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=1 --master_port="${MASTER_PORT:-29558}" \
  main.py imr_lora \
  --model vit_base_patch16_224 --original_model vit_base_patch16_224 \
  --batch-size "${EVAL_BATCH_SIZE:-24}" --epochs 1 --data-path "${DATA_PATH}" \
  --seed "${SEED}" --lr 0.03 --con 0.2 --lora_rank "${CHECKPOINT_RANK}" \
  --En gen --tau -10 --K 5 --sched cosine --dataset "${DATASET}" \
  --lora_momentum 0.4 --lora_type hide --trained_original_model "${TII_DIR}" \
  --num_tasks "${NUM_TASKS}" --rp_head --rp_dim "${RP_DIM}" \
  --rp_activation "${RP_ACT}" --rp_lambda "${RP_LAMBDA}" \
  --rp_feature_source "${RP_SOURCE}" --rp_normalize "${RP_NORM}" \
  --rp_lora_task "${RP_LORA_TASK}" ${CAL_FLAG} \
  --strict_exemplar_free --eval --output_dir "${RUN_DIR}" 2>&1 | tee "${LOG_PATH}"
printf 'Hybrid RP head wall time seconds: %s\n' "$(( $(date +%s) - START_TIME ))" | tee -a "${LOG_PATH}"
echo "Final hybrid metrics:"; grep "Average accuracy till task${NUM_TASKS}" "${LOG_PATH}" | tail -n 1 || true

if [[ -s "${CONVENTIONAL_LOG}" ]]; then
  NUM_TASKS="${NUM_TASKS}" "${PYTHON_BIN}" - "${CONVENTIONAL_LOG}" "${LOG_PATH}" <<'PY' | tee -a "${LOG_PATH}"
import os, re, sys
tasks = os.environ.get('NUM_TASKS', '10')
def final_row(p):
    with open(p, 'r', encoding='utf-8') as h:
        r = [l.strip() for l in h if f'Average accuracy till task{tasks}' in l]
    if not r: raise SystemExit('HYBRID_GATE=FAIL (missing final row)')
    return r[-1]
def metric(row, name):
    mm = re.search(rf'{re.escape(name)}:\s*(-?[0-9]+(?:\.[0-9]+)?)', row)
    return None if not mm else float(mm.group(1))
base, prop = final_row(sys.argv[1]), final_row(sys.argv[2])
checks = {}
print('Comparison with conventional HRM-PET baseline:')
for name in ('Acc@task', 'Acc@1', 'Acc@5', 'Backward'):
    a, b = metric(prop, name), metric(base, name)
    if a is None or b is None: continue
    checks[name] = (a - b) > 0.0
    print(f'  {name} delta={a-b:+.4f}: {"PASS" if checks[name] else "FAIL"}')
for name in ('Loss', 'Forgetting'):
    a, b = metric(prop, name), metric(base, name)
    if a is None or b is None: continue
    checks[name] = (a - b) < 0.0
    print(f'  {name} delta={a-b:+.4f}: {"PASS" if checks[name] else "FAIL"}')
a, b = metric(prop, 'LoRA/sample'), metric(base, 'LoRA/sample')
if a is not None and b is not None:
    print(f'  LoRA/sample {a:.4f} vs {b:.4f} (lower is cheaper)')
print('HYBRID_GATE=' + ('PASS' if all(checks.values()) else 'FAIL'))
PY
fi
