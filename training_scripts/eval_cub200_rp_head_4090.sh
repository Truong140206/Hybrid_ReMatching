#!/usr/bin/env bash
set -euo pipefail

# Routing-free RanPAC-style head on Split-CUB200.
#
# HRM-PET routes to one task LoRA before classifying, so CUB Acc@1 is stuck at
# ~86.5 (exhaustive re-matching, Gaussian re-scoring and rank 8->16 all failed
# to beat it). Published exemplar-free RanPAC reaches ~90.3 on CUB B0I10 with
# NO routing: frozen random projection + nonlinearity, then ridge over an
# accumulated Gram matrix and class prototypes. This evaluates that head on our
# own checkpoints. Only aggregate statistics are kept -- no exemplars.
#
# Sweep knobs (env): RP_DIM=5000  RP_ACT=relu|square|gelu  RP_LAMBDA=10000
#                    RP_SOURCE=original|lora

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
DATA_PATH="${DATA_PATH:-${DATASETS_ROOT}}"
RUN_DIR="${1:-}"
RP_DIM="${RP_DIM:-5000}"
RP_ACT="${RP_ACT:-relu}"
RP_LAMBDA="${RP_LAMBDA:-10000}"
RP_SOURCE="${RP_SOURCE:-original}"
RP_NORM="${RP_NORM:-none}"

if [[ -z "${RUN_DIR}" ]]; then echo "Usage: $0 RUN_DIR" >&2; exit 64; fi
if [[ ! -x "${PYTHON_BIN}" ]]; then echo "Python not found" >&2; exit 1; fi
RUN_BASENAME="$(basename "${RUN_DIR}")"
SEED="${SEED:-}"; if [[ -z "${SEED}" && "${RUN_BASENAME}" =~ seed([0-9]+)$ ]]; then SEED="${BASH_REMATCH[1]}"; fi; SEED="${SEED:-42}"
TII_DIR="${TII_DIR:-${OUTPUT_ROOT}/cub200_tii_original_10tasks_seed${SEED}}"
CONVENTIONAL_LOG="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_conventional.log"

for task_id in $(seq 1 10); do
  [[ -s "${RUN_DIR}/checkpoint/task${task_id}_checkpoint.pth" ]] || { echo "Missing LoRA checkpoint task${task_id}" >&2; exit 2; }
  [[ -s "${TII_DIR}/checkpoint/task${task_id}_checkpoint.pth" ]] || { echo "Missing TII checkpoint task${task_id}" >&2; exit 2; }
done

CHECKPOINT_RANK="$("${PYTHON_BIN}" -c '
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
s = c.get("model", c); m = [v for k, v in s.items() if k.endswith("lora_layer.k_lora_A")]
if len(m) != 1: raise SystemExit(f"Expected one k_lora_A, found {len(m)}")
print(int(m[0].shape[-1]))
' "${RUN_DIR}/checkpoint/task1_checkpoint.pth")"

tag() { printf '%s' "$1" | tr '.' 'p'; }
LOG_PATH="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_rp_${RP_SOURCE}_d${RP_DIM}_${RP_ACT}_l$(tag "${RP_LAMBDA}")_n${RP_NORM}.log"
[[ -s "${LOG_PATH}" ]] && { echo "Refusing to overwrite: ${LOG_PATH}" >&2; exit 3; } || true

cd "${REPO_ROOT}"
echo "Routing-free RP head on Split-CUB200; dim=${RP_DIM} act=${RP_ACT} lambda=${RP_LAMBDA} source=${RP_SOURCE} rank=${CHECKPOINT_RANK}"
START_TIME="$(date +%s)"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=1 --master_port="${MASTER_PORT:-29557}" \
  main.py imr_lora \
  --model vit_base_patch16_224 --original_model vit_base_patch16_224 \
  --batch-size "${EVAL_BATCH_SIZE:-24}" --epochs 1 --data-path "${DATA_PATH}" \
  --seed "${SEED}" --lr 0.03 --con 0.2 --lora_rank "${CHECKPOINT_RANK}" \
  --En gen --tau -10 --K 5 --sched cosine --dataset Split-CUB200 \
  --lora_momentum 0.4 --lora_type hide --trained_original_model "${TII_DIR}" \
  --num_tasks 10 --rp_head --rp_dim "${RP_DIM}" --rp_activation "${RP_ACT}" \
  --rp_lambda "${RP_LAMBDA}" --rp_feature_source "${RP_SOURCE}" \
  --rp_normalize "${RP_NORM}" \
  --strict_exemplar_free --eval --output_dir "${RUN_DIR}" 2>&1 | tee "${LOG_PATH}"
printf 'RP head wall time seconds: %s\n' "$(( $(date +%s) - START_TIME ))" | tee -a "${LOG_PATH}"
echo "Final RP head metrics:"; grep "Average accuracy till task10" "${LOG_PATH}" | tail -n 1 || true

if [[ -s "${CONVENTIONAL_LOG}" ]]; then
  "${PYTHON_BIN}" - "${CONVENTIONAL_LOG}" "${LOG_PATH}" <<'PY' | tee -a "${LOG_PATH}"
import re, sys
def final_row(p):
    with open(p, 'r', encoding='utf-8') as h:
        r = [l.strip() for l in h if 'Average accuracy till task10' in l]
    if not r: raise SystemExit('RP_HEAD_GATE=FAIL (missing task-10 row)')
    return r[-1]
def metric(row, name):
    mm = re.search(rf'{re.escape(name)}:\s*(-?[0-9]+(?:\.[0-9]+)?)', row)
    if not mm: raise SystemExit(f'RP_HEAD_GATE=FAIL (missing {name})')
    return float(mm.group(1))
base, prop = final_row(sys.argv[1]), final_row(sys.argv[2])
print('Comparison with conventional HRM-PET baseline:')
gate = False
for name in ('Acc@1', 'Acc@5'):
    d = metric(prop, name) - metric(base, name)
    ok = d > 0.0
    if name == 'Acc@1': gate = ok
    print(f'  {name} delta={d:+.4f}: {"PASS" if ok else "FAIL"}')
d = metric(prop, 'Loss') - metric(base, 'Loss')
print(f'  Loss delta={d:+.4f}: {"PASS" if d < 0.0 else "FAIL"}')
print(f'  RanPAC published CUB reference: ~90.3 Acc@1 (ours: {metric(prop, "Acc@1"):.2f})')
print('RP_HEAD_GATE=' + ('PASS' if gate else 'FAIL'))
PY
fi
