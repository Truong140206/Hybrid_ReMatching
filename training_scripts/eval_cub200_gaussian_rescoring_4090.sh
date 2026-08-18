#!/usr/bin/env bash
set -euo pipefail

# Gaussian (Mahalanobis) re-scoring on Split-CUB200: class-incremental
# prediction by per-class Gaussian fit over stored cls_mean/cls_cov instead of
# energy re-matching. Acid test -- exhaustive re-matching FAILED here
# (86.46 < conventional 86.53). Beating 86.53 means the Gaussian OOD signal
# suppresses the cross-task hijacks that broke exhaustive.
#
# Sweep knobs (env): COV_MODE=diagonal|full|tied  SHRINK=0.1  GLOGIT=0.0
# (LoRA classifier logit blend weight)  LOGDET=0|1 (add log-likelihood norm).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
DATA_PATH="${DATA_PATH:-${DATASETS_ROOT}}"
RUN_DIR="${1:-}"
COV_MODE="${COV_MODE:-diagonal}"
SHRINK="${SHRINK:-0.1}"
GLOGIT="${GLOGIT:-0.0}"
LOGDET="${LOGDET:-0}"

if [[ -z "${RUN_DIR}" ]]; then echo "Usage: $0 RUN_DIR" >&2; exit 64; fi
if [[ ! -x "${PYTHON_BIN}" ]]; then echo "Python not found" >&2; exit 1; fi
RUN_BASENAME="$(basename "${RUN_DIR}")"
SEED="${SEED:-}"; if [[ -z "${SEED}" && "${RUN_BASENAME}" =~ seed([0-9]+)$ ]]; then SEED="${BASH_REMATCH[1]}"; fi; SEED="${SEED:-42}"
TII_DIR="${TII_DIR:-${OUTPUT_ROOT}/cub200_tii_original_10tasks_seed${SEED}}"
TRAIN_LOG="${OUTPUT_ROOT}/${RUN_BASENAME}.log"
CONVENTIONAL_LOG="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_conventional.log"

for task_id in $(seq 1 10); do
  [[ -s "${RUN_DIR}/checkpoint/task${task_id}_checkpoint.pth" ]] || { echo "Missing LoRA checkpoint task${task_id}" >&2; exit 2; }
  [[ -s "${TII_DIR}/checkpoint/task${task_id}_checkpoint.pth" ]] || { echo "Missing TII checkpoint task${task_id}" >&2; exit 2; }
done
[[ -s "${TRAIN_LOG}" ]] || { echo "Missing training log: ${TRAIN_LOG}" >&2; exit 2; }

tag() { printf '%s' "$1" | tr '.' 'p'; }
LOGDET_FLAG=""; [[ "${LOGDET}" == "1" ]] && LOGDET_FLAG="--gaussian_include_logdet"
LOG_PATH="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_gaussian_${COV_MODE}_s$(tag "${SHRINK}")_g$(tag "${GLOGIT}")_ld${LOGDET}.log"
[[ -s "${LOG_PATH}" ]] && { echo "Refusing to overwrite: ${LOG_PATH}" >&2; exit 3; } || true

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -c '
import sys, torch
from protocols import validate_exemplar_free_training_log
validate_exemplar_free_training_log(sys.argv[1])
c = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
if c.get("real_feature_memory"): raise SystemExit("STRICT_CHECKPOINT_AUDIT=FAIL (real_feature_memory present)")
s = c.get("model", c); m = [v for k, v in s.items() if k.endswith("lora_layer.k_lora_A")]
rank = int(m[0].shape[-1]) if len(m) == 1 else -1
if rank != 8: raise SystemExit(f"STRICT_CHECKPOINT_AUDIT=FAIL (expected rank 8, got {rank})")
print("STRICT_CHECKPOINT_AUDIT=PASS")
' "${TRAIN_LOG}" "${RUN_DIR}/checkpoint/task10_checkpoint.pth"

echo "Gaussian re-scoring on Split-CUB200; cov=${COV_MODE} shrink=${SHRINK} logit_weight=${GLOGIT} logdet=${LOGDET}"
START_TIME="$(date +%s)"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=1 --master_port="${MASTER_PORT:-29556}" \
  main.py imr_lora \
  --model vit_base_patch16_224 --original_model vit_base_patch16_224 \
  --batch-size "${EVAL_BATCH_SIZE:-24}" --epochs 1 --data-path "${DATA_PATH}" \
  --seed "${SEED}" --lr 0.03 --con 0.2 --lora_rank 8 \
  --En gen --tau -10 --K 5 --sched cosine --dataset Split-CUB200 \
  --lora_momentum 0.4 --lora_type hide --trained_original_model "${TII_DIR}" \
  --num_tasks 10 --ca_storage_efficient_method covariance \
  --gaussian_rescoring --gaussian_cov_mode "${COV_MODE}" \
  --gaussian_shrinkage "${SHRINK}" --gaussian_logit_weight "${GLOGIT}" \
  ${LOGDET_FLAG} \
  --strict_exemplar_free --eval --output_dir "${RUN_DIR}" 2>&1 | tee "${LOG_PATH}"
printf 'Gaussian re-scoring wall time seconds: %s\n' "$(( $(date +%s) - START_TIME ))" | tee -a "${LOG_PATH}"
echo "Final Gaussian re-scoring metrics:"; grep "Average accuracy till task10" "${LOG_PATH}" | tail -n 1 || true

if [[ -s "${CONVENTIONAL_LOG}" ]]; then
  "${PYTHON_BIN}" - "${CONVENTIONAL_LOG}" "${LOG_PATH}" <<'PY' | tee -a "${LOG_PATH}"
import re, sys
def final_row(p):
    with open(p, 'r', encoding='utf-8') as h:
        r = [l.strip() for l in h if 'Average accuracy till task10' in l]
    if not r: raise SystemExit('GAUSSIAN_GATE=FAIL (missing task-10 row)')
    return r[-1]
def metric(row, name):
    mm = re.search(rf'{re.escape(name)}:\s*(-?[0-9]+(?:\.[0-9]+)?)', row)
    if not mm: raise SystemExit(f'GAUSSIAN_GATE=FAIL (missing {name})')
    return float(mm.group(1))
base, prop = final_row(sys.argv[1]), final_row(sys.argv[2])
checks = {}
print('Comparison with conventional HRM-PET baseline:')
for name in ('Acc@task', 'Acc@1', 'Acc@5'):
    d = metric(prop, name) - metric(base, name); checks[name] = d > 0.0
    print(f'  {name} delta={d:+.4f}: {"PASS" if checks[name] else "FAIL"}')
for name in ('Loss',):
    d = metric(prop, name) - metric(base, name); checks[name] = d < 0.0
    print(f'  {name} delta={d:+.4f}: {"PASS" if checks[name] else "FAIL"}')
print('GAUSSIAN_GATE=' + ('PASS' if checks.get('Acc@1', False) else 'FAIL'))
PY
fi
