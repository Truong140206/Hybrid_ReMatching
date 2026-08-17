#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-imr_lora_rank8_baseline_10tasks_seed${SEED}}"
RUN_DIR="${1:-${OUTPUT_ROOT}/${RUN_NAME}}"
TII_DIR="${TII_DIR:-${OUTPUT_ROOT}/imr_tii_original_10tasks_seed${SEED}}"
TRAIN_LOG="${OUTPUT_ROOT}/$(basename "${RUN_DIR}").log"
LOG_PATH="${OUTPUT_ROOT}/$(basename "${RUN_DIR}")_eval_router_recall_audit.log"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 1
fi
for task_id in $(seq 1 10); do
  [[ -s "${RUN_DIR}/checkpoint/task${task_id}_checkpoint.pth" ]] || {
    echo "Missing strict LoRA checkpoint: ${RUN_DIR}/checkpoint/task${task_id}_checkpoint.pth" >&2
    exit 2
  }
  [[ -s "${TII_DIR}/checkpoint/task${task_id}_checkpoint.pth" ]] || {
    echo "Missing TII checkpoint: ${TII_DIR}/checkpoint/task${task_id}_checkpoint.pth" >&2
    exit 2
  }
done
[[ -s "${TRAIN_LOG}" ]] || { echo "Missing training log: ${TRAIN_LOG}" >&2; exit 2; }
if [[ -s "${LOG_PATH}" ]]; then
  echo "Refusing to overwrite existing audit log: ${LOG_PATH}" >&2
  exit 3
fi

if [[ -n "${DATA_PATH:-}" ]]; then
  IMR_DATA_PATH="${DATA_PATH}"
elif [[ -d "${DATASETS_ROOT}/imagenet-r/imagenet-r" || -f "${DATASETS_ROOT}/imagenet-r/imagenet-r.tar" ]]; then
  IMR_DATA_PATH="${DATASETS_ROOT}/imagenet-r"
else
  IMR_DATA_PATH="${DATASETS_ROOT}"
fi

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -c '
import sys
import torch
from protocols import validate_exemplar_free_training_log
validate_exemplar_free_training_log(sys.argv[1])
checkpoint = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
if checkpoint.get("real_feature_memory"):
    raise SystemExit("STRICT_CHECKPOINT_AUDIT=FAIL (real_feature_memory present)")
rank = getattr(checkpoint.get("args"), "lora_rank", None)
if rank != 8:
    raise SystemExit(f"STRICT_CHECKPOINT_AUDIT=FAIL (expected LoRA rank 8, got {rank})")
print("STRICT_CHECKPOINT_AUDIT=PASS")
' "${TRAIN_LOG}" "${RUN_DIR}/checkpoint/task10_checkpoint.pth"

"${PYTHON_BIN}" -m pytest -q \
  tests/test_router_recall_audit.py \
  tests/test_progressive_oracle_audit.py

echo "Router-recall audit: rank the exhaustive winner under TII max/energy/margin/mean aggregations."
echo "All scores use only TII logits (no LoRA), so any aggregation ranking the winner higher than max is a free strict cost reduction."

START_TIME="$(date +%s)"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT:-29591}" \
  main.py \
  imr_lora \
  --model vit_base_patch16_224 \
  --original_model vit_base_patch16_224 \
  --batch-size "${EVAL_BATCH_SIZE:-24}" \
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
  --progressive_oracle_audit \
  --router_recall_audit \
  --prediction_proposal_initial_count 2 \
  --prediction_proposal_top_classes 5 \
  --progressive_tii_prior_weight 0.3 \
  --progressive_logit_temperature 1.0 \
  --strict_exemplar_free \
  --eval \
  --output_dir "${RUN_DIR}" \
  2>&1 | tee "${LOG_PATH}"

END_TIME="$(date +%s)"
printf 'Router-recall audit wall time seconds: %s\n' "$((END_TIME - START_TIME))" | tee -a "${LOG_PATH}"

"${PYTHON_BIN}" - "${LOG_PATH}" <<'PY' | tee -a "${LOG_PATH}"
import re
import sys

with open(sys.argv[1], 'r', encoding='utf-8') as handle:
    rows = [line.strip() for line in handle
            if 'Average accuracy till task10' in line]
if not rows:
    raise SystemExit('ROUTER_RECALL_AUDIT=FAIL (missing task-10 metrics)')
row = rows[-1]

def metric(name):
    match = re.search(rf'{re.escape(name)}:\s*(-?[0-9]+(?:\.[0-9]+)?)', row)
    return float(match.group(1)) if match else float('nan')

routers = ('max', 'energy', 'margin', 'mean')
print('Router winner-ranking (lower MeanRank / higher Recall is better):')
print('  router   MeanRank  Recall@1  Recall@2  Recall@3  Recall@4')
data = {}
for name in routers:
    mean_rank = metric(f'Router_{name}_MeanRank')
    recalls = [metric(f'Router_{name}_Recall@{k}') for k in (1, 2, 3, 4)]
    data[name] = (mean_rank, recalls)
    print('  {:<7}  {:>7.4f}  {:>7.3f}  {:>7.3f}  {:>7.3f}  {:>7.3f}'.format(
        name, mean_rank, *recalls))

base_rank = data['max'][0]
base_recall2 = data['max'][1][1]
best = min(routers, key=lambda n: data[n][0])
rank_gain = base_rank - data[best][0]
recall2_gain = data[best][1][1] - base_recall2
print(f'Current router is "max" (MeanRank {base_rank:.4f}, Recall@2 {base_recall2:.3f}).')
print(f'Best alternative is "{best}" (MeanRank gain {rank_gain:+.4f}, '
      f'Recall@2 gain {recall2_gain:+.3f}).')
# Headroom is meaningful only if an alternative both lowers the winner rank and
# lifts top-2 recall by a non-trivial margin.
if best != 'max' and rank_gain >= 0.05 and recall2_gain >= 0.5:
    print('ROUTER_RECALL_AUDIT=HEADROOM_FOUND')
else:
    print('ROUTER_RECALL_AUDIT=NO_FREE_HEADROOM')
PY
