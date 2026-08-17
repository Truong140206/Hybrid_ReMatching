#!/usr/bin/env bash
set -euo pipefail

# Vectorized exact exhaustive rematching on Split-CUB200 (all task LoRAs).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
DATA_PATH="${DATA_PATH:-${DATASETS_ROOT}}"
RUN_DIR="${1:-}"; TASK_CHUNK_SIZE="${2:-4}"; TII_PRIOR_WEIGHT="${3:-0.3}"; LOGIT_TEMPERATURE="${4:-1.0}"
LORA_RANK="${LORA_RANK:-}"

if [[ -z "${RUN_DIR}" ]]; then echo "Usage: $0 RUN_DIR [chunk] [prior] [temp]" >&2; exit 64; fi
if [[ ! -x "${PYTHON_BIN}" ]]; then echo "Python not found" >&2; exit 1; fi
RUN_BASENAME="$(basename "${RUN_DIR}")"
SEED="${SEED:-}"; if [[ -z "${SEED}" && "${RUN_BASENAME}" =~ seed([0-9]+)$ ]]; then SEED="${BASH_REMATCH[1]}"; fi; SEED="${SEED:-42}"
TII_DIR="${TII_DIR:-${OUTPUT_ROOT}/cub200_tii_original_10tasks_seed${SEED}}"

for task_id in $(seq 1 10); do
  [[ -s "${RUN_DIR}/checkpoint/task${task_id}_checkpoint.pth" ]] || { echo "Missing LoRA checkpoint task${task_id}" >&2; exit 2; }
  [[ -s "${TII_DIR}/checkpoint/task${task_id}_checkpoint.pth" ]] || { echo "Missing TII checkpoint task${task_id}" >&2; exit 2; }
done

cd "${REPO_ROOT}"
CHECKPOINT_RANK="$("${PYTHON_BIN}" -c '
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if c.get("real_feature_memory"): raise SystemExit("real_feature_memory present")
s = c.get("model", c); m = [v for k, v in s.items() if k.endswith("lora_layer.k_lora_A")]
if len(m) != 1: raise SystemExit(f"Expected one k_lora_A, found {len(m)}")
print(int(m[0].shape[-1]))
' "${RUN_DIR}/checkpoint/task1_checkpoint.pth")"
[[ -z "${LORA_RANK}" ]] && LORA_RANK="${CHECKPOINT_RANK}"

tag() { printf '%s' "$1" | tr '.' 'p'; }
LOG_PATH="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_vectorized_exhaustive_c${TASK_CHUNK_SIZE}_p$(tag "${TII_PRIOR_WEIGHT}")_t$(tag "${LOGIT_TEMPERATURE}").log"
[[ -s "${LOG_PATH}" ]] && { echo "Refusing to overwrite: ${LOG_PATH}" >&2; exit 3; } || true

echo "Vectorized exhaustive on Split-CUB200; rank=${LORA_RANK}"
START_TIME="$(date +%s)"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=1 --master_port="${MASTER_PORT:-29553}" \
  main.py imr_lora \
  --model vit_base_patch16_224 --original_model vit_base_patch16_224 \
  --batch-size "${EVAL_BATCH_SIZE:-24}" --epochs 1 --data-path "${DATA_PATH}" \
  --seed "${SEED}" --lr 0.03 --con 0.2 --lora_rank "${LORA_RANK}" \
  --En gen --tau -10 --K 5 --sched cosine --dataset Split-CUB200 \
  --lora_momentum 0.4 --lora_type hide --trained_original_model "${TII_DIR}" \
  --num_tasks 10 --vectorized_exhaustive_rematching \
  --vectorized_exhaustive_task_chunk_size "${TASK_CHUNK_SIZE}" \
  --exhaustive_tii_prior_weight "${TII_PRIOR_WEIGHT}" \
  --exhaustive_logit_temperature "${LOGIT_TEMPERATURE}" \
  --strict_exemplar_free --eval --output_dir "${RUN_DIR}" 2>&1 | tee "${LOG_PATH}"
printf 'Vectorized exhaustive wall time seconds: %s\n' "$(( $(date +%s) - START_TIME ))" | tee -a "${LOG_PATH}"
echo "Final vectorized exhaustive metrics:"; grep "Average accuracy till task10" "${LOG_PATH}" | tail -n 1 || true
