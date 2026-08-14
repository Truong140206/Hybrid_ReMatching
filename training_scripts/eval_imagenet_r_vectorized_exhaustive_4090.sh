#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
RUN_DIR="${1:-}"
TASK_CHUNK_SIZE="${2:-4}"
TII_PRIOR_WEIGHT="${3:-0.3}"
LOGIT_TEMPERATURE="${4:-1.0}"
LORA_RANK="${LORA_RANK:-8}"
TII_DIR="${TII_DIR:-${OUTPUT_ROOT}/imr_tii_original_10tasks_seed42}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: $0 RUN_DIR [task_chunk_size] [tii_prior_weight] [logit_temperature]" >&2
  exit 64
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if (( TASK_CHUNK_SIZE < 1 || TASK_CHUNK_SIZE > 10 )); then
  echo "task_chunk_size must be between 1 and 10" >&2
  exit 64
fi

if [[ -n "${DATA_PATH:-}" ]]; then
  IMR_DATA_PATH="${DATA_PATH}"
elif [[ -d "${DATASETS_ROOT}/imagenet-r/imagenet-r" || -f "${DATASETS_ROOT}/imagenet-r/imagenet-r.tar" ]]; then
  IMR_DATA_PATH="${DATASETS_ROOT}/imagenet-r"
else
  IMR_DATA_PATH="${DATASETS_ROOT}"
fi

for task_id in $(seq 1 10); do
  [[ -s "${RUN_DIR}/checkpoint/task${task_id}_checkpoint.pth" ]] || {
    echo "Missing LoRA checkpoint: ${RUN_DIR}/checkpoint/task${task_id}_checkpoint.pth" >&2
    exit 2
  }
  [[ -s "${TII_DIR}/checkpoint/task${task_id}_checkpoint.pth" ]] || {
    echo "Missing TII checkpoint: ${TII_DIR}/checkpoint/task${task_id}_checkpoint.pth" >&2
    exit 2
  }
done

tag_value() {
  printf '%s' "$1" | tr '.' 'p'
}

RUN_BASENAME="$(basename "${RUN_DIR}")"
PRIOR_TAG="$(tag_value "${TII_PRIOR_WEIGHT}")"
TEMP_TAG="$(tag_value "${LOGIT_TEMPERATURE}")"
LOG_PATH="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_vectorized_exhaustive_c${TASK_CHUNK_SIZE}_p${PRIOR_TAG}_t${TEMP_TAG}.log"
REFERENCE_LOG="${REFERENCE_LOG:-${OUTPUT_ROOT}/${RUN_BASENAME}_eval_arrow_oracle_audit_p${PRIOR_TAG}_t${TEMP_TAG}.log}"
if [[ -s "${LOG_PATH}" ]]; then
  echo "Refusing to overwrite existing evaluation log: ${LOG_PATH}" >&2
  exit 3
fi

cd "${REPO_ROOT}"
echo "Vectorized exhaustive rematching: exact exhaustive math, ${TASK_CHUNK_SIZE} task LoRAs per forward call"
echo "Run=${RUN_DIR}; rank=${LORA_RANK}; prior=${TII_PRIOR_WEIGHT}; temperature=${LOGIT_TEMPERATURE}"

START_TIME="$(date +%s)"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT:-29583}" \
  main.py \
  imr_lora \
  --model vit_base_patch16_224 \
  --original_model vit_base_patch16_224 \
  --batch-size "${EVAL_BATCH_SIZE:-24}" \
  --epochs 1 \
  --data-path "${IMR_DATA_PATH}" \
  --seed 42 \
  --lr 0.03 \
  --con 0.2 \
  --lora_rank "${LORA_RANK}" \
  --En gen \
  --tau -10 \
  --K 5 \
  --sched cosine \
  --dataset Split-Imagenet-R \
  --lora_momentum 0.4 \
  --lora_type hide \
  --trained_original_model "${TII_DIR}" \
  --num_tasks 10 \
  --vectorized_exhaustive_rematching \
  --vectorized_exhaustive_task_chunk_size "${TASK_CHUNK_SIZE}" \
  --exhaustive_tii_prior_weight "${TII_PRIOR_WEIGHT}" \
  --exhaustive_logit_temperature "${LOGIT_TEMPERATURE}" \
  --eval \
  --output_dir "${RUN_DIR}" \
  2>&1 | tee "${LOG_PATH}"

END_TIME="$(date +%s)"
ELAPSED="$((END_TIME - START_TIME))"
printf 'Vectorized exhaustive wall time seconds: %s\n' "${ELAPSED}" | tee -a "${LOG_PATH}"
FINAL_LINE="$(grep "Average accuracy till task10" "${LOG_PATH}" | tail -n 1 || true)"
echo "Final vectorized exhaustive metrics:"
echo "${FINAL_LINE}"

if [[ -s "${REFERENCE_LOG}" ]]; then
  REFERENCE_LINE="$(grep "Average accuracy till task10" "${REFERENCE_LOG}" | tail -n 1 || true)"
  "${PYTHON_BIN}" -c '
import re
import sys

candidate, reference = sys.argv[1], sys.argv[2]
names = ("Acc@task", "Acc@1", "Acc@5", "Loss", "Forgetting", "Backward")
limits = {"Acc@task": 0.01, "Acc@1": 0.01, "Acc@5": 0.01,
          "Loss": 0.001, "Forgetting": 0.01, "Backward": 0.01}
def metric(line, name):
    match = re.search(re.escape(name) + r":\s*([-+0-9.]+)", line)
    if match is None:
        raise SystemExit("Missing metric: " + name)
    return float(match.group(1))

passed = True
for name in names:
    delta = metric(candidate, name) - metric(reference, name)
    ok = abs(delta) <= limits[name]
    passed = passed and ok
    print(f"{name} delta={delta:+.6f}: " + ("PASS" if ok else "FAIL"))
print("VECTORIZED_EQUIVALENCE_GATE=" + ("PASS" if passed else "FAIL"))
' "${FINAL_LINE}" "${REFERENCE_LINE}"

  REFERENCE_SECONDS="$(grep -E 'Arrow oracle audit wall time seconds:' "${REFERENCE_LOG}" | tail -n 1 | awk '{print $NF}' || true)"
  if [[ "${REFERENCE_SECONDS}" =~ ^[0-9]+$ ]] && (( ELAPSED > 0 )); then
    "${PYTHON_BIN}" -c 'import sys; print(f"Conservative wall-time speedup: {int(sys.argv[1]) / int(sys.argv[2]):.3f}x")' "${REFERENCE_SECONDS}" "${ELAPSED}"
  fi
else
  echo "Reference log not found; metric-equivalence gate skipped: ${REFERENCE_LOG}"
fi
