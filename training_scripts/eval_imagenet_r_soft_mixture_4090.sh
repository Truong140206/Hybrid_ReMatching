#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
RUN_DIR="${1:-}"
TOP_K="${2:-4}"
TASK_TEMPERATURE="${3:-1.0}"
TII_PRIOR_WEIGHT="${4:-0.3}"
LOGIT_TEMPERATURE="${5:-1.0}"
MODE="${6:-soft}"
LORA_RANK="${LORA_RANK:-}"
TII_DIR="${TII_DIR:-${OUTPUT_ROOT}/imr_tii_original_10tasks_seed42}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: $0 RUN_DIR [top_k] [task_temperature] [tii_prior_weight] [logit_temperature] [soft|hard]" >&2
  exit 64
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if (( TOP_K < 1 || TOP_K > 10 )); then
  echo "top_k must be between 1 and 10" >&2
  exit 64
fi

case "${MODE}" in
  soft)
    REMATCHING_FLAG="--soft_mixture_rematching"
    METHOD_TAG="soft_mixture"
    METHOD_LABEL="Soft-mixture"
    EXPECTED_LORA_COST="4.0001"
    EXPECTED_FORWARD_CALLS="1.0001"
    ;;
  hard)
    REMATCHING_FLAG="--soft_mixture_hard_rematching"
    METHOD_TAG="soft_mixture_hard"
    METHOD_LABEL="Soft-hard"
    EXPECTED_LORA_COST="5.0001"
    EXPECTED_FORWARD_CALLS="2.0001"
    ;;
  *)
    echo "mode must be 'soft' or 'hard'" >&2
    exit 64
    ;;
esac

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

CHECKPOINT_RANK="$("${PYTHON_BIN}" -c '
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
state = checkpoint.get("model", checkpoint)
matches = [
    value for key, value in state.items()
    if key.endswith("lora_layer.k_lora_A")
]
if len(matches) != 1:
    raise SystemExit(
        f"Expected one lora_layer.k_lora_A tensor, found {len(matches)}")
print(int(matches[0].shape[-1]))
' "${RUN_DIR}/checkpoint/task1_checkpoint.pth")"

if [[ -z "${LORA_RANK}" ]]; then
  LORA_RANK="${CHECKPOINT_RANK}"
elif [[ "${LORA_RANK}" != "${CHECKPOINT_RANK}" ]]; then
  echo "Requested LoRA rank ${LORA_RANK}, but checkpoint rank is ${CHECKPOINT_RANK}" >&2
  exit 2
fi
echo "Detected LoRA rank from checkpoint: ${LORA_RANK}"

tag_value() {
  printf '%s' "$1" | tr '.' 'p'
}

RUN_BASENAME="$(basename "${RUN_DIR}")"
TASK_TEMP_TAG="$(tag_value "${TASK_TEMPERATURE}")"
PRIOR_TAG="$(tag_value "${TII_PRIOR_WEIGHT}")"
LOGIT_TEMP_TAG="$(tag_value "${LOGIT_TEMPERATURE}")"
LOG_PATH="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_${METHOD_TAG}_k${TOP_K}_tt${TASK_TEMP_TAG}_p${PRIOR_TAG}_lt${LOGIT_TEMP_TAG}.log"
BASELINE_LOG="${BASELINE_LOG:-${OUTPUT_ROOT}/${RUN_BASENAME}.log}"
EXHAUSTIVE_LOG="${EXHAUSTIVE_LOG:-${OUTPUT_ROOT}/${RUN_BASENAME}_eval_vectorized_exhaustive_c4_p${PRIOR_TAG}_t${LOGIT_TEMP_TAG}.log}"

if [[ ! -s "${EXHAUSTIVE_LOG}" ]]; then
  for candidate in \
      "${OUTPUT_ROOT}/${RUN_BASENAME}"_eval_vectorized_exhaustive*.log \
      "${OUTPUT_ROOT}/${RUN_BASENAME}"_eval_arrow_oracle_audit*.log \
      "${OUTPUT_ROOT}/${RUN_BASENAME}"_eval_lora_response_oracle_audit*.log; do
    if [[ -s "${candidate}" ]]; then
      EXHAUSTIVE_LOG="${candidate}"
      echo "Using exhaustive fallback reference: ${EXHAUSTIVE_LOG}"
      break
    fi
  done
fi

if [[ -s "${LOG_PATH}" ]]; then
  if grep -q "${METHOD_LABEL} wall time seconds:" "${LOG_PATH}"; then
    echo "Refusing to overwrite completed evaluation log: ${LOG_PATH}" >&2
    exit 3
  fi
  FAILED_LOG="${LOG_PATH%.log}_failed_$(date +%Y%m%d_%H%M%S).log"
  mv "${LOG_PATH}" "${FAILED_LOG}"
  echo "Archived incomplete evaluation log: ${FAILED_LOG}"
fi

cd "${REPO_ROOT}"
if [[ "${MODE}" == "hard" ]]; then
  echo "Soft-hard rematching: TII top-${TOP_K} mixture routes, then one selected LoRA classifies"
else
  echo "Soft-mixture rematching: TII top-${TOP_K}, posterior-weighted LoRA residuals, one model forward"
fi
echo "Run=${RUN_DIR}; rank=${LORA_RANK}; task_temperature=${TASK_TEMPERATURE}; prior=${TII_PRIOR_WEIGHT}; logit_temperature=${LOGIT_TEMPERATURE}"

START_TIME="$(date +%s)"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT:-29587}" \
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
  "${REMATCHING_FLAG}" \
  --soft_mixture_top_k "${TOP_K}" \
  --soft_mixture_task_temperature "${TASK_TEMPERATURE}" \
  --soft_mixture_tii_prior_weight "${TII_PRIOR_WEIGHT}" \
  --soft_mixture_logit_temperature "${LOGIT_TEMPERATURE}" \
  --eval \
  --output_dir "${RUN_DIR}" \
  2>&1 | tee "${LOG_PATH}"

END_TIME="$(date +%s)"
ELAPSED="$((END_TIME - START_TIME))"
printf '%s wall time seconds: %s\n' "${METHOD_LABEL}" "${ELAPSED}" | tee -a "${LOG_PATH}"
FINAL_LINE="$(grep "Average accuracy till task10" "${LOG_PATH}" | tail -n 1 || true)"
echo "Final ${METHOD_LABEL} metrics:"
echo "${FINAL_LINE}"

compare_reference() {
  local label="$1"
  local reference_log="$2"
  if [[ ! -s "${reference_log}" ]]; then
    echo "${label} reference not found; comparison skipped: ${reference_log}"
    return
  fi
  local reference_line
  reference_line="$(grep "Average accuracy till task10" "${reference_log}" | tail -n 1 || true)"
  "${PYTHON_BIN}" -c '
import re
import sys

label, candidate, reference = sys.argv[1:]
higher = ("Acc@task", "Acc@1", "Acc@5", "Backward")
lower = ("Loss", "Forgetting")
def metric(line, name):
    match = re.search(re.escape(name) + r":\s*([-+0-9.]+)", line)
    if match is None:
        raise SystemExit("Missing metric: " + name)
    return float(match.group(1))

passed = True
print(label + " comparison:")
for name in higher:
    delta = metric(candidate, name) - metric(reference, name)
    ok = delta >= 0.0
    passed = passed and ok
    print(f"  {name} delta={delta:+.4f}: " + ("PASS" if ok else "FAIL"))
for name in lower:
    delta = metric(candidate, name) - metric(reference, name)
    ok = delta <= 0.0
    passed = passed and ok
    print(f"  {name} delta={delta:+.4f}: " + ("PASS" if ok else "FAIL"))
print(label.upper().replace(" ", "_") + "_GATE=" + ("PASS" if passed else "FAIL"))
' "${label}" "${FINAL_LINE}" "${reference_line}"
}

compare_reference "baseline" "${BASELINE_LOG}"
compare_reference "exhaustive" "${EXHAUSTIVE_LOG}"

"${PYTHON_BIN}" -c '
import re
import sys

line = sys.argv[1]
mode, expected_cost, expected_calls = sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
def metric(name):
    match = re.search(re.escape(name) + r":\s*([-+0-9.]+)", line)
    if match is None:
        raise SystemExit("Missing metric: " + name)
    return float(match.group(1))

cost = metric("LoRA/sample")
calls = metric("ForwardCalls/sample")
passed = cost <= expected_cost and calls <= expected_calls
print(f"Efficiency: LoRA/sample={cost:.4f}; ForwardCalls/sample={calls:.4f}")
print(mode.upper().replace("-", "_") + "_EFFICIENCY_GATE=" + ("PASS" if passed else "FAIL"))
' "${FINAL_LINE}" "${METHOD_LABEL}" "${EXPECTED_LORA_COST}" "${EXPECTED_FORWARD_CALLS}"
