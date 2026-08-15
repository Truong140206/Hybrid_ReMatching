#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
RUN_DIR="${1:-}"
LORA_RANK="${LORA_RANK:-}"
AUDIT_INITIAL_BRANCH="${AUDIT_INITIAL_BRANCH:-0}"
AUDIT_CROSS_ADAPTER="${AUDIT_CROSS_ADAPTER:-0}"
TASK_MASS_FUSION="${TASK_MASS_FUSION:-0}"
CONDITIONAL_FUSION="${CONDITIONAL_FUSION:-0}"

if [[ "${TASK_MASS_FUSION}" == "1" && "${CONDITIONAL_FUSION}" == "1" ]]; then
  echo "TASK_MASS_FUSION and CONDITIONAL_FUSION are mutually exclusive" >&2
  exit 64
fi
if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: $0 RUN_DIR" >&2
  exit 64
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 1
fi

RUN_BASENAME="$(basename "${RUN_DIR}")"
SEED="${SEED:-}"
if [[ -z "${SEED}" && "${RUN_BASENAME}" =~ seed([0-9]+)$ ]]; then
  SEED="${BASH_REMATCH[1]}"
fi
SEED="${SEED:-42}"
TII_DIR="${TII_DIR:-${OUTPUT_ROOT}/imr_tii_original_10tasks_seed${SEED}}"
TRAIN_LOG="${TRAIN_LOG:-${OUTPUT_ROOT}/${RUN_BASENAME}.log}"
BASELINE_LOG="${BASELINE_LOG:-}"
if [[ -z "${BASELINE_LOG}" ]]; then
  CONVENTIONAL_LOG="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_conventional.log"
  if [[ -s "${CONVENTIONAL_LOG}" ]]; then
    BASELINE_LOG="${CONVENTIONAL_LOG}"
  else
    BASELINE_LOG="${TRAIN_LOG}"
  fi
fi
EXHAUSTIVE_LOG="${EXHAUSTIVE_LOG:-${OUTPUT_ROOT}/${RUN_BASENAME}_eval_vectorized_exhaustive_c4_p0p3_t1p0.log}"

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

if [[ ! -s "${TRAIN_LOG}" ]]; then
  echo "Training log required for protocol audit: ${TRAIN_LOG}" >&2
  exit 2
fi

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -c '
import sys
from protocols import validate_exemplar_free_training_log
validate_exemplar_free_training_log(sys.argv[1])
' "${TRAIN_LOG}"

CHECKPOINT_RANK="$("${PYTHON_BIN}" -c '
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if checkpoint.get("real_feature_memory"):
    raise SystemExit(
        "Checkpoint protocol audit failed: real_feature_memory is present")
state = checkpoint.get("model", checkpoint)
matches = [value for key, value in state.items()
           if key.endswith("lora_layer.k_lora_A")]
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
if [[ ! -s "${EXHAUSTIVE_LOG}" ]]; then
  echo "Exhaustive reference required: ${EXHAUSTIVE_LOG}" >&2
  echo "Run eval_imagenet_r_vectorized_exhaustive_4090.sh first." >&2
  exit 2
fi

AUDIT_SUFFIX=""
AUDIT_ARGS=()
if [[ "${AUDIT_INITIAL_BRANCH}" == "1" ]]; then
  AUDIT_SUFFIX="_initial_branch_dominance_audit"
  AUDIT_ARGS+=(--prediction_proposal_initial_branch_audit)
fi
if [[ "${AUDIT_CROSS_ADAPTER}" == "1" ]]; then
  AUDIT_SUFFIX="${AUDIT_SUFFIX}_cross_adapter_borda_audit"
  AUDIT_ARGS+=(--prediction_proposal_cross_adapter_audit)
fi
if [[ "${TASK_MASS_FUSION}" == "1" ]]; then
  AUDIT_SUFFIX="${AUDIT_SUFFIX}_taskmass"
  AUDIT_ARGS+=(--prediction_proposal_task_mass_fusion)
fi
if [[ "${CONDITIONAL_FUSION}" == "1" ]]; then
  AUDIT_SUFFIX="${AUDIT_SUFFIX}_conditional"
  AUDIT_ARGS+=(--prediction_proposal_conditional_fusion)
fi
LOG_PATH="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_prediction_proposal_i2_p3_c5_tiicomplete_strict${AUDIT_SUFFIX}.log"
if [[ -s "${LOG_PATH}" ]]; then
  echo "Refusing to overwrite existing evaluation log: ${LOG_PATH}" >&2
  exit 3
fi

"${PYTHON_BIN}" -m pytest -q \
  tests/test_prediction_proposal_audit.py \
  tests/test_progressive_oracle_audit.py \
  tests/test_exemplar_free_protocol.py

echo "Operational prediction-induced task proposal rematching"
echo "TII top-2 + three post-LoRA class-prediction proposals with TII probability completion"
if [[ "${TASK_MASS_FUSION}" == "1" ]]; then
  echo "Fusion: P_TII(task|x) * P_LoRA(class|task,x); no learned calibration"
elif [[ "${CONDITIONAL_FUSION}" == "1" ]]; then
  echo "Fusion: conditional LoRA log-probability + fixed standardized TII prior"
fi
echo "Fixed deployment budget: at most 5 LoRAs/sample in 2 vectorized model calls"
echo "Run=${RUN_DIR}; seed=${SEED}; rank=${LORA_RANK}"
echo "Protocol source log: ${TRAIN_LOG}"
echo "Baseline reference: ${BASELINE_LOG}"
echo "Exhaustive reference: ${EXHAUSTIVE_LOG}"

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
  --seed "${SEED}" \
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
  --prediction_proposal_rematching \
  --prediction_proposal_initial_count 2 \
  --prediction_proposal_count 3 \
  --prediction_proposal_top_classes 5 \
  --prediction_proposal_tii_completion \
  "${AUDIT_ARGS[@]}" \
  --progressive_tii_prior_weight 0.3 \
  --progressive_logit_temperature 1.0 \
  --strict_exemplar_free \
  --eval \
  --output_dir "${RUN_DIR}" \
  2>&1 | tee "${LOG_PATH}"

END_TIME="$(date +%s)"
printf 'Prediction-proposal evaluation wall time seconds: %s\n' "$((END_TIME - START_TIME))" | tee -a "${LOG_PATH}"
FINAL_LINE="$(grep "Average accuracy till task10" "${LOG_PATH}" | tail -n 1 || true)"
echo "Final operational prediction-proposal metrics:"
echo "${FINAL_LINE}"
if [[ "${AUDIT_INITIAL_BRANCH}" == "1" ]]; then
  echo "Per-task initial-branch/proposal complementarity (tasks 1-10):"
  grep "InitialProposalAudit" "${LOG_PATH}" | tail -n 10 | nl -w1 -s': '
fi
if [[ "${AUDIT_CROSS_ADAPTER}" == "1" ]]; then
  echo "Per-task cross-adapter consensus audit (tasks 1-10):"
  grep "CrossAdapterAudit" "${LOG_PATH}" | tail -n 10 | nl -w1 -s': '
  echo "Per-task Borda rank-consensus audit (tasks 1-10):"
  grep "CrossBordaAudit" "${LOG_PATH}" | tail -n 10 | nl -w1 -s': '
fi

"${PYTHON_BIN}" - "${LOG_PATH}" "${BASELINE_LOG}" "${EXHAUSTIVE_LOG}" <<'PY'
import re
import sys

METRICS = ('Acc@task', 'Acc@1', 'Acc@5', 'Loss', 'Forgetting', 'Backward')
HIGHER = ('Acc@task', 'Acc@1', 'Acc@5', 'Backward')
LOWER = ('Loss', 'Forgetting')


def final_row(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as handle:
        rows = [line.strip() for line in handle
                if 'Average accuracy till task10' in line]
    if not rows:
        raise SystemExit(f'Missing task-10 metrics in {path}')
    return rows[-1]


def metric(row, name):
    match = re.search(
        rf'{re.escape(name)}:\s*(-?[0-9]+(?:\.[0-9]+)?)', row)
    if not match:
        raise SystemExit(f'Missing {name}: {row}')
    return float(match.group(1))


candidate = final_row(sys.argv[1])
baseline = final_row(sys.argv[2])
exhaustive = final_row(sys.argv[3])

checks = {}
print('Comparison with conventional baseline:')
for name in HIGHER:
    delta = metric(candidate, name) - metric(baseline, name)
    checks[name] = delta >= 0.0
    print(f'  {name} delta={delta:+.4f}: {"PASS" if checks[name] else "FAIL"}')
for name in LOWER:
    delta = metric(candidate, name) - metric(baseline, name)
    checks[name] = delta <= 0.0
    print(f'  {name} delta={delta:+.4f}: {"PASS" if checks[name] else "FAIL"}')

print('Retention relative to exhaustive:')
for name in METRICS:
    delta = metric(candidate, name) - metric(exhaustive, name)
    print(f'  {name} delta={delta:+.4f}')

cost_ok = metric(candidate, 'LoRA/sample') <= 5.0001
call_ok = metric(candidate, 'ForwardCalls/sample') <= 2.0001
print(f'  LoRA/sample <= 5: {"PASS" if cost_ok else "FAIL"}')
print(f'  ForwardCalls/sample <= 2: {"PASS" if call_ok else "FAIL"}')
print('BASELINE_ALL_METRIC_GATE=' + (
    'PASS' if all(checks.values()) else 'FAIL'))
print('OPERATIONAL_PROPOSAL_EFFICIENCY_GATE=' + (
    'PASS' if cost_ok and call_ok else 'FAIL'))
if 'InitialProposalOracleAcc@1:' in candidate:
    initial = metric(candidate, 'InitialBranchAcc@1')
    proposal = metric(candidate, 'ProposalAuditAcc@1')
    oracle = metric(candidate, 'InitialProposalOracleAcc@1')
    initial_only = metric(candidate, 'InitialOnlyCorrect')
    proposal_only = metric(candidate, 'ProposalOnlyCorrect')
    dominance = metric(candidate, 'DominanceAcc@1')
    select_rate = metric(candidate, 'InitialSelectRate')
    print('Free initial-branch complementarity audit:')
    print(f'  Initial branch Acc@1={initial:.4f}')
    print(f'  Proposal Acc@1={proposal:.4f}')
    print(f'  Initial-only correct={initial_only:.4f}')
    print(f'  Proposal-only correct={proposal_only:.4f}')
    print(f'  Label oracle Acc@1={oracle:.4f}; headroom={oracle - proposal:+.4f}')
    print(f'  Parameter-free dominance Acc@1={dominance:.4f}; '
          f'gain={dominance - proposal:+.4f}; '
          f'initial select rate={select_rate:.4f}')
    viable = initial_only >= 0.25 and proposal_only > 0.0
    print('INITIAL_BRANCH_COMPLEMENTARITY_AUDIT=' + (
        'PASS' if viable else 'FAIL'))
    print('PARAMETER_FREE_DOMINANCE_GATE=' + (
        'PASS' if dominance >= proposal else 'FAIL'))
if 'CrossAdapterOracleAcc@1:' in candidate:
    proposal = metric(candidate, 'Acc@1')
    vote = metric(candidate, 'CrossVoteAcc@1')
    adapter_oracle = metric(candidate, 'CrossAdapterOracleAcc@1')
    vote_only = metric(candidate, 'CrossVoteOnlyCorrect')
    proposal_only = metric(candidate, 'ProposalOnlyVsCrossVote')
    proposal_vote_oracle = metric(candidate, 'CrossProposalOracleAcc@1')
    rescue = metric(candidate, 'CrossRescueAcc@1')
    rescue_rate = metric(candidate, 'CrossRescueRate')
    borda = metric(candidate, 'CrossBordaAcc@1')
    borda_only = metric(candidate, 'CrossBordaOnlyCorrect')
    proposal_only_borda = metric(candidate, 'ProposalOnlyVsCrossBorda')
    proposal_borda_oracle = metric(
        candidate, 'CrossBordaProposalOracleAcc@1')
    borda_rescue = metric(candidate, 'CrossBordaRescueAcc@1')
    borda_rescue_rate = metric(candidate, 'CrossBordaRescueRate')
    borda_support = metric(candidate, 'CrossBordaTop5Support')
    print('Cross-adapter full-logit audit:')
    print(f'  Proposal Acc@1={proposal:.4f}')
    print(f'  Global plurality vote Acc@1={vote:.4f}')
    print(f'  Any-adapter label oracle Acc@1={adapter_oracle:.4f}; '
          f'headroom={adapter_oracle - proposal:+.4f}')
    print(f'  Vote-only correct={vote_only:.4f}; '
          f'proposal-only correct={proposal_only:.4f}')
    print(f'  Proposal/vote oracle Acc@1={proposal_vote_oracle:.4f}')
    print(f'  Strict-majority rescue Acc@1={rescue:.4f}; '
          f'gain={rescue - proposal:+.4f}; rescue rate={rescue_rate:.4f}')
    print('CROSS_ADAPTER_HEADROOM_GATE=' + (
        'PASS' if adapter_oracle - proposal >= 0.25 else 'FAIL'))
    print('CROSS_ADAPTER_RESCUE_GATE=' + (
        'PASS' if rescue >= proposal else 'FAIL'))
    print('Calibration-free Borda rank audit:')
    print(f'  Borda Acc@1={borda:.4f}')
    print(f'  Borda-only correct={borda_only:.4f}; '
          f'proposal-only correct={proposal_only_borda:.4f}')
    print(f'  Proposal/Borda oracle Acc@1={proposal_borda_oracle:.4f}')
    print(f'  Majority-top5 Borda rescue Acc@1={borda_rescue:.4f}; '
          f'gain={borda_rescue - proposal:+.4f}; '
          f'rescue rate={borda_rescue_rate:.4f}; '
          f'average top5 support={borda_support:.4f}')
    print('CROSS_BORDA_HEADROOM_GATE=' + (
        'PASS' if proposal_borda_oracle - proposal >= 0.25 else 'FAIL'))
    print('CROSS_BORDA_RESCUE_GATE=' + (
        'PASS' if borda_rescue >= proposal else 'FAIL'))
PY
