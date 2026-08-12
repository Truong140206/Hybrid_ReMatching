#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
  echo "Usage: $0 [smoke|full]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
SEED="${SEED:-42}"
GPUS="${GPUS:-1}"

if [[ -n "${DATA_PATH:-}" ]]; then
  IMR_DATA_PATH="${DATA_PATH}"
elif [[ -d "${DATASETS_ROOT}/imagenet-r/imagenet-r" || -f "${DATASETS_ROOT}/imagenet-r/imagenet-r.tar" ]]; then
  IMR_DATA_PATH="${DATASETS_ROOT}/imagenet-r"
else
  IMR_DATA_PATH="${DATASETS_ROOT}"
fi

TII_DIR="${TII_DIR:-${OUTPUT_ROOT}/imr_tii_original_10tasks_seed${SEED}}"
LORA_EPOCHS="${LORA_EPOCHS:-50}"
CRCT_EPOCHS="${CRCT_EPOCHS:-30}"
CFS_EPOCHS="${CFS_EPOCHS:-20}"
MAX_TRAIN_TASKS="${MAX_TRAIN_TASKS:-0}"
RUN_NAME="${RUN_NAME_OVERRIDE:-imr_lora_hybrid_real_cfs_crct${CRCT_EPOCHS}_r025_m48_seed${SEED}}"
SEMANTIC_DISTILL="${SEMANTIC_DISTILL:-0}"
SEMANTIC_CLASS_NAME_FILE="${SEMANTIC_CLASS_NAME_FILE:-${REPO_ROOT}/configs/imagenet_class_names.json}"

if [[ "${MODE}" == "smoke" ]]; then
  LORA_EPOCHS="${SMOKE_LORA_EPOCHS:-1}"
  CRCT_EPOCHS="${SMOKE_CRCT_EPOCHS:-1}"
  CFS_EPOCHS="${SMOKE_CFS_EPOCHS:-1}"
  MAX_TRAIN_TASKS="${SMOKE_TASKS:-2}"
  RUN_NAME="${RUN_NAME_OVERRIDE:-imr_hybrid_real_replay_smoke_seed${SEED}}"
fi

OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_PATH="${OUTPUT_ROOT}/${RUN_NAME}.log"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 2
fi

SEMANTIC_ARGS=()
if [[ "${SEMANTIC_DISTILL}" == "1" ]]; then
  if [[ ! -s "${SEMANTIC_CLASS_NAME_FILE}" ]]; then
    echo "Semantic class-name mapping not found: ${SEMANTIC_CLASS_NAME_FILE}" >&2
    exit 2
  fi
  if ! "${PYTHON_BIN}" -c "import open_clip" >/dev/null 2>&1; then
    echo "open_clip_torch is required for semantic CTIRD." >&2
    echo "Install it with: ${PYTHON_BIN} -m pip install open_clip_torch ftfy regex" >&2
    exit 2
  fi
  SEMANTIC_ARGS=(
    --semantic_distill
    --semantic_backend clip
    --semantic_clip_model "${SEMANTIC_CLIP_MODEL:-ViT-B-16}"
    --semantic_clip_pretrained "${SEMANTIC_CLIP_PRETRAINED:-openai}"
    --semantic_clip_templates "${SEMANTIC_CLIP_TEMPLATES:-a photo of a {}.|a painting of a {}.|a rendition of a {}.}"
    --semantic_class_name_file "${SEMANTIC_CLASS_NAME_FILE}"
    --semantic_mode "${SEMANTIC_MODE:-topk_mix}"
    --semantic_top_k "${SEMANTIC_TOP_K:-3}"
    --semantic_alpha "${SEMANTIC_ALPHA:-0.03}"
  )
fi

for task_id in $(seq 1 10); do
  checkpoint="${TII_DIR}/checkpoint/task${task_id}_checkpoint.pth"
  if [[ ! -s "${checkpoint}" ]]; then
    echo "Missing TII checkpoint: ${checkpoint}" >&2
    exit 3
  fi
done

if [[ -d "${OUTPUT_DIR}/checkpoint" ]] && compgen -G "${OUTPUT_DIR}/checkpoint/task*_checkpoint.pth" > /dev/null; then
  echo "Refusing to overwrite checkpoints in: ${OUTPUT_DIR}" >&2
  echo "Set RUN_NAME_OVERRIDE to a new name before rerunning." >&2
  exit 4
fi

if [[ -s "${LOG_PATH}" ]]; then
  echo "Refusing to overwrite existing log: ${LOG_PATH}" >&2
  echo "Set RUN_NAME_OVERRIDE to a new name before rerunning." >&2
  exit 4
fi

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

echo "Mode: ${MODE}"
echo "Dataset: ${IMR_DATA_PATH}"
echo "TII checkpoints: ${TII_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Real replay: old/new=${REAL_OLD_REPLAY_RATIO:-0.35}/${REAL_NEW_REPLAY_RATIO:-0.10}, memory/class=${REAL_MEMORY_PER_CLASS:-48}, hard ratio=${REAL_HARD_RATIO:-0.5}"
if [[ "${SEMANTIC_DISTILL}" == "1" ]]; then
  echo "Semantic CTIRD: backend=CLIP, mode=${SEMANTIC_MODE:-topk_mix}, top-k=${SEMANTIC_TOP_K:-3}, alpha=${SEMANTIC_ALPHA:-0.03}"
  echo "Semantic class names: ${SEMANTIC_CLASS_NAME_FILE}"
fi

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT:-29541}" \
  main.py \
  imr_lora \
  --model vit_base_patch16_224 \
  --original_model vit_base_patch16_224 \
  --batch-size "${LORA_BATCH_SIZE:-24}" \
  --epochs "${LORA_EPOCHS}" \
  --data-path "${IMR_DATA_PATH}" \
  --ca_lr "${CA_LR:-0.005}" \
  --crct_epochs "${CRCT_EPOCHS}" \
  --seed "${SEED}" \
  --lr 0.03 \
  --con 0.2 \
  --lora_rank 5 \
  --En gen \
  --tau -10 \
  --K 5 \
  --sched cosine \
  --dataset Split-Imagenet-R \
  --lora_momentum 0.4 \
  --lora_type hide \
  --trained_original_model "${TII_DIR}" \
  --num_tasks 10 \
  --max_train_tasks "${MAX_TRAIN_TASKS}" \
  --cfs_sampling \
  --cfs_epochs "${CFS_EPOCHS}" \
  --cfs_train_max_samples "${CFS_TRAIN_MAX_SAMPLES:-1024}" \
  --cfs_candidate_multiplier "${CFS_CANDIDATE_MULTIPLIER:-2}" \
  --cfs_boundary_replay \
  --cfs_boundary_ratio "${CFS_BOUNDARY_RATIO:-0.10}" \
  --cfs_boundary_multiplier "${CFS_BOUNDARY_MULTIPLIER:-3}" \
  --cfs_boundary_density_quantile "${CFS_BOUNDARY_DENSITY_QUANTILE:-0.9}" \
  --crct_real_feature_replay \
  --crct_real_memory_per_class "${REAL_MEMORY_PER_CLASS:-48}" \
  --crct_real_replay_ratio "${REAL_REPLAY_RATIO:-0.25}" \
  --crct_real_old_replay_ratio "${REAL_OLD_REPLAY_RATIO:-0.35}" \
  --crct_real_new_replay_ratio "${REAL_NEW_REPLAY_RATIO:-0.10}" \
  --crct_real_hard_ratio "${REAL_HARD_RATIO:-0.5}" \
  --crct_real_outlier_quantile "${REAL_OUTLIER_QUANTILE:-0.9}" \
  --crct_real_diversity_weight "${REAL_DIVERSITY_WEIGHT:-0.7}" \
  --crct_hybrid_samples_per_class "${HYBRID_SAMPLES_PER_CLASS:-120}" \
  --crct_use_all_samples \
  --crct_balanced_batches \
  "${SEMANTIC_ARGS[@]}" \
  --output_dir "${OUTPUT_DIR}" \
  2>&1 | tee "${LOG_PATH}"

EXPECTED_TASKS=10
if [[ "${MAX_TRAIN_TASKS}" -gt 0 ]]; then
  EXPECTED_TASKS="${MAX_TRAIN_TASKS}"
fi
for task_id in $(seq 1 "${EXPECTED_TASKS}"); do
  checkpoint="${OUTPUT_DIR}/checkpoint/task${task_id}_checkpoint.pth"
  if [[ ! -s "${checkpoint}" ]]; then
    echo "Missing checkpoint after training: ${checkpoint}" >&2
    exit 5
  fi
done

echo "Final metrics:"
grep "Average accuracy till task${EXPECTED_TASKS}" "${LOG_PATH}" | tail -n 1 || true
