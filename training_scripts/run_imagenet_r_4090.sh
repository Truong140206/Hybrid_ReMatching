#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}" 
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
SEED="${SEED:-42}"
GPUS="${GPUS:-1}"
MODE="${1:-check}"

# Accept both datasets/imagenet-r/imagenet-r and datasets/imagenet-r layouts.
if [[ -n "${DATA_PATH:-}" ]]; then
  IMR_DATA_PATH="${DATA_PATH}"
elif [[ -d "${DATASETS_ROOT}/imagenet-r/imagenet-r" || -f "${DATASETS_ROOT}/imagenet-r/imagenet-r.tar" ]]; then
  IMR_DATA_PATH="${DATASETS_ROOT}/imagenet-r"
else
  IMR_DATA_PATH="${DATASETS_ROOT}"
fi

TII_DIR="${TII_DIR:-${OUTPUT_ROOT}/imr_tii_original_10tasks_seed${SEED}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -d "${IMR_DATA_PATH}/imagenet-r" && ! -f "${IMR_DATA_PATH}/imagenet-r.tar" ]]; then
  echo "ImageNet-R was not found under: ${IMR_DATA_PATH}" >&2
  echo "Expected imagenet-r/ or imagenet-r.tar in that directory." >&2
  echo "Override the location with DATA_PATH=/path/to/dataset-root." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

require_checkpoints() {
  local run_dir="$1"
  local count="$2"
  local label="$3"
  local task_id

  for task_id in $(seq 1 "${count}"); do
    if [[ ! -s "${run_dir}/checkpoint/task${task_id}_checkpoint.pth" ]]; then
      echo "Missing ${label} checkpoint: ${run_dir}/checkpoint/task${task_id}_checkpoint.pth" >&2
      return 1
    fi
  done
  echo "${label} checkpoints OK: task1..task${count}"
}

refuse_overwrite() {
  local output_dir="$1"
  local log_path="$2"

  if [[ -d "${output_dir}/checkpoint" ]] && compgen -G "${output_dir}/checkpoint/task*_checkpoint.pth" > /dev/null; then
    echo "Refusing to overwrite checkpoints in: ${output_dir}" >&2
    echo "Set RUN_NAME_OVERRIDE to a new name." >&2
    exit 2
  fi
  if [[ -s "${log_path}" ]]; then
    echo "Refusing to overwrite existing log: ${log_path}" >&2
    echo "Set RUN_NAME_OVERRIDE to a new name." >&2
    exit 2
  fi
}

run_tii() {
  local log_path="${OUTPUT_ROOT}/imr_tii_original_10tasks_seed${SEED}.log"

  if require_checkpoints "${TII_DIR}" 10 "TII"; then
    echo "TII is already complete; nothing to run."
    return 0
  fi

  if [[ -d "${TII_DIR}/checkpoint" ]] && compgen -G "${TII_DIR}/checkpoint/task*_checkpoint.pth" > /dev/null; then
    echo "TII has incomplete checkpoints in: ${TII_DIR}" >&2
    echo "The current code cannot safely resume midway through the task sequence." >&2
    echo "Keep that folder and set TII_DIR to a new directory before rerunning." >&2
    exit 3
  fi

  refuse_overwrite "${TII_DIR}" "${log_path}"
  mkdir -p "${TII_DIR}"
  cd "${REPO_ROOT}"

  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node="${GPUS}" \
    --master_port="${MASTER_PORT:-29530}" \
    main.py \
    imr_hideprompt_5e \
    --model vit_base_patch16_224 \
    --original_model vit_base_patch16_224 \
    --batch-size "${TII_BATCH_SIZE:-64}" \
    --ca_storage_efficient_method covariance \
    --epochs "${TII_EPOCHS:-20}" \
    --data-path "${IMR_DATA_PATH}" \
    --lr 0.0005 \
    --ca_lr 0.005 \
    --crct_epochs "${TII_CRCT_EPOCHS:-30}" \
    --seed "${SEED}" \
    --num_tasks 10 \
    --train_inference_task_only \
    --output_dir "${TII_DIR}" \
    2>&1 | tee "${log_path}"

  require_checkpoints "${TII_DIR}" 10 "TII"
}

run_lora() {
  local variant="$1"
  local epochs="${LORA_EPOCHS:-50}"
  local crct_epochs="${CRCT_EPOCHS:-30}"
  local run_name
  local master_port
  local extra_args=()

  require_checkpoints "${TII_DIR}" 10 "TII" || {
    echo "Run '$0 tii' first." >&2
    exit 4
  }

  case "${variant}" in
    baseline)
      run_name="imr_lora_baseline_10tasks_seed${SEED}"
      master_port="${MASTER_PORT:-29531}"
      ;;
    improved)
      run_name="imr_lora_cfs_energy_10tasks_seed${SEED}"
      master_port="${MASTER_PORT:-29532}"
      extra_args=(
        --cfs_sampling
        --cfs_epochs "${CFS_EPOCHS:-20}"
        --cfs_train_max_samples "${CFS_TRAIN_MAX_SAMPLES:-1024}"
        --cfs_candidate_multiplier "${CFS_CANDIDATE_MULTIPLIER:-2}"
        --cfs_boundary_replay
        --cfs_boundary_ratio "${CFS_BOUNDARY_RATIO:-0.25}"
        --cfs_boundary_multiplier "${CFS_BOUNDARY_MULTIPLIER:-3}"
        --cfs_boundary_density_quantile "${CFS_BOUNDARY_DENSITY_QUANTILE:-0.9}"
        --crct_use_all_samples
        --ctird_task_selection task_energy
        --ctird_task_temperature 0.1
        --ctird_task_weighting energy
        --ctird_weight_temperature 1.0
        --ctird_weight_floor 0.2
      )
      ;;
    quick)
      run_name="imr_lora_cfs_energy_quick_10tasks_seed${SEED}"
      master_port="${MASTER_PORT:-29533}"
      epochs=1
      crct_epochs=1
      extra_args=(
        --cfs_sampling
        --cfs_epochs 1
        --cfs_train_max_samples 256
        --cfs_candidate_multiplier 2
        --cfs_boundary_replay
        --cfs_boundary_ratio 0.25
        --cfs_boundary_multiplier 2
        --cfs_boundary_density_quantile 0.9
        --crct_use_all_samples
        --ctird_task_selection task_energy
        --ctird_task_temperature 0.1
        --ctird_task_weighting energy
        --ctird_weight_temperature 1.0
        --ctird_weight_floor 0.2
      )
      ;;
    *)
      echo "Unknown LoRA variant: ${variant}" >&2
      exit 5
      ;;
  esac

  run_name="${RUN_NAME_OVERRIDE:-${run_name}}"
  local output_dir="${OUTPUT_ROOT}/${run_name}"
  local log_path="${OUTPUT_ROOT}/${run_name}.log"

  refuse_overwrite "${output_dir}" "${log_path}"
  mkdir -p "${output_dir}"
  cd "${REPO_ROOT}"

  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node="${GPUS}" \
    --master_port="${master_port}" \
    main.py \
    imr_lora \
    --model vit_base_patch16_224 \
    --original_model vit_base_patch16_224 \
    --batch-size "${LORA_BATCH_SIZE:-24}" \
    --epochs "${epochs}" \
    --data-path "${IMR_DATA_PATH}" \
    --ca_lr "${CA_LR:-0.005}" \
    --crct_epochs "${crct_epochs}" \
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
    "${extra_args[@]}" \
    --output_dir "${output_dir}" \
    2>&1 | tee "${log_path}"

  require_checkpoints "${output_dir}" 10 "${variant} LoRA"
  echo "Final metrics:"
  grep "Average accuracy till task10" "${log_path}" | tail -n 1 || true
}

case "${MODE}" in
  check)
    echo "Repository: ${REPO_ROOT}"
    echo "Python: ${PYTHON_BIN}"
    echo "ImageNet-R root passed to the loader: ${IMR_DATA_PATH}"
    echo "Output root: ${OUTPUT_ROOT}"
    if require_checkpoints "${TII_DIR}" 10 "TII"; then
      echo "Ready for baseline, quick, or improved LoRA runs."
    else
      echo "Dataset is ready; train TII first with '$0 tii'."
    fi
    ;;
  tii)
    run_tii
    ;;
  baseline)
    run_lora baseline
    ;;
  quick)
    run_lora quick
    ;;
  improved)
    run_lora improved
    ;;
  *)
    echo "Usage: $0 [check|tii|baseline|quick|improved]" >&2
    exit 64
    ;;
esac
