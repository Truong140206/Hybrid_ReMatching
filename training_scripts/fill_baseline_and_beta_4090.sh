#!/usr/bin/env bash
set -uo pipefail

# Fill the two gaps the gate decision left open, in one overnight pass.
#
# Phase B, the conventional HRM-PET baseline on every cell. The claim that the
# hybrid both learns each task better and retains more of it currently rests on
# two cells, because that is where a baseline log exists. Nineteen cells have
# none, so the strongest statement in the paper is the least measured one.
#
# Phase A, the ungated beta curve on ImageNet-A and 5-Datasets. beta=0.3 was
# chosen on ImageNet-R alone; asserting it elsewhere is the error this repo
# spent a day catching in its own ablation.
#
# B runs first even though A is the one the ablation cannot go to press without:
# B uses a script that has never run, and a new script should fail in the first
# ten minutes rather than after A has held the GPU for nine hours.
#
# Cells whose checkpoints are absent are skipped, and so is any evaluation whose
# log already exists, so the script can be re-run after an interruption without
# repeating finished work.
#
# Usage: bash training_scripts/fill_baseline_and_beta_4090.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
OUT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
SEED="${SEED:-42}"
BETAS="${BETAS:-0.1 0.2 0.5}"
MIN_FREE_MIB="${MIN_FREE_MIB:-8000}"

# ImageNet-R was trained before the directory convention settled, so MoCo v3
# sits under `mocov3` there and `moco1k` elsewhere, and it is the only benchmark
# with an MAE run.
BB_IMR="|vit_base_patch16_224 _mocov3|vit_base_patch16_224_mocov3 _ibot1k|vit_base_patch16_224_ibot _ibot21k|vit_base_patch16_224_21k_ibot _dino|vit_base_patch16_224_dino _mae|vit_base_patch16_224_mae"
BB_OTH="|vit_base_patch16_224 _moco1k|vit_base_patch16_224_mocov3 _ibot1k|vit_base_patch16_224_ibot _ibot21k|vit_base_patch16_224_21k_ibot _dino|vit_base_patch16_224_dino"

D_IMR="imr|Split-Imagenet-R|imr_lora|10"
D_CIF="cifar100|Split-CIFAR100|cifar100_lora|10"
D_IMA="ima|Split-Imagenet-A|ima_lora|10"
D_FIV="fivedatasets|5-datasets|five_datasets_lora|5"

FREE="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
FREE="${FREE:-0}"
echo "GPU trong ${FREE}MiB"
if (( FREE < MIN_FREE_MIB )); then
  echo "Duoi nguong ${MIN_FREE_MIB}MiB nen dung lai truoc khi phong ca loat."
  exit 1
fi

ran=0; skipped=0

ready() {  # run_dir tii_dir tasks
  local run="$1" tii="$2" tasks="$3" task
  for task in $(seq 1 "${tasks}"); do
    [[ -s "${run}/checkpoint/task${task}_checkpoint.pth" ]] || return 1
    [[ -s "${tii}/checkpoint/task${task}_checkpoint.pth" ]] || return 1
  done
}

baseline() {  # prefix dataset config tasks dir_tag backbone
  local prefix="$1" dataset="$2" config="$3" tasks="$4" tag="$5" model="$6"
  local base="${prefix}${tag}_lora_rank8_baseline_${tasks}tasks_seed${SEED}"
  local run="${OUT}/${base}" tii="${OUT}/${prefix}${tag}_tii_original_${tasks}tasks_seed${SEED}"
  if ! ready "${run}" "${tii}" "${tasks}"; then
    echo "--- bo qua moc ${prefix}${tag}: thieu checkpoint"; skipped=$((skipped + 1)); return
  fi
  if [[ -s "${OUT}/${base}_eval_conventional.log" ]]; then
    echo "--- da co moc ${prefix}${tag}"; return
  fi
  echo "##### $(date +%H:%M) MOC ${prefix}${tag} #####"
  env DATASET="${dataset}" CONFIG="${config}" NUM_TASKS="${tasks}" \
      BACKBONE="${model}" TII_DIR="${tii}" SEED="${SEED}" \
      bash "${SCRIPT_DIR}/eval_conventional_any_4090.sh" "${run}" && ran=$((ran + 1))
}

sweep() {  # prefix dataset config tasks dir_tag backbone
  local prefix="$1" dataset="$2" config="$3" tasks="$4" tag="$5" model="$6" beta
  local base="${prefix}${tag}_lora_rank8_baseline_${tasks}tasks_seed${SEED}"
  local run="${OUT}/${base}" tii="${OUT}/${prefix}${tag}_tii_original_${tasks}tasks_seed${SEED}"
  if ! ready "${run}" "${tii}" "${tasks}"; then
    echo "--- bo qua quet ${prefix}${tag}: thieu checkpoint"; skipped=$((skipped + 1)); return
  fi
  for beta in ${BETAS}; do
    echo "##### $(date +%H:%M) QUET ${prefix}${tag} beta=${beta} #####"
    env DATASET="${dataset}" TII_DIR="${tii}" CONFIG="${config}" SEED="${SEED}" \
        NUM_TASKS="${tasks}" BACKBONE="${model}" RP_FUSE=1 RP_FUSE_DRM=1 \
        RP_CLS_MIN=1 CALIBRATE=0 RP_FUSE_W=0.7 RP_CLS_GATE=none RP_CLS_W="${beta}" \
        bash "${SCRIPT_DIR}/eval_rp_head_any_4090.sh" "${run}" && ran=$((ran + 1))
  done
}

over() {  # phase entry backbone_list
  local phase="$1" entry="$2" list="$3" pair prefix dataset config tasks tag model
  IFS='|' read -r prefix dataset config tasks <<< "${entry}"
  for pair in ${list}; do
    IFS='|' read -r tag model <<< "${pair}"
    "${phase}" "${prefix}" "${dataset}" "${config}" "${tasks}" "${tag}" "${model}"
  done
}

echo "===================== PHAN B: moc HRM-PET ====================="
over baseline "${D_IMR}" "${BB_IMR}"
over baseline "${D_CIF}" "${BB_OTH}"
over baseline "${D_IMA}" "${BB_OTH}"
over baseline "${D_FIV}" "${BB_OTH}"

echo "===================== PHAN A: quet beta ======================="
over sweep "${D_IMA}" "${BB_OTH}"
over sweep "${D_FIV}" "${BB_OTH}"

echo "##### XONG: chay ${ran} luot, bo qua ${skipped} o #####"
