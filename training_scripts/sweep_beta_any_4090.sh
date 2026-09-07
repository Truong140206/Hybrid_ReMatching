#!/usr/bin/env bash
set -uo pipefail

# Sweep the class-fusion weight beta over datasets, backbones and gate modes.
#
# The gate is currently defended on ImageNet-R alone, and only there has any
# arm been swept. On the other three benchmarks the ungated arm has a single
# measurement, at beta=0.5, which the ImageNet-R sweep shows is not the value
# it wants: ungated peaks at 0.3 on five of six backbones and loses several
# points by 0.5. Scoring one arm at its tuned setting against another at an
# untuned one is the error tools/beta_sweep_table.py exists to prevent, so the
# missing settings get measured rather than argued about.
#
# A dataset or backbone whose checkpoints are absent is skipped, not fatal, and
# the run refuses to start at all if another process is holding the GPU.
#
# Usage:
#   bash training_scripts/sweep_beta_any_4090.sh
#   BETAS="0.3 0.5" GATES="none relative" bash training_scripts/...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
SEED="${SEED:-42}"
BETAS="${BETAS:-0.3}"
GATES="${GATES:-none relative}"
FUSE_W="${FUSE_W:-0.7}"
MIN_FREE_MIB="${MIN_FREE_MIB:-8000}"

# prefix | dataset string | config module | tasks
ENTRIES_DEFAULT=(
  "cifar100|Split-CIFAR100|cifar100_lora|10"
  "ima|Split-Imagenet-A|ima_lora|10"
  "fivedatasets|5-datasets|five_datasets_lora|5"
)

# directory tag | timm backbone. ImageNet-R was trained before the directory
# convention settled and uses `mocov3` where these three use `moco1k`.
BACKBONES=(
  "|vit_base_patch16_224"
  "_moco1k|vit_base_patch16_224_mocov3"
  "_ibot1k|vit_base_patch16_224_ibot"
  "_ibot21k|vit_base_patch16_224_21k_ibot"
  "_dino|vit_base_patch16_224_dino"
  "_mae|vit_base_patch16_224_mae"
)

FREE="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
FREE="${FREE:-0}"
echo "GPU trong ${FREE}MiB"
if (( FREE < MIN_FREE_MIB )); then
  echo "Duoi nguong ${MIN_FREE_MIB}MiB nen dung lai truoc khi phong ca loat."
  echo "Lan truoc 18 luot chet lien tiep vi mot tien trinh khac giu 21GB."
  exit 1
fi

complete() {
  local dir="$1" tasks="$2" task
  for task in $(seq 1 "${tasks}"); do
    [[ -s "${dir}/checkpoint/task${task}_checkpoint.pth" ]] || return 1
  done
}

total=0; ran=0; skipped=0
for entry in "${ENTRIES[@]:-${ENTRIES_DEFAULT[@]}}"; do
  IFS='|' read -r prefix dataset config tasks <<< "${entry}"
  for pair in "${BACKBONES[@]}"; do
    IFS='|' read -r tag model <<< "${pair}"
    run_dir="${OUTPUT_ROOT}/${prefix}${tag}_lora_rank8_baseline_${tasks}tasks_seed${SEED}"
    tii_dir="${OUTPUT_ROOT}/${prefix}${tag}_tii_original_${tasks}tasks_seed${SEED}"
    if ! complete "${run_dir}" "${tasks}" || ! complete "${tii_dir}" "${tasks}"; then
      echo "--- bo qua ${prefix}${tag}: thieu checkpoint"
      skipped=$((skipped + 1))
      continue
    fi
    for gate in ${GATES}; do
      for beta in ${BETAS}; do
        total=$((total + 1))
        echo "##### $(date +%H:%M) ${prefix}${tag} | cong=${gate} | beta=${beta} #####"
        env DATASET="${dataset}" TII_DIR="${tii_dir}" CONFIG="${config}" SEED="${SEED}" NUM_TASKS="${tasks}" BACKBONE="${model}" RP_FUSE=1 RP_FUSE_DRM=1 RP_CLS_MIN=1 CALIBRATE=0 RP_FUSE_W="${FUSE_W}" RP_CLS_GATE="${gate}" RP_CLS_W="${beta}" bash "${SCRIPT_DIR}/eval_rp_head_any_4090.sh" "${run_dir}" && ran=$((ran + 1))
      done
    done
  done
done
echo "##### XONG: chay ${ran}/${total} luot, bo qua ${skipped} backbone #####"
