#!/usr/bin/env bash
set -uo pipefail

# Block until the GPU can take a run: enough free memory and nothing of ours
# already training or evaluating on it.
#
# The pattern lives in this file rather than on a caller's command line on
# purpose: pgrep -f matches whole command lines, so an inline `pgrep -f "..."`
# inside a `bash -c` string finds that very string and waits on itself forever.
#
# Usage: bash training_scripts/wait_for_gpu.sh [FREE_MIB] [POLL_SECONDS]

NEED="${1:-12000}"
POLL="${2:-120}"

while :; do
  free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1)"
  busy="$(pgrep -f 'torch.distributed.run' 2>/dev/null | tr '\n' ' ')"
  if [[ "${free}" =~ ^[0-9]+$ ]] && (( free >= NEED )) && [[ -z "${busy// /}" ]]; then
    echo "GPU ready: ${free} MiB free, no run in flight"
    break
  fi
  echo "[$(date +%H:%M:%S)] waiting: free=${free:-?} MiB (need ${NEED}) busy_pids=[${busy}]"
  sleep "${POLL}"
done
