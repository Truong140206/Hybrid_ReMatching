#!/usr/bin/env bash
set -uo pipefail

# Sweep the routing-free RP head on CUB. Each run is ~90s, so a wide lambda
# sweep is cheap. Targets: beat conventional HRM-PET 86.53 Acc@1; published
# RanPAC reference on CUB B0I10 is ~90.3.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${1:-}"
if [[ -z "${RUN_DIR}" ]]; then echo "Usage: $0 RUN_DIR" >&2; exit 64; fi

run_one() {
  local dim="$1" lam="$2" norm="$3" act="$4"
  echo "=== dim=${dim} lambda=${lam} norm=${norm} act=${act} ==="
  RP_DIM="${dim}" RP_LAMBDA="${lam}" RP_NORM="${norm}" RP_ACT="${act}" \
    bash "${SCRIPT_DIR}/eval_cub200_rp_head_4090.sh" "${RUN_DIR}" 2>&1 \
    | grep -E "Average accuracy till task10|RP_HEAD_GATE|Refusing" | tail -n 2
}

# Lambda is the dominant knob for ridge; scan several decades.
for lam in 100 1000 10000 100000 1000000; do
  run_one 5000 "${lam}" none relu
done

# Normalized features change the effective scale, so rescan lambda there.
for lam in 0.01 0.1 1 10 100; do
  run_one 5000 "${lam}" l2 relu
done

# Best-guess wider projection at the published width.
run_one 10000 10000 none relu
run_one 10000 1 l2 relu

echo "=== sweep done ==="
