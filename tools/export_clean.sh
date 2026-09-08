#!/usr/bin/env bash
set -euo pipefail

# Build a clean export of this repository for a public release.
#
# It copies an explicit keep-list into a fresh directory rather than deleting
# from the working tree, for two reasons. Deleting is not reversible in the
# moment it matters, and the release should be a thing we assembled on purpose
# rather than what happens to be left after a series of removals.
#
# Every engine under engines/ is kept even though most belong to directions that
# were abandoned: main.py reaches all twenty-six through the imports in
# trainers/lora_trainer.py and engines/hide_tii_engine.py, so removing any of
# them is code surgery, not file deletion, and belongs in a separate step with
# an evaluation run to confirm the numbers are unchanged.
#
# Usage: bash export_clean.sh SOURCE_DIR TARGET_DIR

SOURCE="${1:?dua duong dan repo goc}"
TARGET="${2:?dua duong dan thu muc dich}"

if [[ -e "${TARGET}" ]]; then
  echo "Thu muc dich da ton tai: ${TARGET}" >&2
  echo "Xoa no hoac chon ten khac; script khong ghi de." >&2
  exit 1
fi

KEEP=(
  # Diem vao va ha tang chung
  main.py
  utils.py
  datasets.py
  protocols.py
  attention.py
  requirements.txt
  README.md
  .gitignore

  # Cau hinh: bon bo du lieu, moi bo mot cap TII va LoRA
  configs/cifar100_hideprompt_5e.py
  configs/cifar100_lora.py
  configs/five_datasets_hideprompt_5e.py
  configs/five_datasets_lora.py
  configs/ima_hideprompt_5e.py
  configs/ima_lora.py
  configs/imr_hideprompt_5e.py
  configs/imr_lora.py
  configs/imagenet_class_names.json

  # Nap du lieu
  continual_datasets/continual_datasets.py

  # Bo thich ung
  peft/__init__.py
  peft/lora/__init__.py
  peft/lora/continual_lora.py
  peft/lora/hide_lora.py
  peft/lora/momentum_lora.py
  peft/prompt/__init__.py
  peft/prompt/hide_prompt.py

  # Mang xuong song
  vits/base.py
  vits/hide_prompt_vision_transformer.py
  vits/hrm_lora_vision_transformer.py

  # Huan luyen
  trainers/lora_trainer.py
  trainers/tii_trainer.py

  # Script chay
  training_scripts/train_any_4090.sh
  training_scripts/eval_rp_head_any_4090.sh
  training_scripts/eval_imagenet_r_conventional_4090.sh
  training_scripts/eval_cifar100_conventional_4090.sh
  training_scripts/eval_cub200_conventional_4090.sh
  training_scripts/rollout_hybrid_4090.sh

  # Cong cu doc ket qua va dung du lieu
  tools/collect_results.py
  tools/beta_sweep_table.py
  tools/prepare_datasets.py
  tools/imagenet_folder_from_parquet.py
  tools/audit_exemplar_free_checkpoint.py

  # Kiem thu: phuong phap, va tuyen bo khong dung mau cu
  tests/test_rp_head_and_fusion.py
  tests/test_exemplar_free_protocol.py

  # Bai bao
  reports/lncs_method_en.tex
  reports/lncs_phuong_phap.tex
)

mkdir -p "${TARGET}"

missing=0
for item in "${KEEP[@]}"; do
  if [[ ! -e "${SOURCE}/${item}" ]]; then
    echo "THIEU o ban goc: ${item}" >&2
    missing=$((missing + 1))
    continue
  fi
  mkdir -p "${TARGET}/$(dirname "${item}")"
  cp "${SOURCE}/${item}" "${TARGET}/${item}"
done

# engines/ giu nguyen ca thu muc: moi tep deu nam trong bao dong nhap cua main.py
mkdir -p "${TARGET}/engines"
cp "${SOURCE}"/engines/*.py "${TARGET}/engines/"

copied=$(find "${TARGET}" -type f | wc -l)
total=$(find "${SOURCE}" -type f -not -path '*/.git/*' -not -path '*__pycache__*' | wc -l)
echo
echo "Da chep ${copied} tep sang ${TARGET}"
echo "Ban goc co ${total} tep (khong ke .git va __pycache__)"
if (( missing > 0 )); then
  echo "CANH BAO: ${missing} muc trong danh sach giu khong tim thay o ban goc." >&2
fi
echo
echo "Buoc kiem bat buoc truoc khi day len repo moi:"
echo "  cd ${TARGET} && python -c \"import ast,sys,os"
echo "  [ast.parse(open(os.path.join(r,f),encoding='utf-8').read())"
echo "   for r,_,fs in os.walk('.') for f in fs if f.endswith('.py')]\""
echo "  roi chay mot luot danh gia va doi chieu Acc@1 voi ban goc."
