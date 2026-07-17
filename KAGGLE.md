# Running HRM-PET on Kaggle

This repository can run in a Kaggle notebook with a GPU accelerator. The main Kaggle differences are:

- put datasets under `/kaggle/input`
- write checkpoints and logs under `/kaggle/working`
- install the pinned `timm` version used by the model code
- use `torch.distributed.run` instead of the deprecated `torch.distributed.launch`

## 1. Add the code and data

Add this repository to the notebook, or upload it as a Kaggle dataset. Also add the ImageNet-R dataset. The runner expects this layout by default:

```text
/kaggle/input/imagenet-r/imagenet-r/<class folders>
```

If your Kaggle dataset has a different folder name, set `DATA_PATH` before running the script.

## 2. Install dependencies

Run this in the first notebook cell:

```bash
pip install -r /kaggle/input/<repo-dataset-name>/requirements.txt
```

If you cloned the repo into `/kaggle/working/HRM-PET`, use:

```bash
pip install -r /kaggle/working/HRM-PET/requirements.txt
```

## 3. Run the ImageNet-R LoRA experiment

From the repository directory:

```bash
bash training_scripts/kaggle_train_imr_lora_sup21k.sh
```

For a smoke test, reduce the epochs:

```bash
TII_EPOCHS=1 LORA_EPOCHS=1 bash training_scripts/kaggle_train_imr_lora_sup21k.sh
```

For a different dataset mount:

```bash
DATA_PATH=/kaggle/input/<your-imagenet-r-dataset> bash training_scripts/kaggle_train_imr_lora_sup21k.sh
```

Outputs are written to:

```text
/kaggle/working/hrm-pet-output
```

## Notes

- Internet must be enabled if pretrained `timm` weights or torchvision datasets need to be downloaded.
- If internet is disabled, attach a Kaggle dataset containing the required cached weights and datasets.
- A single Kaggle GPU should use `GPUS=1`, which is the default.
