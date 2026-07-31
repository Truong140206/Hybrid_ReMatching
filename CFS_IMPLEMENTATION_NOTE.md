# Note: Applying CFS to HRM-PET

## 1. Purpose

This note documents the changes made to integrate **CFS (Contrastive Feature Selection)** into the HRM-PET codebase.

The goal is to improve the quality of synthetic/replayed features used in the **CRCT / classifier alignment** stage. Instead of randomly sampling features from a Gaussian distribution, the modified version samples a larger candidate pool and uses a lightweight contrastive model to select more diverse and representative features.

In short:

```text
Original HRM-PET:
Gaussian feature distribution -> random feature sampling -> CRCT

Modified HRM-PET + CFS:
Gaussian feature distribution -> candidate sampling -> contrastive feature selection -> CRCT
```

## 2. Original HRM-PET Behavior

After each task is trained, HRM-PET extracts feature vectors from the model for every class in the current task.

For each class, the original code computes:

```text
cls_mean[class_id]
cls_cov[class_id]
```

These values describe a Gaussian feature distribution for each class:

```text
N(cls_mean, cls_cov)
```

During CRCT, the original code samples synthetic features directly from this Gaussian:

```python
m = MultivariateNormal(mean.float(), cov.float())
sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
```

These sampled features are then used to train the classifier alignment head through `fc_only=True`.

The limitation is that random Gaussian sampling may produce redundant or poorly distributed samples. It can miss diverse regions of the class feature distribution, especially when the number of sampled features is limited.

## 3. What CFS Adds

CFS adds a selection step after Gaussian candidate sampling.

For each class:

1. Real features are collected after task training.
2. A small contrastive MLP is trained on those class features.
3. During CRCT, instead of directly using random Gaussian samples, the code samples a larger candidate pool.
4. Candidate features are projected through the CFS MLP.
5. Features with lower similarity to already selected features are greedily selected.
6. The selected features are used for CRCT classifier alignment.

The new flow is:

```text
real class features
-> train CFS MLP
-> estimate Gaussian mean/covariance
-> sample candidate features from Gaussian
-> project candidates with CFS MLP
-> select diverse candidates
-> CRCT classifier alignment
```

## 4. Where CFS Is Applied

CFS is currently applied only to **CRCT feature sampling**.

It is not applied to:

- image datasets
- dataloaders
- backbone training
- CTIRD loss
- task inference
- LoRA parameter matching

The current integration is intentionally narrow. It modifies the quality of pseudo features used for classifier alignment while keeping the main HRM-PET pipeline unchanged.

## 5. Files Changed

### `utils.py`

New utilities were added:

```python
class CFSContrastiveMLP(torch.nn.Module)
def use_cfs_sampling(args)
def train_cfs_model(features, args, device)
def sample_cfs_features(mean, cov, num_samples, args, device, cfs_model=None)
```

Their roles:

- `CFSContrastiveMLP`: lightweight MLP used to project features into a contrastive space.
- `train_cfs_model`: trains one CFS model per class using real class features.
- `sample_cfs_features`: samples candidates from the Gaussian distribution and selects diverse samples.
- `use_cfs_sampling`: checks whether CFS is enabled by command-line arguments.

### `engines/hide_tii_engine.py`

This engine is used for the TII / original-model stage.

Changes:

1. Added a global dictionary:

```python
cls_cfs_model = dict()
```

2. After collecting real class features and gathering them across distributed processes, CFS trains one model per class:

```python
gathered_features_per_cls = torch.cat(features_per_cls_list, dim=0)
if utils.use_cfs_sampling(args):
    cls_cfs_model[cls_id] = utils.train_cfs_model(gathered_features_per_cls, args, device)
```

3. During CRCT sampling, the original random sampling was replaced with:

```python
sampled_data_single = utils.sample_cfs_features(
    mean.float(), cov.float(), num_sampled_pcls, args, device,
    cfs_model=cls_cfs_model.get(c_id))
```

If CFS is not enabled, or if no CFS model is available, the helper falls back to normal Gaussian sampling.

### `engines/hrm_lora_wtp_and_tap_engine.py`

This engine is used for the LoRA HRM-PET stage.

The same CFS integration was added here:

1. Store per-class CFS models.
2. Train CFS models after collecting real class features.
3. Use CFS-selected Gaussian features during CRCT.

This is the most important stage for the final HRM-PET result because LoRA training and final classifier alignment happen here.

### Config Files

CFS command-line arguments were added to the dataset config files:

- `configs/cifar100_hideprompt_5e.py`
- `configs/cifar100_lora.py`
- `configs/imr_hideprompt_5e.py`
- `configs/imr_lora.py`
- `configs/ima_hideprompt_5e.py`
- `configs/ima_lora.py`
- `configs/five_datasets_hideprompt_5e.py`
- `configs/five_datasets_lora.py`

New arguments:

```python
--cfs_sampling
--cfs_epochs
--cfs_lr
--cfs_momentum
--cfs_hidden_dim
--cfs_batch_size
--cfs_train_max_samples
--cfs_candidate_multiplier
--cfs_tau
```

### `training_scripts/kaggle_train_imr_lora_sup21k.sh`

The Kaggle script was updated so CFS can be enabled through environment variables:

```bash
export CFS_SAMPLING=1
export CFS_EPOCHS=20
export CFS_CANDIDATE_MULTIPLIER=2
```

The script creates `CFS_ARGS` and appends them to both training stages when enabled.

## 6. Detailed CFS Logic

### 6.1 Training the CFS MLP

For each class, real features are gathered:

```text
features_per_cls = [f1, f2, f3, ...]
```

The CFS MLP maps each feature into a normalized contrastive space:

```text
feature -> Linear -> LeakyReLU -> Linear -> normalize
```

The training objective discourages features from collapsing into the same direction. For each batch, cosine similarities are computed:

```python
sim = torch.mm(out, out.t()) / tau
sim.fill_diagonal_(float('-inf'))
loss = torch.logsumexp(sim, dim=1).mean()
```

Minimizing this loss encourages features to become more separated in the CFS space.

### 6.2 Selecting Gaussian Candidates

During CRCT, the original code needs `num_sampled_pcls` synthetic features for each class.

With CFS, the code first samples more candidates:

```text
candidate_count = num_sampled_pcls * cfs_candidate_multiplier
```

Example:

```text
num_sampled_pcls = 64
cfs_candidate_multiplier = 2
candidate_count = 128
```

Then all candidates are projected by the CFS MLP:

```python
embeddings = cfs_model(candidates.float())
```

A pairwise similarity matrix is computed:

```python
sim = torch.exp(torch.mm(embeddings, embeddings.t()) / tau)
```

The selection is greedy:

1. Start from one random candidate.
2. Keep track of average similarity to selected candidates.
3. Select the candidate with the lowest similarity.
4. Repeat until `num_sampled_pcls` features are selected.

This produces a more diverse synthetic feature set than direct random sampling.

## 7. Why This Is Different From the Original Version

The original version treats every Gaussian sample equally.

The modified version assumes that not every Gaussian sample is equally useful. Some samples are too similar or redundant, so CFS selects a subset that better covers the feature distribution.

Comparison:

```text
Original:
sample N features from Gaussian
use all N features

CFS version:
sample N * K candidate features from Gaussian
score diversity in contrastive space
select N diverse features
use selected N features
```

The number of features used by CRCT stays the same. Only the selection quality changes.

## 8. Safety and Backward Compatibility

CFS is disabled by default.

If `--cfs_sampling` is not provided, the behavior falls back to the original HRM-PET behavior:

```python
return distribution.sample(sample_shape=(num_samples,))
```

This means existing scripts can still run without CFS.

Fallback behavior:

- If CFS is disabled: use original Gaussian sampling.
- If a class has too few features: use original Gaussian sampling.
- If no CFS model exists for a class: use original Gaussian sampling.

This keeps the implementation safer for long-running experiments.

## 9. How to Run With CFS

### Direct command-line usage

Add:

```bash
--cfs_sampling
--cfs_epochs 20
--cfs_candidate_multiplier 2
```

Recommended initial values:

```bash
--cfs_sampling
--cfs_epochs 20
--cfs_candidate_multiplier 2
--cfs_train_max_samples 1024
--cfs_batch_size 128
```

### Kaggle script usage

Set environment variables before running:

```bash
export CFS_SAMPLING=1
export CFS_EPOCHS=20
export CFS_CANDIDATE_MULTIPLIER=2
export CFS_TRAIN_MAX_SAMPLES=1024
export CFS_BATCH_SIZE=128
```

### Colab usage

Use the same command-line flags, but keep:

```bash
export GPUS=1
```

because Colab normally provides only one GPU.

## 10. Experimental Results

### CIFAR100 10-task run with CFS

Configuration:

```text
TII_EPOCHS=5
LORA_EPOCHS=10
CRCT_EPOCHS=3
NUM_TASKS=10
CFS_EPOCHS=20
CFS_CANDIDATE_MULTIPLIER=2
CFS_TRAIN_MAX_SAMPLES=1024
CFS_BATCH_SIZE=128
```

Final result:

```text
[Average accuracy till task10]
Acc@task: 88.0600
Acc@1:    88.0000
Acc@5:    97.9400
Loss:     0.5232
Forgetting: 3.8889
Backward:  -3.7000
```

Checkpoints were successfully saved for both stages:

```text
cifar100_tii_cfs_10tasks_seed42/checkpoint/task1_checkpoint.pth
...
cifar100_tii_cfs_10tasks_seed42/checkpoint/task10_checkpoint.pth

cifar100_lora_cfs_10tasks_seed42/checkpoint/task1_checkpoint.pth
...
cifar100_lora_cfs_10tasks_seed42/checkpoint/task10_checkpoint.pth
```

### Comparison with previous non-CFS run

Previous CIFAR100 10-task run without CFS:

```text
Acc@task: 87.0500
Acc@1:    86.9000
Acc@5:    97.5900
Loss:     0.5860
```

CFS run:

```text
Acc@task: 88.0600
Acc@1:    88.0000
Acc@5:    97.9400
Loss:     0.5232
```

Observed improvement:

```text
Acc@task: +1.01
Acc@1:    +1.10
Acc@5:    +0.35
Loss:     -0.0628
```

This suggests that CFS-selected synthetic features improved CRCT classifier alignment in the tested CIFAR100 setting.

## 11. Interpretation

The result supports the motivation that the quality of replayed feature samples matters.

CRCT depends on synthetic features generated from class-wise distributions. If these features are sampled randomly, they may be redundant. CFS improves this by selecting a more diverse subset.

The improvement is not caused by changing the backbone, increasing data, or modifying the main classifier. It comes from improving the synthetic feature selection process used during classifier alignment.

## 12. Limitations

The current implementation applies CFS only to CRCT sampling.

It does not yet apply CFS to:

- CTIRD relation distillation
- semantic-aware projection
- task selection
- prompt matching
- LoRA parameter selection

Also, for `multi-centroid` mode, the current implementation uses one CFS model per class and applies it to samples from each centroid. This is reasonable, but the cleanest match to the original CFS idea is with:

```bash
--ca_storage_efficient_method covariance
```

because CFS was originally designed around class-wise Gaussian feature distributions.

## 13. Possible Future Work

Potential next steps:

1. Apply CFS-selected anchors to CTIRD.
2. Combine CFS with semantic-aware projection.
3. Train separate CFS models for each centroid in multi-centroid mode.
4. Tune `cfs_candidate_multiplier`.
5. Tune `cfs_epochs`.
6. Compare CFS vs non-CFS across multiple seeds.
7. Test on ImageNet-R and ImageNet-A.

## 14. Presentation Summary

The modification can be summarized as:

```text
We improve HRM-PET by integrating CFS into the CRCT feature replay stage.
Instead of using random Gaussian samples for classifier alignment, we train a lightweight per-class contrastive MLP and use it to select diverse synthetic features from a larger Gaussian candidate pool.
This keeps the original HRM-PET pipeline intact while improving the quality of replayed feature samples.
On CIFAR100 10 tasks, the CFS version improves Acc@1 from 86.90 to 88.00 and reduces loss from 0.5860 to 0.5232.
```

