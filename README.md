# CSC8099 Thesis Codebase

Newcastle University CSC8099 thesis project — complete code and experiment outputs
Topic — training strategy study for ResNet-101 on CIFAR-100
Covers baseline runs, single-factor ablations, and best-recipe reproductions with multiple seeds

## Top-level layout

* `mixup` — experiments and results for MixUp
* `randaug` — experiments and results for RandAugment
* `warmup` — learning-rate warm-up experiments
* `wd` — weight decay experiments
* `resnet101_baseline` — baseline config and outputs
* `resnet101_baseline_differentseed` — baseline reproduced with different seeds
* `resnet101_best_differentseed` — best recipe reproduced with different seeds
* `train_cifar100.py` — single run training script
* `train_cifar100_30_60_100.py` — early-stopping workflow with 30 60 100 schedule

## Inside an experiment folder

* `config.yaml` — final run configuration snapshot
* `run_cfg.json` — script level arguments and metadata
* `env.yml` — conda environment export
* `requirements.txt` — pip dependencies
* `metrics_YYYYMMDD_HHMMSS.csv` — per-epoch log with train\_loss, val\_loss, train\_acc, val\_acc, val\_ece, learning rate and more
* `summary_YYYYMMDD_HHMMSS.csv` — key metrics summary
* `curve_YYYYMMDD_HHMMSS.png` — training curves

  * train\_loss across epochs
  * val\_acc across epochs
* `tb` — TensorBoard logs

## Quick start

```bash
# create environment with conda
conda env create -f env.yml
conda activate csc8099

# run a training job
python train_cifar100.py

# or use the 30 60 100 early-stopping workflow
python train_cifar100_30_60_100.py
```

## Data and model

* Dataset — CIFAR-100 auto-downloaded by the script
* Backbone — ResNet-101
* Techniques — MixUp, RandAugment, warm-up, weight decay, early stopping

## Reproducibility

* Every run stores configs, logs, and summaries
* To reproduce a result, reuse the `config.yaml` inside the target folder
* For analysis, read the per-epoch CSV in `metrics_*.csv` and the final summary in `summary_*.csv`


