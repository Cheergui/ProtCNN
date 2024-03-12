# CLI Documentation for `hyperparameters.py` (Hyperparameter Tuning)

## Overview
This extended CLI tool, derived from the `train.py` script, is enhanced for hyperparameter tuning of a Protein Convolutional Neural Network (ProtCNN) using PyTorch Lightning and Optuna. Default parameter values are loaded from `params.json`, but can be overridden using the command-line options described below. Only the new commands are presented in this document since the other commands are similar to the `train.py` script.

## CLI Options (Hyperparameter Tuning Specific)

### Hyperparameter Options
- `--batch-size <int>`: Batch sizes to explore for training (accepts multiple values).
- `--max-len <int>`: Maximum sequence lengths to explore (accepts multiple values).
- `--conv-channels <int>`: Convolution channels to explore (accepts multiple values).
- `--num-residual-blocks <int>`: Number of residual blocks to explore (accepts multiple values).
- `--optim <str>`: Optimizers to explore (accepts multiple values).
- `--lr <float>`: Learning rates to explore (accepts multiple values).
- `--metric <str>`: Metric used for model objective (e.g., `test_acc`, `val_acc`).
- `--direction <str>`: Mode for model objective (max/min).
- `--n_trials <int>`: Number of trials for the hyperparameter tuning study.

## Example Usage
```bash
python hyperparameters.py --data-dir ./data --batch-size [32, 64, 128, 256] --max-epochs 20 --lr [0.0001, 0.001, 0.01]
