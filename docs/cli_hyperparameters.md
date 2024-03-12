# CLI Documentation for `train.py` (Hyperparameter Tuning)

## Overview
This extended CLI tool, derived from the `train.py` script, is enhanced for hyperparameter tuning of a Protein Convolutional Neural Network (ProtCNN) using PyTorch Lightning and Optuna. It leverages a mix of default parameter values from `params.json` and user-specified options through command-line arguments.

## CLI Options (Hyperparameter Tuning Specific)

### Data Loader Options
- `--data-dir <path>`: Directory within the root folder containing the dataset. (Default: as specified in `params.json`)
- `--batch-size <int>`: Batch sizes to explore for training (accepts multiple values).
- `--max-len <int>`: Maximum sequence lengths to explore (accepts multiple values).
- `--shuffle <bool>`: Shuffle the dataset. (Default: as specified in `params.json`)
- `--num-workers <int>`: Number of workers for data loading. (Default: as specified in `params.json`)
- `--pin-memory <bool>`: Use pinned memory for data loading. (Default: as specified in `params.json`)

### Model Configuration Options
- `--num-classes <int>`: Number of classes. (Default: as specified in `params.json`)
- `--in-channels <int>`: Number of input channels. (Default: as specified in `params.json`)
- `--conv-channels <int>`: Convolution channels to explore (accepts multiple values).
- `--conv-kernel-size <int>`: Kernel size for convolution layers. (Default: as specified in `params.json`)
- `--conv-padding <int>`: Padding for convolution layers. (Default: as specified in `params.json`)
- `--bias <bool>`: Use bias in convolution layers. (Default: as specified in `params.json`)
- `--num-residual-blocks <int>`: Number of residual blocks to explore (accepts multiple values).
- `--residual-blocks-kernel-size <int>`: Kernel size for residual blocks. (Default: as specified in `params.json`)
- `--residual-blocks-bias <bool>`: Use bias in residual blocks. (Default: as specified in `params.json`)
- `--residual-blocks-dilation <int>`: Dilation for residual blocks. (Default: as specified in `params.json`)
- `--residual-blocks-padding <int>`: Padding for residual blocks. (Default: as specified in `params.json`)
- `--pool-kernel-size <int>`: Kernel size for pooling layers. (Default: as specified in `params.json`)
- `--pool-stride <int>`: Stride for pooling layers. (Default: as specified in `params.json`)
- `--pool-padding <int>`: Padding for pooling layers. (Default: as specified in `params.json`)

### Training Options
- `--optim <str>`: Optimizers to explore (accepts multiple values).
- `--lr <float>`: Learning rates to explore (accepts multiple values).
- `--weight-decay <float>`: Weight decay rate. (Default: as specified in `params.json`)
- `--scheduler-milestones <int>`: Scheduler milestones (accepts multiple values).
- `--scheduler-gamma <float>`: Scheduler gamma. (Default: as specified in `params.json`)
- `--accelerator <str>`: Type of accelerator. (Default: as specified in `params.json`)
- `--max-epochs <int>`: Maximum number of epochs. (Default: as specified in `params.json`)
- `--devices <int>`: Number of devices. (Default: as specified in `params.json`)
- `--precision <str>`: Precision for training. (Default: as specified in `params.json`)

### Logging, Callbacks, and Hyperparameter Tuning Options
- `--save-dir <path>`: Directory within the root folder for saving experiment outputs. (Default: as specified in `params.json`)
- `--name <str>`: Subdirectory name within save-dir for storing specific experiment results. (Default: as specified in `params.json`)
- `--save-top-k <int>`: Number of best models to save. (Default: as specified in `params.json`)
- `--monitor <str>`: Metric to monitor for model saving. (Default: as specified in `params.json`)
- `--mode <str>`: Mode for model saving (max/min). (Default: as specified in `params.json`)
- `--metric <str>`: Metric used for model objective (e.g., `test_acc`, `val_acc`).
- `--direction <str>`: Mode for model objective (max/min).
- `--n_trials <int>`: Number of trials for the hyperparameter tuning study.

## Example Usage
```bash
python train.py --data-dir ./data --batch-size 32 --batch-size 64 --max-epochs 20 --lr 0.001 --lr 0.0005 --save-dir ./model_saves
