# CLI Documentation for `train.py` (Training)

## Overview
This CLI tool, part of the `train.py` script, is designed for training a Protein Convolutional Neural Network (ProtCNN) using PyTorch Lightning. Default parameter values are loaded from `params.json`, but can be overridden using the command-line options described below.

## CLI Options

### Data Loader Options
- `--data-dir <path>`: Name of the directory in the root folder that contains the dataset.
- `--batch-size <int>`: Batch size for training.
- `--max-len <int>`: Maximum sequence length.
- `--shuffle <bool>`: Whether to shuffle the dataset.
- `--num-workers <int>`: Number of workers for data loading.
- `--pin-memory <bool>`: Use pinned memory for data loading.

### Model Configuration Options
- `--num-classes <int>`: Number of classes.
- `--in-channels <int>`: Number of input channels.
- `--conv-channels <int>`: Number of convolution channels.
- `--conv-kernel-size <int>`: Kernel size for convolution layers.
- `--conv-padding <int>`: Padding for convolution layers.
- `--bias <bool>`: Use bias in convolution layers.
- `--num-residual-blocks <int>`: Number of residual blocks.
- `--residual-blocks-kernel-size <int>`: Kernel size for residual blocks.
- `--residual-blocks-bias <bool>`: Use bias in residual blocks.
- `--residual-blocks-dilation <int>`: Dilation for residual blocks.
- `--residual-blocks-padding <int>`: Padding for residual blocks.
- `--pool-kernel-size <int>`: Kernel size for pooling layers.
- `--pool-stride <int>`: Stride for pooling layers.
- `--pool-padding <int>`: Padding for pooling layers.

### Training Options
- `--optim <str>`: Optimizer.
- `--lr <float>`: Learning rate.
- `--weight-decay <float>`: Weight decay rate.
- `--scheduler-milestones <int>`: Scheduler milestones (accepts multiple values).
- `--scheduler-gamma <float>`: Scheduler gamma.
- `--accelerator <str>`: Type of accelerator (e.g., GPU).
- `--max-epochs <int>`: Maximum number of epochs for training.
- `--devices <int>`: Number of devices to use for training.
- `--precision <str>`: Precision for training (e.g., 16-bit).

### Logging and Callbacks Options
- `--save-dir <path>`: Directory to save models and logs.
- `--name <str>`: Experiment name.
- `--save-top-k <int>`: Number of best models to save.
- `--monitor <str>`: Metric to monitor for model saving.
- `--mode <str>`: Mode for model saving (e.g., 'max' or 'min').

## Example Usage
```bash
python train.py --data-dir ./data --batch-size 64 --max-epochs 20 --lr 0.001 --save-dir ./model_saves
