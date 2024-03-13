# CLI Documentation for `evaluate.py` (Evaluation)

## Overview
This CLI tool is designed for evaluating a pre-trained Protein Convolutional Neural Network (ProtCNN) using PyTorch Lightning. The script uses a checkpoint of a trained model and evaluates its performance on a specified dataset.

## CLI Options

### Evaluation Options
- `--checkpoint-path <path>`: Path to the trained model checkpoint.
- `--data-dir <path>`: Name of the directory in the root folder that contains the dataset.
- `--batch-size <int>`: (required field) Batch size for evaluation. This must be the batch size used for training.
- `--max-len <int>`:  (required field) Maximum sequence length for the model. This must be the maximum sequence length used for training.
- `--shuffle <bool>`: Whether to shuffle the dataset during evaluation.
- `--num-workers <int>`: Number of workers for data loading during evaluation.
- `--pin-memory <bool>`: Use pinned memory for data loading during evaluation.

### Example Usage
```bash
python evaluate.py --checkpoint-path /path/to/checkpoint.ckpt --batch-size 64
