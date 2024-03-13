# ProtCNN: Protein Family Classification

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1-red.svg)](https://pytorch.org/)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch_Lightning-2.2.1-brightgreen.svg)](https://www.pytorchlightning.ai/)
[![Optuna](https://img.shields.io/badge/Optuna-3.5.0-blueviolet.svg)](https://optuna.org/)
[![Click](https://img.shields.io/badge/Click-8.1.7-9cf.svg)](https://click.palletsprojects.com/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2.1-blue.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26.4-blue.svg)](https://numpy.org/)

## Project Overview

ProtCNN is a cutting-edge machine learning project focused on classifying protein sequences into Pfam families. It is inspired by the innovative ProtCNN model and utilizes the latest advancements in deep learning frameworks, namely PyTorch and PyTorch Lightning. This project not only demonstrates an application in bioinformatics but also emphasizes best practices in software engineering for machine learning, such as modular design, reproducibility, and CLI-based parameter management.

## Structure

- `data/`: Core data processing modules. Includes data loaders and utilities for handling the PFAM dataset.
- `docs/`: Contains markdown files (`cli_evaluation.md`, `cli_hyperparameters.md`, `cli_train.md`) that provide detailed documentation on how to use the command line interfaces for various scripts.
- `models/`: Implementation of the ProtCNN model, along with utility functions for model manipulation and inspection.
- `scripts/`: Collection of Python scripts (`evaluate.py`, `hyperparameters.py`, `train.py`) which are the main interface for interacting with the model for tasks like training, evaluation, and hyperparameter tuning.
- `tests/`: A suite of unit tests ensuring the reliability and correctness of the data processing modules, model implementation, and overall functionality.
- `Dockerfile`: Configuration file for Docker, which encapsulates the project in a containerized environment, ensuring consistency across different systems.
- `params.json`: Configuration file containing default parameters used across the project, aiding in reproducibility and flexibility.
- `poetry.lock` & `pyproject.toml`: Define the project dependencies and manage them through Poetry, enhancing the project's portability and dependency management.

## Key Libraries and Dependencies

This project leverages several powerful libraries and frameworks:
- **PyTorch & PyTorch Lightning**: For efficient and scalable deep learning model development.
- **Optuna**: For advanced hyperparameter optimization.
- **Click**: For creating user-friendly command-line interfaces.
- **Pandas & NumPy**: For data manipulation and numerical operations.
- **Torchmetrics**: For comprehensive model evaluation metrics.
- **Torchinfo**: For detailed model summaries.
- **Pytest**: For robust unit testing.

## Docker Usage

To encapsulate the project's environment and dependencies:

1. **Build the Docker Image**:
    ```shell
    docker build -t protcnn_project .
    ```

2. **Run a Docker Container in Interactive Mode**:
    ```shell
    docker run -it --rm protcnn_project /bin/bash
    ```
    This allows you to execute the training, evaluation, or hyperparameter tuning scripts in an isolated environment.

## Getting Started

1. **Clone the Repository**: Download the project files to get started.
2. **Build the Docker Image**: Follow the Docker usage instructions to create your container.
3. **Run the Docker Container**: Use the container environment for running the project's scripts.
