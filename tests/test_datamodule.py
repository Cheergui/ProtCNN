import pytest
import json
from types import SimpleNamespace
from pathlib import Path
from data.datamodule import DataModule

@pytest.fixture()
def datamodule_params():
    root = Path(__file__).parent.parent
    json_path = root / 'params.json'
    with open(json_path, 'r') as file:
        params = json.load(file, object_hook=lambda d: SimpleNamespace(**d))
    return params.dataloader

def test_datamodule_init(datamodule_params):
    root = Path(__file__).parent.parent
    data_dir = root / datamodule_params.data_dir
    datamodule = DataModule(data_dir=data_dir,
                            batch_size=datamodule_params.batch_size,
                            max_len=datamodule_params.max_len,
                            shuffle=datamodule_params.shuffle,
                            num_workers=datamodule_params.num_workers,
                            pin_memory=datamodule_params.pin_memory)
    assert datamodule.batch_size == datamodule_params.batch_size, (
        f"Batch size is incorrect: Expected {datamodule_params.batch_size}, got {datamodule.batch_size}"
    )
    assert datamodule.max_len == datamodule_params.max_len, (
        f"Max length is incorrect: Expected {datamodule_params.max_len}, got {datamodule.max_len}"
    )
    assert datamodule.shuffle == datamodule_params.shuffle, (
        f"Shuffle parameter is incorrect: Expected {datamodule_params.shuffle}, got {datamodule.shuffle}"
    )
    assert datamodule.num_workers == datamodule_params.num_workers, (
        f"Number of workers is incorrect: Expected {datamodule_params.num_workers}, got {datamodule.num_workers}"
    )
    assert datamodule.pin_memory == datamodule_params.pin_memory, (
        f"Pin memory setting is incorrect: Expected {datamodule_params.pin_memory}, got {datamodule.pin_memory}"
    )
    assert str(datamodule.data_dir) == str(data_dir), (
        f"Data directory is incorrect: Expected {data_dir}, got {datamodule.data_dir}"
    )

def test_train_dataloader(datamodule_params):
    root = Path(__file__).parent.parent
    data_dir = root / datamodule_params.data_dir
    datamodule = DataModule(data_dir=data_dir,
                            batch_size=datamodule_params.batch_size,
                            max_len=datamodule_params.max_len,
                            shuffle=datamodule_params.shuffle,
                            num_workers=datamodule_params.num_workers,
                            pin_memory=datamodule_params.pin_memory)
    datamodule.setup('fit')

    train_dataloader = datamodule.train_dataloader()
    train_batch = next(iter(train_dataloader))
    assert train_batch is not None, "Train dataloader unexpectedly returned None."

    X_train_batch, y_train_batch = train_batch['sequence'], train_batch['target']
    assert len(X_train_batch) == datamodule_params.batch_size, (
        f"Train dataloader sequence batch size is incorrect: Expected {datamodule_params.batch_size}, got {len(X_train_batch)}"
    )
    assert len(y_train_batch) == datamodule_params.batch_size, (
        f"Train dataloader target batch size is incorrect: Expected {datamodule_params.batch_size}, got {len(y_train_batch)}"
    )


def test_val_dataloader(datamodule_params):
    root = Path(__file__).parent.parent
    data_dir = root / datamodule_params.data_dir
    datamodule = DataModule(data_dir=data_dir,
                            batch_size=datamodule_params.batch_size,
                            max_len=datamodule_params.max_len,
                            shuffle=datamodule_params.shuffle,
                            num_workers=datamodule_params.num_workers,
                            pin_memory=datamodule_params.pin_memory)
    datamodule.setup('fit')

    val_dataloader = datamodule.val_dataloader()
    val_batch = next(iter(val_dataloader))
    assert val_batch is not None, "Validation dataloader unexpectedly returned None."

    X_val_batch, y_val_batch = val_batch['sequence'], val_batch['target']
    assert len(X_val_batch) == datamodule_params.batch_size, (
        f"Validation dataloader sequence batch size is incorrect: Expected {datamodule_params.batch_size}, got {len(X_val_batch)}"
    )
    assert len(y_val_batch) == datamodule_params.batch_size, (
        f"Validation dataloader target batch size is incorrect: Expected {datamodule_params.batch_size}, got {len(y_val_batch)}"
    )


def test_test_dataloader(datamodule_params):
    root = Path(__file__).parent.parent
    data_dir = root / datamodule_params.data_dir
    datamodule = DataModule(data_dir=data_dir,
                            batch_size=datamodule_params.batch_size,
                            max_len=datamodule_params.max_len,
                            shuffle=datamodule_params.shuffle,
                            num_workers=datamodule_params.num_workers,
                            pin_memory=datamodule_params.pin_memory)
    datamodule.setup('test')

    test_dataloader = datamodule.test_dataloader()
    test_batch = next(iter(test_dataloader))
    assert test_batch is not None, "Test dataloader unexpectedly returned None."

    X_test_batch, y_test_batch = test_batch['sequence'], test_batch['target']
    assert len(X_test_batch) == datamodule_params.batch_size, (
        f"Test dataloader sequence batch size is incorrect: Expected {datamodule_params.batch_size}, got {len(X_test_batch)}"
    )
    assert len(y_test_batch) == datamodule_params.batch_size, (
        f"Test dataloader target batch size is incorrect: Expected {datamodule_params.batch_size}, got {len(y_test_batch)}"
    )

    
print("All tests for datamodule passed.")

