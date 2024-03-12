import pytest
from data.dataloader import SequenceDataset
from pathlib import Path
import json
from types import SimpleNamespace

@pytest.fixture
def load_params():
    root = Path(__file__).parent.parent
    json_path = root / 'params.json'
    with open(json_path, 'r') as file:
        params = json.load(file, object_hook=lambda d: SimpleNamespace(**d))
    return params

def test_dataset_initialization(load_params):
    dataloader_params = load_params.dataloader
    root = Path(__file__).parent.parent
    data_path = root / dataloader_params.data_dir
    max_len = dataloader_params.max_len
    dataset = SequenceDataset(max_len=max_len, data_path=data_path, split='train')
    
    assert len(dataset) > 0, "Dataset is empty"

def test_get_item(load_params):
    dataloader_params = load_params.dataloader
    model_params = load_params.model
    root = Path(__file__).parent.parent
    data_path = root / dataloader_params.data_dir
    max_len = dataloader_params.max_len
    in_channels = model_params.in_channels
    dataset = SequenceDataset(max_len=max_len, data_path=data_path, split='train')
    
    first_item = dataset[0]
    assert first_item['sequence'].shape == (in_channels, max_len), (
        f"Sequence shape is incorrect: Expected ({in_channels}, {max_len}), got {first_item['sequence'].shape}"
    )
    assert isinstance(first_item['target'], (int, float)), (
        "Target value type is incorrect: Expected a scalar (int or float)"
    )

print("All tests for dataloader passed.")
