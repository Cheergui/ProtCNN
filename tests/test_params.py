import json
from pathlib import Path
from types import SimpleNamespace

def test_params_json():
    root = Path(__file__).parent.parent
    json_path = root / 'params.json'

    

    assert json_path.exists(), "params.json file does not exist"

    with open(json_path, 'r') as file:
        params = json.load(file, object_hook=lambda d: SimpleNamespace(**d))

    expected_fields = {
        "dataloader": ["data_dir", "batch_size", "shuffle", "num_workers", "pin_memory", "max_len"],
        "model": ["num_classes", "in_channels", "conv_channels", "conv_kernel_size", "conv_padding",
                  "bias", "num_residual_blocks", "residual_blocks_kernel_size", "residual_blocks_bias",
                  "residual_blocks_dilation", "residual_blocks_padding", "pool_kernel_size", "pool_stride",
                  "pool_padding", "optim", "lr", "weight_decay", "scheduler_milestones", "scheduler_gamma"],
        "trainer": ["accelerator", "max_epochs", "devices", "precision", "logger", "callbacks"],
        "evaluation": ["checkpoint_path"],
        "hyperparameter_tuning": ["max_len", "batch_size", "conv_channels","num_residual_blocks",
                                  "lr", "optim", "metric", "direction", "n_trials"]
    }

    for section, fields in expected_fields.items():
        assert hasattr(params, section), f"{section} section is missing"
        section_obj = getattr(params, section)
        for field in fields:
            assert hasattr(section_obj, field), f"{field} in {section} is missing"
    

    print("All required fields in params.json are present.")

