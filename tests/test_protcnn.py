import pytest
import json
from pathlib import Path
from types import SimpleNamespace
from models.protcnn import ProtCNN, ResidualBlock, Lambda
import torch

@pytest.fixture()
def dataloader_model_params():
    json_path = Path('.') / 'params.json'
    with open(json_path, 'r') as file:
        params = json.load(file, object_hook=lambda d: SimpleNamespace(**d))
    return params.dataloader, params.model

@pytest.fixture()
def prot_cnn(dataloader_model_params):
    
    model_params = dataloader_model_params[1]
    
    num_classes = model_params.num_classes
    in_channels = model_params.in_channels
    conv_channels = model_params.conv_channels
    conv_kernel_size = model_params.conv_kernel_size
    conv_padding = model_params.conv_padding
    bias = model_params.bias
    num_residual_blocks = model_params.num_residual_blocks
    residual_blocks_kernel_size = model_params.residual_blocks_kernel_size
    residual_blocks_bias = model_params.residual_blocks_bias
    residual_blocks_dilation = model_params.residual_blocks_dilation
    residual_blocks_padding = model_params.residual_blocks_padding
    pool_kernel_size = model_params.pool_kernel_size
    pool_stride = model_params.pool_stride
    pool_padding = model_params.pool_padding
    optim = model_params.optim
    lr = model_params.lr
    weight_decay = model_params.weight_decay
    scheduler_milestones = model_params.scheduler_milestones
    scheduler_gamma = model_params.scheduler_gamma
    
    prot_cnn = ProtCNN(num_classes=num_classes,
                    in_channels=in_channels,
                    conv_channels=conv_channels,
                    conv_kernel_size=conv_kernel_size,
                    conv_padding=conv_padding,
                    bias=bias,
                    num_residual_blocks=num_residual_blocks,
                    residual_blocks_kernel_size=residual_blocks_kernel_size,
                    residual_blocks_bias=residual_blocks_bias,
                    residual_blocks_dilation=residual_blocks_dilation,
                    residual_blocks_padding=residual_blocks_padding,
                    pool_kernel_size=pool_kernel_size,
                    pool_stride=pool_stride,
                    pool_padding=pool_padding,
                    optim=optim,
                    lr=lr,
                    weight_decay=weight_decay,
                    scheduler_milestones=scheduler_milestones,
                    scheduler_gamma=scheduler_gamma
                    )  
    
    return prot_cnn

def test_conv1d_layer(dataloader_model_params, prot_cnn):

    model_params = dataloader_model_params[1]
    
    in_channels = model_params.in_channels
    conv_channels = model_params.conv_channels
    conv_kernel_size = model_params.conv_kernel_size
    conv_padding = model_params.conv_padding
    
    conv1d_layer = prot_cnn.model[0]
    assert isinstance(conv1d_layer, torch.nn.Conv1d), "First layer should be Conv1d"
    assert conv1d_layer.in_channels == in_channels, "Incorrect in_channels in Conv1d"
    assert conv1d_layer.out_channels == conv_channels, "Incorrect out_channels in Conv1d"
    assert conv1d_layer.kernel_size[0] == conv_kernel_size, "Incorrect kernel_size in Conv1d"
    assert conv1d_layer.padding[0] == conv_padding, "Incorrect padding in Conv1d"
    
def test_residual_block_layers(dataloader_model_params, prot_cnn):
    
    model_params = dataloader_model_params[1]
    
    num_residual_blocks = model_params.num_residual_blocks
    conv_channels = model_params.conv_channels
    residual_blocks_kernel_size = model_params.residual_blocks_kernel_size
    residual_blocks_bias = model_params.residual_blocks_bias
    residual_blocks_dilation = model_params.residual_blocks_dilation
    residual_blocks_padding = model_params.residual_blocks_padding

    residual_block_count = 0
    for i in range(1, num_residual_blocks + 1):
        residual_block_layer = prot_cnn.model[i]
        
        assert isinstance(residual_block_layer, ResidualBlock), f"Layer {i} should be ResidualBlock"
        assert residual_block_layer.conv1.in_channels == conv_channels, "Incorrect in_channels in first convolution of ResidualBlock"
        assert residual_block_layer.conv1.out_channels == conv_channels, "Incorrect out_channels in first convolution of ResidualBlock"
        assert residual_block_layer.conv1.kernel_size[0] == residual_blocks_kernel_size, "Incorrect kernel size in first convolution of ResidualBlock"
        assert residual_block_layer.conv1.padding[0] == residual_blocks_padding, "Incorrect padding in first convolution of ResidualBlock"
        assert residual_block_layer.conv1.dilation[0] == residual_blocks_dilation, "Incorrect dilation in first convolution of ResidualBlock"
        assert residual_block_layer.conv1.bias is not None if residual_blocks_bias else residual_block_layer.conv1.bias is None, "Incorrect bias setting in first convolution of ResidualBlock"
        
        assert residual_block_layer.conv2.in_channels == conv_channels, "Incorrect in_channels in second convolution of ResidualBlock"
        assert residual_block_layer.conv2.out_channels == conv_channels, "Incorrect out_channels in second convolution of ResidualBlock"
        assert residual_block_layer.conv2.kernel_size[0] == residual_blocks_kernel_size, "Incorrect kernel size in second convolution of ResidualBlock"
        assert residual_block_layer.conv2.padding[0] == residual_blocks_padding, "Incorrect padding in second convolution of ResidualBlock"
        assert residual_block_layer.conv2.dilation[0] == residual_blocks_dilation, "Incorrect dilation in second convolution of ResidualBlock"
        assert residual_block_layer.conv2.bias is not None if residual_blocks_bias else residual_block_layer.conv1.bias is None, "Incorrect bias setting in second convolution of ResidualBlock"
        
        residual_block_count += 1

    assert residual_block_count == num_residual_blocks, "Incorrect number of ResidualBlocks in the model"
    
def test_maxpool_layer(dataloader_model_params, prot_cnn):
    model_params = dataloader_model_params[1]

    num_residual_blocks = model_params.num_residual_blocks
    pool_kernel_size = model_params.pool_kernel_size
    pool_stride = model_params.pool_stride
    pool_padding = model_params.pool_padding
    
    maxpool_layer = prot_cnn.model[num_residual_blocks + 1]
    assert isinstance(maxpool_layer, torch.nn.MaxPool1d), "MaxPool1d layer missing or in incorrect position"
    assert maxpool_layer.kernel_size == pool_kernel_size, "Incorrect kernel_size in MaxPool1d"
    assert maxpool_layer.stride == pool_stride, "Incorrect stride in MaxPool1d"
    assert maxpool_layer.padding == pool_padding, "Incorrect padding in MaxPool1d"

def test_flatten_layer(dataloader_model_params, prot_cnn):
    model_params = dataloader_model_params[1]
    num_residual_blocks = model_params.num_residual_blocks

    assert isinstance(prot_cnn.model[num_residual_blocks + 2], Lambda), "Flatten layer missing or in incorrect position"

def test_linear_layer(dataloader_model_params, prot_cnn):
    model_params = dataloader_model_params[1]
    num_residual_blocks = model_params.num_residual_blocks

    linear_layer = prot_cnn.model[num_residual_blocks + 3]
    assert isinstance(linear_layer, torch.nn.Linear), "Final layer should be Linear"
    
def test_optimizer(dataloader_model_params, prot_cnn):
    model_params = dataloader_model_params[1]
    optim = model_params.optim
    
    created_optim = prot_cnn.configure_optimizers()["optimizer"]

    if optim == "adam":
        assert isinstance(created_optim, torch.optim.Adam), "Optimizer should be Adam"
    elif optim == "sgd":
        assert isinstance(created_optim, torch.optim.SGD), "Optimizer should be SGD"

print("All tests for protcnn passed.")