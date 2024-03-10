import json
from types import SimpleNamespace
from pathlib import Path


import torch
from models.model_utils import *
import torchmetrics
from pytorch_lightning import LightningModule
from torchinfo import summary

class ProtCNN(LightningModule):

    def __init__(self, 
                 num_classes=17930,
                 in_channels=22, 
                 conv_channels=128,
                 conv_kernel_size=1, 
                 conv_padding=0,
                 bias=False,
                 num_residual_blocks=2,
                 residual_blocks_kernel_size=3,
                 residual_blocks_bias=False,
                 residual_blocks_dilation=1,
                 residual_blocks_padding=1,
                 pool_kernel_size=3,
                 pool_stride=2,
                 pool_padding=1,
                 lr=1e-5,
                 weight_decay=1e-2,
                 scheduler_milestones=[5, 8, 10, 12, 14, 16, 18, 20],
                 scheduler_gamma=0.9):
        
        super().__init__()
        
        layers = []
        
        # Conv1D layer
        layers.append(torch.nn.Conv1d(in_channels=in_channels, out_channels=conv_channels, kernel_size=conv_kernel_size, padding=conv_padding, bias=bias))
        
        # ResidualBlock layers
        for _ in range(num_residual_blocks):
            layers.append(
                            ResidualBlock(in_channels=conv_channels,
                                        out_channels=conv_channels,
                                        kernel_size=residual_blocks_kernel_size,
                                        bias=residual_blocks_bias,
                                        dilation=residual_blocks_dilation,
                                        padding=residual_blocks_padding)
                          )
        
        # MaxPool1D layer
        layers.append(torch.nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding))
        
        # Flatten layer
        layers.append(Lambda(lambda x: x.flatten(start_dim=1)))
        
        # Linear layer
        layers.append(torch.nn.LazyLinear(out_features=num_classes))
        
        self.model = torch.nn.Sequential(*layers)
        
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        
        self.optim_params = {'lr':lr, 'weight_decay':weight_decay, 'scheduler_milestones':scheduler_milestones, 'scheduler_gamma':scheduler_gamma}
        
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x.float())

    def training_step(self, batch, batch_idx):
        x, y = batch['sequence'], batch['target']
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        pred = torch.argmax(y_hat, dim=1)
        self.train_acc(pred, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['sequence'], batch['target']
        y_hat = self(x)
        pred = torch.argmax(y_hat, dim=1)
        acc = self.val_acc(pred, y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)

        return acc
    
    def test_step(self, batch, batch_idx):
        x, y = batch['sequence'], batch['target']
        y_hat = self(x)
        pred = torch.argmax(y_hat, dim=1)
        acc = self.val_acc(pred, y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)

        return acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.optim_params["lr"], weight_decay=self.optim_params["weight_decay"])
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.optim_params["scheduler_milestones"], gamma=self.optim_params["scheduler_gamma"])

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }

if __name__ == "__main__":
    
    # Getting the json data into an object params
    curr_path = Path('.')
    
    json_path = curr_path / 'params.json'
    
    with open(json_path, 'r') as file:
        params = json.load(file, object_hook=lambda d: SimpleNamespace(**d))
    
    # Getting model paramaters
    model_params = params.model
    
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
    lr = model_params.lr
    weight_decay = model_params.weight_decay
    scheduler_milestones = model_params.scheduler_milestones
    scheduler_gamma = model_params.scheduler_gamma
    
    # Getting some dataloader paramaters
    dataloader_params = params.dataloader
    
    batch_size = dataloader_params.batch_size
    max_len = dataloader_params.max_len
    
    
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
                    lr=lr,
                    weight_decay=weight_decay,
                    scheduler_milestones=scheduler_milestones,
                    scheduler_gamma=scheduler_gamma
                    )
    
    # Priting the summary state of the model
    summary(prot_cnn, input_size=(batch_size, in_channels, max_len))
    
    # Test the Conv1D layer parameters
    conv1d_layer = prot_cnn.model[0]
    assert isinstance(conv1d_layer, torch.nn.Conv1d), "First layer should be Conv1d"
    assert conv1d_layer.in_channels == in_channels, "Incorrect in_channels in Conv1d"
    assert conv1d_layer.out_channels == conv_channels, "Incorrect out_channels in Conv1d"
    assert conv1d_layer.kernel_size[0] == conv_kernel_size, "Incorrect kernel_size in Conv1d"
    assert conv1d_layer.padding[0] == conv_padding, "Incorrect padding in Conv1d"

    # Test the ResidualBlock layers
    residual_block_count = 0
    for i in range(1, num_residual_blocks + 1):
        residual_block_count += 1
        residual_block_layer = prot_cnn.model[i]
        first_conv = residual_block_layer.conv1
        second_conv = residual_block_layer.conv2
        assert isinstance(prot_cnn.model[i], ResidualBlock), f"Layer {i} should be ResidualBlock"
        assert residual_block_layer.in_channels == conv_channels, "Incorrect in_channels in ResidualBlock"
        assert residual_block_layer.out_channels == conv_channels, "Incorrect out_channels in ResidualBlock"
        assert residual_block_layer.kernel_size == residual_blocks_kernel_size, "Incorrect kernel size in ResidualBlock"
        assert residual_block_layer.bias == residual_blocks_bias, "Incorrect bias setting in ResidualBlock"
        assert residual_block_layer.dilation == residual_blocks_dilation, "Incorrect dilation in ResidualBlock"
        assert residual_block_layer.padding == residual_blocks_padding, "Incorrect padding in ResidualBlock"
    assert residual_block_count == num_residual_blocks, "Incorrect number of ResidualBlocks in the model"

    # Test the MaxPool1D layer
    maxpool_layer = prot_cnn.model[num_residual_blocks + 1]
    assert isinstance(maxpool_layer, torch.nn.MaxPool1d), "MaxPool1d layer missing or in incorrect position"
    assert maxpool_layer.kernel_size == pool_kernel_size, "Incorrect kernel_size in MaxPool1d"
    assert maxpool_layer.stride == pool_stride, "Incorrect stride in MaxPool1d"
    assert maxpool_layer.padding == pool_padding, "Incorrect padding in MaxPool1d"

    # Test the Flatten layer
    assert isinstance(prot_cnn.model[num_residual_blocks + 2], Lambda), "Flatten layer missing or in incorrect position"

    # Test the Linear layer
    linear_layer = prot_cnn.model[num_residual_blocks + 3]
    assert isinstance(linear_layer, torch.nn.Linear), "Final layer should be Linear"
    
    print("All tests passed.")
        

        
        
        
        
        
        
