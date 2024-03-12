import json
from types import SimpleNamespace
from pathlib import Path
import click

from models.protcnn import ProtCNN
from data.datamodule import DataModule
from scripts.script_utils import version

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torchinfo import summary

root = Path(__file__).parent.parent
json_path = root / 'params.json'

with open(json_path, 'r') as file:
        params = json.load(file, object_hook=lambda d: SimpleNamespace(**d))
        
        
# Extracting default values
dataloader_params = params.dataloader
model_params = params.model
trainer_params = params.trainer
logger_params = trainer_params.logger
callback_params = trainer_params.callbacks

@click.command()
@click.option('--data-dir', default=root / dataloader_params.data_dir, type=str, help='Directory within the root folder containing the dataset.')
@click.option('--batch-size', default=dataloader_params.batch_size, type=int, help='Batch size for training')
@click.option('--max-len', default=dataloader_params.max_len, type=int, help='Maximum sequence length')
@click.option('--shuffle', default=dataloader_params.shuffle, type=bool, help='Shuffle the dataset')
@click.option('--num-workers', default=dataloader_params.num_workers, type=int, help='Number of workers for data loading')
@click.option('--pin-memory', default=dataloader_params.pin_memory, type=bool, help='Use pinned memory for data loading')
@click.option('--num-classes', default=model_params.num_classes, type=int, help='Number of classes')
@click.option('--in-channels', default=model_params.in_channels, type=int, help='Number of input channels')
@click.option('--conv-channels', default=model_params.conv_channels, type=int, help='Number of convolution channels')
@click.option('--conv-kernel-size', default=model_params.conv_kernel_size, type=int, help='Kernel size for convolution layers')
@click.option('--conv-padding', default=model_params.conv_padding, type=int, help='Padding for convolution layers')
@click.option('--bias', default=model_params.bias, type=bool, help='Use bias in convolution layers')
@click.option('--num-residual-blocks', default=model_params.num_residual_blocks, type=int, help='Number of residual blocks')
@click.option('--residual-blocks-kernel-size', default=model_params.residual_blocks_kernel_size, type=int, help='Kernel size for residual blocks')
@click.option('--residual-blocks-bias', default=model_params.residual_blocks_bias, type=bool, help='Use bias in residual blocks')
@click.option('--residual-blocks-dilation', default=model_params.residual_blocks_dilation, type=int, help='Dilation for residual blocks')
@click.option('--residual-blocks-padding', default=model_params.residual_blocks_padding, type=int, help='Padding for residual blocks')
@click.option('--pool-kernel-size', default=model_params.pool_kernel_size, type=int, help='Kernel size for pooling layers')
@click.option('--pool-stride', default=model_params.pool_stride, type=int, help='Stride for pooling layers')
@click.option('--pool-padding', default=model_params.pool_padding, type=int, help='Padding for pooling layers')
@click.option('--optim', default=model_params.optim, type=str, help='Optimizer')
@click.option('--lr', default=model_params.lr, type=float, help='Learning rate')
@click.option('--weight-decay', default=model_params.weight_decay, type=float, help='Weight decay rate')
@click.option('--scheduler-milestones', default=model_params.scheduler_milestones, type=int, multiple=True, help='Scheduler milestones')
@click.option('--scheduler-gamma', default=model_params.scheduler_gamma, type=float, help='Scheduler gamma')
@click.option('--accelerator', default=trainer_params.accelerator, type=str, help='Type of accelerator')
@click.option('--max-epochs', default=trainer_params.max_epochs, type=int, help='Maximum number of epochs')
@click.option('--devices', default=trainer_params.devices, type=int, help='Number of devices')
@click.option('--precision', default=trainer_params.precision, type=str, help='Precision for training')
@click.option('--save-dir', default=root / logger_params.save_dir, type=str, help='Directory within the root folder for saving experiment outputs.')
@click.option('--name', default=logger_params.name, type=str, help='Subdirectory name within save-dir for storing specific experiment results.')
@click.option('--save-top-k', default=callback_params.save_top_k, type=int, help='Number of best models to save')
@click.option('--monitor', default=callback_params.monitor, type=str, help='Metric to monitor for model saving')
@click.option('--mode', default=callback_params.mode, type=str, help='Mode for model saving (max/min)')
def train(data_dir, batch_size, max_len, shuffle, num_workers, pin_memory, 
          num_classes, in_channels, conv_channels, conv_kernel_size, conv_padding, bias, 
          num_residual_blocks, residual_blocks_kernel_size, residual_blocks_bias, 
          residual_blocks_dilation, residual_blocks_padding, pool_kernel_size, 
          pool_stride, pool_padding, optim, lr, weight_decay, scheduler_milestones, scheduler_gamma, 
          accelerator, max_epochs, devices, precision, save_dir, name, save_top_k, monitor, mode):


        
        # Defining the datamodule
        datamodule = DataModule(data_dir=data_dir,
                                batch_size=batch_size,
                                max_len=max_len,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                pin_memory=pin_memory)
        
        
        
        # Defining the logger and the callbacks
        logger = CSVLogger(save_dir=save_dir, name=name)
        
        checkpoint_path = Path(save_dir) / name
        checkpoint_path = checkpoint_path / version(checkpoint_path.absolute())
        callbacks = ModelCheckpoint(dirpath=checkpoint_path, save_last=True, save_top_k=save_top_k, filename="model-{epoch:02d}-{val_loss:.2f}", monitor=monitor, mode=mode)
        
        # Defining the model
        model = ProtCNN(num_classes=num_classes,
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
        
        seed_everything(0)
        
        # Priting the summary state of the model before stating the training
        summary(model, input_size=(batch_size, in_channels, max_len))
        
        # Defining the trainer
        trainer = Trainer(accelerator=accelerator,
                        max_epochs=max_epochs,
                        devices=devices,
                        precision=precision,
                        callbacks=callbacks,
                        logger=logger)
        
        
        # Start the training
        trainer.fit(model=model, datamodule=datamodule)
        
        # Evaluate the model
        trainer.test(model=model, datamodule=datamodule)
    
if __name__ == "__main__":
        train()
         
        print("End of the training.")
        