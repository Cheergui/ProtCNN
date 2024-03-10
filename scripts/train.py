import json
from types import SimpleNamespace
from pathlib import Path
import click

from models.protcnn import ProtCNN
from data.datamodule import DataModule

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from torchinfo import summary
    
# @click.command()
# @click.option('--num-classes', default=17930, type=int, help='Number of classes for the model.')
# @click.option('--in-channels', default=22, type=int, help='Number of input channels.')
# @click.option('--conv-channels', default=128, type=int, help='Number of convolutional channels.')
# @click.option('--conv-kernel-size', default=1, type=int, help='Size of the convolutional kernel.')
# @click.option('--conv-padding', default=0, type=int, help='Padding for the convolution layer.')
# @click.option('--bias', default=False, type=bool, help='Whether to use bias in convolution layers.')
# @click.option('--dilation', default=2, type=int, help='Dilation rate for the convolution layer.')
# @click.option('--num-residual-blocks', default=2, type=int, help='Number of residual blocks in the model.')
# @click.option('--pool-kernel-size', default=3, type=int, help='Size of the kernel for max pooling.')
# @click.option('--pool-stride', default=2, type=int, help='Stride for max pooling.')
# @click.option('--pool-padding', default=1, type=int, help='Padding for max pooling.')
# @click.option('--lr', default=1e-5, type=float, help='Learning rate for the optimizer.')
# @click.option('--weight-decay', default=1e-2, type=float, help='Weight decay factor for optimizer.')
# @click.option('--scheduler-milestones', default=[5, 8, 10, 12, 14, 16, 18, 20], type=list, multiple=True, help='Milestones for the learning rate scheduler.')
# @click.option('--scheduler-gamma', default=0.9, type=float, help='Gamma value for the learning rate scheduler.')
def train(data_dir, batch_size, max_len, shuffle, num_workers, pin_memory, 
          num_classes, in_channels, conv_channels, conv_kernel_size, conv_padding, bias, 
          num_residual_blocks, residual_blocks_kernel_size, residual_blocks_bias, 
          residual_blocks_dilation, residual_blocks_padding, pool_kernel_size, 
          pool_stride, pool_padding, lr, weight_decay, scheduler_milestones, scheduler_gamma, 
          accelerator, max_epochs, devices, precision, dirpath, save_top_k, monitor, mode):
        
        
        
        # Defining the datamodule
        datamodule = DataModule(data_dir=data_dir,
                                batch_size=batch_size,
                                max_len=max_len,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                pin_memory=pin_memory)
        
        
        # Defining the callbacks
        callbacks = [TQDMProgressBar(refresh_rate=1),
                     ModelCheckpoint(dirpath=dirpath, save_last=True, save_top_k=save_top_k, filename="model-{epoch:02d}-{val_loss:.2f}", monitor=monitor, mode=mode),
                     EarlyStopping(monitor=monitor, mode=mode, patience=5, verbose=True)]
        
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
                        callbacks=callbacks)
        
        
        # Start the training
        trainer.fit(model=model, datamodule=datamodule)
    
    
    
if __name__ == "__main__":

        curr_path = Path('.')
        
        # json_path = curr_path / 'params.json'
        
        json_path = 'params.json'
    
        with open(json_path, 'r') as file:
                params = json.load(file, object_hook=lambda d: SimpleNamespace(**d))
                
                
        # Getting dataloader parameters
        dataloader_params = params.dataloader
        
        data_dir = curr_path / dataloader_params.data_dir
        batch_size = dataloader_params.batch_size
        max_len = dataloader_params.max_len
        shuffle = dataloader_params.shuffle
        num_workers = dataloader_params.num_workers
        pin_memory = dataloader_params.pin_memory
        
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
        
        # Getting trainer paramaters
        trainer_params = params.trainer
        
        accelerator = trainer_params.accelerator
        max_epochs = trainer_params.max_epochs
        devices = trainer_params.devices
        precision = trainer_params.precision
        
        callback_params = trainer_params.callbacks
        
        dirpath = callback_params.dirpath
        save_top_k = callback_params.save_top_k
        monitor = callback_params.monitor
        mode = callback_params.mode
        
        train(data_dir=data_dir, batch_size=batch_size, max_len=max_len, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, 
                num_classes=num_classes, in_channels=in_channels, conv_channels=conv_channels, conv_kernel_size=conv_kernel_size, 
                conv_padding=conv_padding, bias=bias, num_residual_blocks=num_residual_blocks, 
                residual_blocks_kernel_size=residual_blocks_kernel_size, residual_blocks_bias=residual_blocks_bias, 
                residual_blocks_dilation=residual_blocks_dilation, residual_blocks_padding=residual_blocks_padding, 
                pool_kernel_size=pool_kernel_size, pool_stride=pool_stride, pool_padding=pool_padding, 
                lr=lr, weight_decay=weight_decay, scheduler_milestones=scheduler_milestones, scheduler_gamma=scheduler_gamma, 
                accelerator=accelerator, max_epochs=max_epochs, devices=devices, precision=precision, 
                dirpath=dirpath, save_top_k=save_top_k, monitor=monitor, mode=mode)

        
        print("End of the training.")
        