import json
import optuna
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
json_path = 'params.json'

with open(root / json_path, 'r') as file:
        params = json.load(file, object_hook=lambda d: SimpleNamespace(**d))

# Extracting default values
dataloader_params = params.dataloader
model_params = params.model
trainer_params = params.trainer
logger_params = trainer_params.logger
callback_params = trainer_params.callbacks
hyperparameter_tuning_params = params.hyperparameter_tuning


@click.command()
@click.option('--data-dir', default=root / dataloader_params.data_dir, type=str, help='Directory within the root folder containing the dataset.')
@click.option('--batch-size', default=hyperparameter_tuning_params.batch_size, type=int, multiple=True, help='Batch sizes to explore for training')
@click.option('--max-len', default=hyperparameter_tuning_params.max_len, type=int, multiple=True, help='Maximum sequence lengths to explore')
@click.option('--shuffle', default=dataloader_params.shuffle, type=bool, help='Shuffle the dataset')
@click.option('--num-workers', default=dataloader_params.num_workers, type=int, help='Number of workers for data loading')
@click.option('--pin-memory', default=dataloader_params.pin_memory, type=bool, help='Use pinned memory for data loading')
@click.option('--num-classes', default=model_params.num_classes, type=int, help='Number of classes')
@click.option('--in-channels', default=model_params.in_channels, type=int, help='Number of input channels')
@click.option('--conv-channels', default=hyperparameter_tuning_params.conv_channels, type=int, multiple=True, help='Convolution channels to explore')
@click.option('--conv-kernel-size', default=model_params.conv_kernel_size, type=int, help='Kernel size for convolution layers')
@click.option('--conv-padding', default=model_params.conv_padding, type=int, help='Padding for convolution layers')
@click.option('--bias', default=model_params.bias, type=bool, help='Use bias in convolution layers')
@click.option('--num-residual-blocks', default=hyperparameter_tuning_params.num_residual_blocks, type=int, multiple=True, help='Number of residual blocks to explore')
@click.option('--residual-blocks-kernel-size', default=model_params.residual_blocks_kernel_size, type=int, help='Kernel size for residual blocks')
@click.option('--residual-blocks-bias', default=model_params.residual_blocks_bias, type=bool, help='Use bias in residual blocks')
@click.option('--residual-blocks-dilation', default=model_params.residual_blocks_dilation, type=int, help='Dilation for residual blocks')
@click.option('--residual-blocks-padding', default=model_params.residual_blocks_padding, type=int, help='Padding for residual blocks')
@click.option('--pool-kernel-size', default=model_params.pool_kernel_size, type=int, help='Kernel size for pooling layers')
@click.option('--pool-stride', default=model_params.pool_stride, type=int, help='Stride for pooling layers')
@click.option('--pool-padding', default=model_params.pool_padding, type=int, help='Padding for pooling layers')
@click.option('--optim', default=hyperparameter_tuning_params.optim, type=str, multiple=True, help='Optimizers to explore')
@click.option('--lr', default=hyperparameter_tuning_params.lr, type=float, multiple=True, help='Learning rates to explore')
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
@click.option('--metric', default=hyperparameter_tuning_params.metric, type=str, help='Metric used for model objective (test_acc, val_acc))')
@click.option('--direction', default=hyperparameter_tuning_params.direction, type=str, help='Mode for model objective (max/min)')
@click.option('--n_trials', default=hyperparameter_tuning_params.n_trials, type=int, help='Number of the trials for the hyperparameter tuning study.')
def hyperparameter_tuning(data_dir, batch_size, max_len, shuffle, num_workers, pin_memory, 
          num_classes, in_channels, conv_channels, conv_kernel_size, conv_padding, bias, 
          num_residual_blocks, residual_blocks_kernel_size, residual_blocks_bias, 
          residual_blocks_dilation, residual_blocks_padding, pool_kernel_size, 
          pool_stride, pool_padding, optim, lr, weight_decay, scheduler_milestones, scheduler_gamma, 
          accelerator, max_epochs, devices, precision, save_dir, name, save_top_k, monitor, mode, metric, direction, n_trials):
    
    def objective(trial):
        trial_batch_size = trial.suggest_categorical('batch_size', batch_size)
        trial_max_len = trial.suggest_categorical('max_len', max_len)
        trial_conv_channels = trial.suggest_categorical('conv_channels', conv_channels)
        trial_num_residual_blocks = trial.suggest_categorical('num_residual_blocks', num_residual_blocks)
        trial_optim = trial.suggest_categorical('optim', optim)
        trial_lr = trial.suggest_categorical('lr', lr)
        
        
        # Defining the datamodule
        datamodule = DataModule(data_dir=data_dir,
                                batch_size=trial_batch_size,
                                max_len=trial_max_len,
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
                        conv_channels=trial_conv_channels,
                        conv_kernel_size=conv_kernel_size,
                        conv_padding=conv_padding,
                        bias=bias,
                        num_residual_blocks=trial_num_residual_blocks,
                        residual_blocks_kernel_size=residual_blocks_kernel_size,
                        residual_blocks_bias=residual_blocks_bias,
                        residual_blocks_dilation=residual_blocks_dilation,
                        residual_blocks_padding=residual_blocks_padding,
                        pool_kernel_size=pool_kernel_size,
                        pool_stride=pool_stride,
                        pool_padding=pool_padding,
                        optim=trial_optim,
                        lr=trial_lr,
                        weight_decay=weight_decay,
                        scheduler_milestones=scheduler_milestones,
                        scheduler_gamma=scheduler_gamma
                        )
        
        seed_everything(0)
        
        # Priting the summary state of the model before stating the training
        summary(model, input_size=(trial_batch_size, in_channels, trial_max_len))
        
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
        
        return trainer.callback_metrics[metric]
    
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

    print(study.best_trial.params)
    
    root = Path(__file__).parent.parent
    with open(root / "best_hyperparameters.json", "w") as file:
        json.dump(study.best_trial.params, file, indent=4)

    print("Best hyperparameters saved to best_hyperparameters.json")


if __name__ == "__main__":

    hyperparameter_tuning()
    
    print("End of the hyperparameter tuning.")