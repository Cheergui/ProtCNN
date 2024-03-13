from pathlib import Path
import json
from types import SimpleNamespace
import click

from models.protcnn import ProtCNN
from data.datamodule import DataModule


from pytorch_lightning import Trainer

root = Path(__file__).parent.parent
json_path = root / 'params.json'

with open(json_path, 'r') as file:
        params = json.load(file, object_hook=lambda d: SimpleNamespace(**d))
        
# Extracting default values
evaluation_params = params.evaluation
dataloader_params = params.dataloader


@click.command()
@click.option('--checkpoint-path', default=evaluation_params.checkpoint_path, type=str, help='Directory within the root folder containing the dataset.')
@click.option('--data-dir', default=root / dataloader_params.data_dir, type=str, help='Directory within the root folder containing the dataset.')
@click.option('--batch-size', required=True, type=int, help='Batch size for evaluation')
@click.option('--max-len', required=True, type=int, help='Maximum sequence length')
@click.option('--shuffle', default=dataloader_params.shuffle, type=bool, help='Shuffle the dataset during evaluation')
@click.option('--num-workers', default=dataloader_params.num_workers, type=int, help='Number of workers for data loading during evaluation')
@click.option('--pin-memory', default=dataloader_params.pin_memory, type=bool, help='Use pinned memory for data loading during evaluation')
def evaluate(checkpoint_path, 
             data_dir,
             batch_size,
             max_len,
             shuffle,
             num_workers,
             pin_memory):
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"The checkpoint path {checkpoint_path} does not exist.")
        
    prot_cnn_loaded = ProtCNN.load_from_checkpoint(checkpoint_path=checkpoint_path)
    
    # Defining the datamodule
    datamodule = DataModule(data_dir=data_dir,
                            batch_size=batch_size,
                            max_len=max_len,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            pin_memory=pin_memory)
    
    # Defining the trainer
    trainer = Trainer()

    # Performing evaluation
    evaluation = trainer.test(model=prot_cnn_loaded, datamodule=datamodule)
    
    return evaluation


if __name__ == "__main__":
    
        evaluation_results = evaluate()
        print(evaluation_results)