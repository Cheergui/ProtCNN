from pathlib import Path
import json
from types import SimpleNamespace

from models.protcnn import ProtCNN
from data.datamodule import DataModule


from pytorch_lightning import Trainer


def evaluate(checkpoint_path, 
             data_dir,
             batch_size,
             max_len,
             shuffle,
             num_workers,
             pin_memory):
    
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
    
    root = Path(__file__).parent.parent
    json_path = root / 'params.json'

    with open(json_path, 'r') as file:
            params = json.load(file, object_hook=lambda d: SimpleNamespace(**d))
            
    # Getting the checkpoint_path
    evaluation_params = params.evaluation
    checkpoint_path = evaluation_params.checkpoint_path
    
    # Getting dataloader parameters
    dataloader_params = params.dataloader
    
    data_dir = root / dataloader_params.data_dir
    batch_size = dataloader_params.batch_size
    max_len = dataloader_params.max_len
    shuffle = dataloader_params.shuffle
    num_workers = dataloader_params.num_workers
    pin_memory = dataloader_params.pin_memory
    
    if Path(checkpoint_path).exists:
        evaluation_results = evaluation_results = evaluate(checkpoint_path=checkpoint_path,
                                    data_dir=data_dir,
                                    batch_size=batch_size,
                                    max_len=max_len,
                                    shuffle=shuffle,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory)
        print(evaluation_results)