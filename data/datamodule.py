import json
from types import SimpleNamespace
from pathlib import Path


from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from data.dataloader import SequenceDataset

class DataModule(LightningDataModule):
    
    def __init__(self, data_dir, batch_size=256, max_len=120, shuffle=True, num_workers=4, pin_memory=True):
        super().__init__()
        
        self.data_dir = data_dir
        self.max_len = max_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
    def setup(self, stage):
        if stage == 'fit':
            self.data_train = SequenceDataset(self.max_len, self.data_dir, 'train')
            self.data_val = SequenceDataset(self.max_len, self.data_dir, 'dev')
        if stage == 'test':
            self.data_test = SequenceDataset(self.max_len, self.data_dir, 'test')
        
    def train_dataloader(self):
        return DataLoader(dataset=self.data_train,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory
                          )
        
    def val_dataloader(self):
        return DataLoader(dataset=self.data_val,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory
                          )
        
    def test_dataloader(self):
        return DataLoader(dataset=self.data_test,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory
                          )
        
        
if __name__ == "__main__":
    
    # Getting the json data into an object params
    curr_path = Path('.')
    
    json_path = curr_path / 'params.json'
    
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
    
    datamodule = DataModule(data_dir=data_dir,
                            batch_size=batch_size,
                            max_len=max_len,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            pin_memory=pin_memory)
    
    datamodule.setup('train')
    datamodule.setup('test')
    
    # Test train dataloader
    train_dataloader = datamodule.train_dataloader()
    train_batch = next(iter(train_dataloader))
    assert train_batch is not None, "Train dataloader returned None"
    X_train_batch, y_train_batch = train_batch['sequence'], train_batch['target']
    assert len(X_train_batch) == batch_size and len(y_train_batch) == batch_size, f"Train dataloader batch size is not correct."

    # Test validation dataloader
    val_dataloader = datamodule.val_dataloader()
    val_batch = next(iter(val_dataloader))
    assert val_batch is not None, "Validation dataloader returned None"
    X_val_batch, y_val_batch = val_batch['sequence'], val_batch['target']
    assert len(X_val_batch) == batch_size and len(y_val_batch) == batch_size, "Validation dataloader batch size is not correct."
    

    # Test test dataloader
    test_dataloader = datamodule.test_dataloader()
    test_batch = next(iter(test_dataloader))
    assert test_batch is not None, "Test dataloader returned None"
    X_test_batch, y_test_batch = test_batch['sequence'], test_batch['target']
    assert len(X_test_batch) == batch_size and len(y_test_batch) == batch_size, "Test dataloader batch size is not correct."
    
    print("All tests passed.")