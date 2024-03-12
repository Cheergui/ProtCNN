import json
from types import SimpleNamespace
from pathlib import Path


from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from data.dataloader import SequenceDataset
from data.data_utils import *

class DataModule(LightningDataModule):
    """
    A PyTorch Lightning data module for handling sequence data.

    This class extends PyTorch Lightning's LightningDataModule and is specifically designed for sequence data handling. It is responsible for setting up the datasets for different stages (training, validation, testing) and creating the corresponding data loaders.

    Parameters
    ----------
    data_dir : str
        The directory where the data files are located.
    batch_size : int, optional
        The size of the batches of data (default is 256).
    max_len : int, optional
        The maximum length of the sequences (default is 120).
    shuffle : bool, optional
        Whether to shuffle the data at every epoch (default is True).
    num_workers : int, optional
        The number of subprocesses to use for data loading (default is 4).
    pin_memory : bool, optional
        If True, the data loader will copy Tensors into CUDA pinned memory before returning them (default is True).

    Attributes
    ----------
    data_train : SequenceDataset
        The dataset for training.
    data_val : SequenceDataset
        The dataset for validation.
    data_test : SequenceDataset
        The dataset for testing.
    """
    
    def __init__(self, data_dir, batch_size=256, max_len=120, shuffle=True, num_workers=4, pin_memory=True):
        super().__init__()
        
        self.data_dir = data_dir
        self.max_len = max_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        data, label = reader('train', self.data_dir)
        self.word2id = build_vocab(data)
        self.fam2label = build_labels(label)
        
    def setup(self, stage):
        """
        Set up the data for the given stage.

        This method initializes the datasets for training, validation, and testing based on the provided stage. It is responsible for loading and preprocessing the data.

        Parameters
        ----------
        stage : str or None
            The stage for which to set up the data. Can be 'fit', 'test', or None. If None, all datasets will be set up.
        """
        if stage == 'fit':
            self.data_train = SequenceDataset(self.word2id, self.fam2label, self.max_len, self.data_dir, 'train')
            self.data_val = SequenceDataset(self.word2id, self.fam2label, self.max_len, self.data_dir, 'dev')
        if stage == 'test':
            self.data_test = SequenceDataset(self.word2id, self.fam2label, self.max_len, self.data_dir, 'test')
        
    def train_dataloader(self):
        """
        Create a data loader for the training set.

        Returns
        -------
        DataLoader
            The DataLoader for the training dataset.
        """
        return DataLoader(dataset=self.data_train,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory
                          )
        
    def val_dataloader(self):
        """
        Create a data loader for the validation set.

        Returns
        -------
        DataLoader
            The DataLoader for the validation dataset.
        """
        return DataLoader(dataset=self.data_val,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory
                          )
        
    def test_dataloader(self):
        """
        Create a data loader for the test set.

        Returns
        -------
        DataLoader
            The DataLoader for the testing dataset.
        """
        return DataLoader(dataset=self.data_test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory
                          )
        
        
if __name__ == "__main__":
    
    # Getting the json data into an object params
    root = Path(__file__).parent.parent
    json_path = root / 'params.json'
    
    with open(json_path, 'r') as file:
        params = json.load(file, object_hook=lambda d: SimpleNamespace(**d))
    
    # Getting dataloader parameters
    dataloader_params = params.dataloader
    
    data_dir = root / dataloader_params.data_dir
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
    
    datamodule.setup('fit')
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