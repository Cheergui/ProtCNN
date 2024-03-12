import json
from types import SimpleNamespace
from pathlib import Path

from data.data_utils import *

from torch.utils.data import Dataset
import torch
import numpy as np


class SequenceDataset(Dataset):
    """
    A custom dataset class for handling sequence data.

    This class is designed to work with sequence data, particularly for tasks like sequence classification. It initializes by reading data using a reader function, building a vocabulary and label mappings, and setting a maximum sequence length. It provides methods to get the length of the dataset and to retrieve a specific item from the dataset, suitably preprocessed.

    Parameters
    ----------
    max_len : int
        The maximum length of the sequences. Sequences longer than `max_len` are truncated, and shorter ones are padded.
    data_path : str
        The path to the dataset directory.
    split : str
        The specific subset of the dataset to be used (e.g., 'train', 'test').

    Attributes
    ----------
    data : pandas.Series
        A series containing the sequence data.
    label : pandas.Series
        A series containing the labels for the sequence data.
    word2id : dict
        A dictionary mapping each unique word in the dataset to a unique integer ID.
    fam2label : dict
        A dictionary mapping each unique label in the dataset to a unique integer ID.
    max_len : int
        The maximum length of sequences after preprocessing.
    """

    def __init__(self, word2id, fam2label, max_len, data_path, split):

        self.data, self.label = reader(split, data_path)
        
        self.word2id = word2id
        self.fam2label = fam2label
        self.max_len = max_len


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq = self.preprocess(self.data.iloc[index])
        label = self.fam2label.get(self.label.iloc[index], self.fam2label['<unk>'])

        return {'sequence': seq, 'target' : label}

    def preprocess(self, text):
        """
        Preprocess a single sequence for model input.

        This method takes a sequence as input, truncates it to `max_len` if necessary, encodes it using the `word2id` dictionary, pads it to `max_len`, and then converts it to a one-hot encoded tensor. 

        Parameters
        ----------
        text : str
            The sequence to be preprocessed.

        Returns
        -------
        torch.Tensor
            The preprocessed sequence in one-hot encoded form, with shape (num_classes, max_len).
        """
        seq = []

        # Encode into IDs
        for word in text[:self.max_len]:
            seq.append(self.word2id.get(word, self.word2id['<unk>']))

        # Pad to maximal length
        if len(seq) < self.max_len:
            seq += [self.word2id['<pad>'] for _ in range(self.max_len - len(seq))]

        # Convert list into tensor
        seq = torch.from_numpy(np.array(seq))

        # One-hot encode
        one_hot_seq = torch.nn.functional.one_hot(seq, num_classes=len(self.word2id), )

        # Permute channel (one-hot) dim first
        one_hot_seq = one_hot_seq.permute(1,0)

        return one_hot_seq
    
    
if __name__ == "__main__":
    
    # Getting the json data into an object params
    root = Path(__file__).parent.parent
    
    json_path = root / 'params.json'
    
    with open(json_path, 'r') as file:
        params = json.load(file, object_hook=lambda d: SimpleNamespace(**d))
    
    # Getting dataloader parameters
    dataloader_params = params.dataloader
    
    data_path = root / dataloader_params.data_dir
    max_len = dataloader_params.max_len   

    data, label = reader('train', data_path)
    word2id = build_vocab(data)
    fam2label = build_labels(label)

    dataset = SequenceDataset(word2id=word2id, fam2label=fam2label, max_len=max_len, data_path=data_path, split='train')
    print(f"Dataset size: {len(dataset)}")

    assert len(dataset) > 0, "Dataset is empty"

    first_item = dataset[0]
    print("First item in dataset:")
    print(first_item)

    assert first_item['sequence'].shape == (22, max_len), f"Sequence shape is not (22, {max_len})"
    assert isinstance(first_item['target'], (int, float)), "target value is not a scalar"

    
    print("All tests passed.")
    