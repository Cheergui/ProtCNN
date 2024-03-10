from data.data_utils import *

from torch.utils.data import Dataset
import torch
import numpy as np


class SequenceDataset(Dataset):

    def __init__(self, max_len, data_path, split):

        self.data, self.label = reader(split, data_path)
        
        self.word2id = build_vocab(self.data)
        self.fam2label = build_labels(self.label)
        self.max_len = max_len


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq = self.preprocess(self.data.iloc[index])
        label = self.fam2label.get(self.label.iloc[index], self.fam2label['<unk>'])

        return {'sequence': seq, 'target' : label}

    def preprocess(self, text):
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
    
    max_len = 120
    data_path = '/Users/aminejelloul/VisualStudioProjects/InstaDeep_TakeHome/.data'
    split = 'train'

    dataset = SequenceDataset(max_len=max_len, data_path=data_path, split=split)
    print(f"Dataset size: {len(dataset)}")

    assert len(dataset) > 0, "Dataset is empty"

    first_item = dataset[0]
    print("First item in dataset:")
    print(first_item)

    assert first_item['sequence'].shape == (22, max_len), f"Sequence shape is not (22, {max_len})"
    assert isinstance(first_item['target'], (int, float)), "target value is not a scalar"

    
    print("All tests passed.")
    