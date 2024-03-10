from pathlib import Path

import pandas as pd
from tqdm import tqdm


def reader(partition, data_path):
    """
    Reads and aggregates the csv files located in the specific partition within the data path.
    
    Parameters
    ----------
    
    partition (str): The name of the partition directory within the data path folder. Allowable values are 'train', 'test' and 'dev'.
                     This directory should contain the csv files to be processed.
    
    data_path (str): The main path of the data folder that contains the different partition directories.
    
    Returns
    -------
    
    tuple: A tuple containing two pandas Series:
            - The first Series contains the 'sequence' column aggregated from all files. This is the raw X input.
            - The second Series contains the 'family_accession' column aggregated from all files. This is the raw Y input.
    
    """
    data = []
    path = Path(data_path) / partition
    for file in tqdm(path.iterdir()):
        data.append(pd.read_csv(file, index_col=None, usecols=["sequence", "family_accession"]))
        all_data = pd.concat(data)
    return all_data["sequence"], all_data["family_accession"]


def build_labels(targets):
    """
    Contructs a dictionary that maps each unique target to unique integer values (id labels).
    
    Parameters
    ----------
    
    targets (pandas.Series): A series containing target labels.
    
    Returns
    -------
    
    fam2label (dic): A dictionary where keys are unique labels from the input 'targets' and values are 
    corresponding unique integers. Includes the special label '<unk>' mapped to 0.
    
    """
    unique_targets = targets.unique()
    fam2label = {target: i for i, target in enumerate(unique_targets, start=1)}
    fam2label['<unk>'] = 0

    return fam2label


def build_vocab(data):
    """
    Builds a dictionary that encodes each amino acids into an unique integer value.

    Parameters
    ----------

    data (pandas.Series): A series containing amino acids sequences.

    Returns
    -------

    word2id (dic): A dictionary where keys are unique amino acids from the sequences of the input 'data' and values are 
    corresponding unique integers. Includes the special labels '<unk>' mapped to 1 and '<pad>' mapped to 0.

    """
    voc = set()
    rare_AAs = {'X', 'U', 'B', 'O', 'Z'}
    for sequence in data:
        voc.update(sequence)

    unique_AAs = sorted(voc - rare_AAs)

    word2id = {w: i for i, w in enumerate(unique_AAs, start=2)}
    word2id['<pad>'] = 0
    word2id['<unk>'] = 1

    return word2id



