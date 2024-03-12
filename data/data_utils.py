from pathlib import Path

import pandas as pd
from tqdm import tqdm


def reader(partition, data_path):
    """
    Read and concatenate CSV files in a given directory.

    This function iterates over all CSV files in a specified directory (determined by the `partition` and `data_path`), reading each file into a DataFrame. The function then concatenates these DataFrames and returns two Series: sequences and family accessions.

    Parameters
    ----------
    partition : str
        The specific directory within `data_path` to search for CSV files.
    data_path : str
        The path to the main directory containing the data.

    Returns
    -------
    pandas.Series
        A Series containing sequences from the CSV files.
    pandas.Series
        A Series containing family accession numbers from the CSV files.
    """
    data = []
    path = Path(data_path) / partition
    for file in tqdm(path.iterdir()):
        data.append(pd.read_csv(file, index_col=None, usecols=["sequence", "family_accession"]))
        all_data = pd.concat(data)
    return all_data["sequence"], all_data["family_accession"]


def build_labels(targets):
    """
    Create a mapping of unique targets to numerical labels.

    This function takes a Series of targets (e.g., family accessions) and maps each unique target to a unique integer. Additionally, an unknown class '<unk>' is mapped to 0.

    Parameters
    ----------
    targets : pandas.Series
        A Series of target labels (e.g., family accessions).

    Returns
    -------
    dict
        A dictionary mapping each unique target to a numerical label.
    """
    unique_targets = targets.unique()
    fam2label = {target: i for i, target in enumerate(unique_targets, start=1)}
    fam2label['<unk>'] = 0

    return fam2label


def build_vocab(data):
    """
    Build a vocabulary from a sequence dataset.

    This function creates a vocabulary dictionary where each unique amino acid in the data is assigned a unique integer ID. Special tokens '<pad>' and '<unk>' are also included in the vocabulary, representing padding and unknown characters, respectively.

    Parameters
    ----------
    data : pandas.Series
        A Series containing amino acid sequences.

    Returns
    -------
    dict
        A dictionary mapping each unique amino acid (and special tokens) to a unique integer ID.
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



