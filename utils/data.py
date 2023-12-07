import pandas as pd
import torch

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_packed_sequence


class DNADataset(Dataset):
    """
    Dataset class for DNA sequences.
    """

    def __init__(self, seqs: list, labels: list):
        """
        Initialize the dataset.

        :param data: list of DNA sequences
        :param labels: list of labels
        """
        self.seqs = seqs
        self.labels = labels

    def __len__(self) -> int:
        """
        Return the length of the dataset.
        """
        return len(self.seqs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the item at the given index.

        :param idx: index of the item
        """
        return torch.tensor(self.seqs[idx]), torch.tensor(self.labels[idx])


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]], pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for the dataset.

    :param batch: batch of data
    """
    seqs, labels = zip(*batch)
    seqs = pad_packed_sequence(seqs, padding_value=pad_token_id)
    return seqs, torch.tensor(labels)


def load_data(path: str) -> tuple[list[str], list[int]]:
    """
    Load data from the data file.

    :param path: path to the data file
    """

    df = pd.read_csv(path)
    seqs = [str(seq) for seq in df["sequence"].tolist()]
    labels = [int(label) for label in df["label"].tolist()]
    return seqs, labels
