import torch
import pandas as pd

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_packed_sequence

from .const import DATA_FILE_DELIMITER


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

def build_vocab(path: str, k: int = 1) -> dict:
    """
    Build vocabulary from the data file.

    :param path: path to the data file
    :param k: k-mer size

    :return: vocabulary
    """
    df = pd.read_csv(path, sep=DATA_FILE_DELIMITER, header=None)


class DNASequenceDataset(Dataset):
    def __init__(self, sequences: list[str], labels: list[int]):
        """
        Initialize the dataset.
        """
        assert len(sequences) == len(
            labels
        ), "Sequences and labels must be the same length."
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        """
        Get the length of the dataset.
        """
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        """
        Get the item at the index.
        """
        return torch.tensor(self.sequences[idx]), torch.tensor(self.labels[idx])
