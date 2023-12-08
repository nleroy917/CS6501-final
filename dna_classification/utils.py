import torch
from torch.utils.data import Dataset


def build_vocab(path: str, k: int = 1) -> dict:
    """
    Build vocabulary from the data file.

    :param path: path to the data file
    :param k: k-mer size
    """
    pass


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
