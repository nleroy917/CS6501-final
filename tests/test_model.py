import pytest

import torch

from dna_classification.models import DNASequenceClassifier

@pytest.fixture
def pre_tokenized_sequences() -> list[list[int]]:
    return [
        [0, 1, 2, 0, 3, 2, 1, 2, 0],
        [1, 2, 0, 3, 2, 1, 2, 0, 1],
        [0, 3, 2, 2, 1, 2, 0, 1, 2],
        [0, 1, 2, 0, 3, 2, 1, 2, 0],
    ]

@pytest.fixture
def pre_tokenized_sequences_labels() -> list[int]:
    return [0, 1, 0, 0]

def test_forward(pre_tokenized_sequences: list[list[int]], pre_tokenized_sequences_labels: list[int]):
    model = DNASequenceClassifier(
        vocab_size=4, embedding_dim=384, hidden_dim=64, num_layers=2, num_classes=2
    )

    batch = torch.tensor(pre_tokenized_sequences)

    y = model(batch)
    assert y.shape == (len(pre_tokenized_sequences), 2)  