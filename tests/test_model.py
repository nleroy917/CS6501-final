import pytest

import torch

from dna_classification.models import DNASequenceClassifier
from dna_classification.tokenization import DNATokenizer


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


def test_forward(
    pre_tokenized_sequences: list[list[int]], pre_tokenized_sequences_labels: list[int]
):
    model = DNASequenceClassifier(
        vocab_size=4, embedding_dim=384, hidden_dim=64, num_layers=2, num_classes=2
    )

    batch = torch.tensor(pre_tokenized_sequences)

    y = model(batch)
    assert y.shape == (len(pre_tokenized_sequences), 2)

def test_export_reload_model():
    model = DNASequenceClassifier(
        vocab_size=4, embedding_dim=384, hidden_dim=64, num_layers=2, num_classes=2
    )

    tokenizer = DNATokenizer(
        vocab_size=4, k=1
    )
    tokenizer.token_to_id = {"A": 0, "T": 1, "G": 2, "C": 3}

    model.add_tokenizer(tokenizer)

    # this is for testing
    model.label_map = {0: "negative", 1: "positive"}

    # get embedding for A
    a_embedding = model.embedding(torch.tensor([0]))

    model.export("tests/data/model")

    model.from_pretrained("tests/data/model")

    # get embedding for A
    a_embedding_new = model.embedding(torch.tensor([0]))

    assert model.vocab_size == 4
    assert torch.allclose(a_embedding, a_embedding_new)

    