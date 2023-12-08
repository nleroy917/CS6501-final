import pytest

from dna_classification.tokenization import DNATokenizer


@pytest.fixture
def one_mers() -> str:
    return "tests/data/1mers.txt"


@pytest.fixture
def two_mers() -> str:
    return "tests/data/2mers.txt"


def test_init_tokenizer(two_mers: str):
    tokenizer = DNATokenizer.from_pretrained(two_mers)
    assert tokenizer.vocab_size == 17  # all 2-mers + <pad>
    assert tokenizer.k == 2


def test_tokenize_1mers(one_mers: str):
    seq = "ATGACGTGA"
    tokenizer = DNATokenizer.from_pretrained(one_mers)
    tokens = tokenizer.tokenize(seq)
    assert tokens == [0, 1, 2, 0, 3, 2, 1, 2, 0]

def test_tokenize_2mers(two_mers: str):
    seq = "ATGACGTGA"
    tokenizer = DNATokenizer.from_pretrained(two_mers)
    tokens = tokenizer.tokenize(seq)
    assert tokens == [0, 1, 2, 0, 3, 2, 1, 2, 0]