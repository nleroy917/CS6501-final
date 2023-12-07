class DNATokenizer:
    def __init__(self, vocab_size: int, k: int = 1, pad_token: str = "<PAD>", unk_token: str = "<UNK>"):
        self.vocab_size = vocab_size
        self.k = k
        self.pad_token = pad_token
        self.unk_token = unk_token

    def tokenize(self, sequence: str) -> list[int]:
        """
        Tokenize the DNA sequence.

        :param sequence: DNA sequence

        :return: list of tokens
        """
        pass

    def tokenize_batch(self, batch: list[str]) -> list[list[int]]:
        """
        Tokenize the batch of DNA sequences. This will
        pad the sequences to the same length.


        :param batch: list of DNA sequences

        :return: list of tokenized sequences
        """
        pass