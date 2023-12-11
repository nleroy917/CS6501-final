from typing import Union

from tqdm import tqdm

from .const import PAD_TOKEN


class DNATokenizer:
    def __init__(
        self,
        vocab: Union[str, dict[str, int]] = None,
    ):
        """
        Initialize the tokenizer.

        :param vocab: vocabulary file or dictionary
        """
        self.pad_token = PAD_TOKEN
        self.pad_token_id = None
        self._init_tokenizer(vocab)

    def _init_tokenizer(self, vocab: Union[str, dict[str, int]]):
        """
        Initialize the tokenizer.

        :param vocab: vocabulary file or dictionary
        """
        if vocab is None:
            return  # do nothing
        elif isinstance(vocab, str):
            with open(vocab, "r") as f:
                vocab = f.read().splitlines()
            vocab = {mer: i for i, mer in enumerate(vocab)}
        elif isinstance(vocab, dict):
            vocab = vocab

        # add the pad token if it's not in the vocab
        if PAD_TOKEN not in vocab:
            vocab[PAD_TOKEN] = len(vocab)

        self.token_to_id = vocab
        self.pad_token_id = vocab[PAD_TOKEN]
        self.vocab_size = len(vocab)
        self.k = len(list(vocab.keys())[0])

    def build_vocab(self, path: str, k: int = 1):
        """
        Reads DNA sequences from a file and constructs
        a k-mer vocabulary using the specified k value.
        """
        vocab = set()
        with open(path, "r") as file:
            next(file)  # skip the first line
            for line in tqdm(file, desc="Building vocab"):
                sequence = line.split("\t")[0].strip()
                kmers = [sequence[i : i + k] for i in range(len(sequence) - k + 1)]
                vocab.update(kmers)

        vocab = {mer: i for i, mer in tqdm(enumerate(vocab), total=len(vocab))}

        self._init_tokenizer(vocab)

    @classmethod
    def from_pretrained(cls, path: str) -> "DNATokenizer":
        """
        Load the tokenizer from the files.

        :param path: path to the tokenizer file

        :return: tokenizer
        """
        # read in the mers
        with open(path, "r") as f:
            mers = f.read().splitlines()

        # create the vocab
        vocab = {mer: i for i, mer in enumerate(mers)}

        tokenizer = cls(vocab)

        return tokenizer

    def tokenize(self, sequence: str) -> list[int]:
        """
        Tokenize the DNA sequence.

        :param sequence: DNA sequence

        :return: list of tokens
        """
        # don't know how token_to_id is made
        ids = []
        for i in range(len(sequence) - self.k + 1):
            token = sequence[i : i + self.k]
            ids.append(self.token_to_id[token])

        return ids

    def tokenize_batch(self, batch: list[str]) -> list[list[int]]:
        """
        Tokenize the batch of DNA sequences. This will
        pad the sequences to the same length.


        :param batch: list of DNA sequences

        :return: list of tokenized sequences
        """
        tokenized_sequences = []
        max_len = 0
        for sequence in batch:
            max_len = max(max_len, len(sequence))
            tokenized_sequences.append(self.tokenize(sequence))

        # for i, sequence in enumerate(tokenized_sequences):
        #     tokenized_sequences[i] = sequence + self.token_to_id[self.pad_token] * (
        #         max_len - len(sequence)
        #     )

        return tokenized_sequences

    def tokenize_no_overlap(self, sequence: str) -> list[int]:
        ids = []
        for i in range(0, len(sequence) - self.k + 1, self.k):
            token = sequence[i : i + self.k]
            ids.append(self.token_to_id[token])

        return ids