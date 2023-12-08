from typing import Union

from .const import PAD_TOKEN

class DNATokenizer:
    def __init__(
        self,
        vocab: Union[str, dict[str, int]],
    ):
        """
        Initialize the tokenizer.

        :param vocab: vocabulary file or dictionary
        """
        self._init_tokenizer(vocab)
    
    def _init_tokenizer(self, vocab: Union[str, dict[str, int]]):
        """
        Initialize the tokenizer.

        :param vocab: vocabulary file or dictionary
        """
        if isinstance(vocab, str):
            with open(vocab, "r") as f:
                vocab = f.read().splitlines()
            vocab = {mer: i for i, mer in enumerate(vocab)}
        elif isinstance(vocab, dict):
            vocab = vocab
        
        # add the pad token if it's not in the vocab
        if PAD_TOKEN not in vocab:
            vocab[PAD_TOKEN] = len(vocab)

        self.token_to_id = vocab
        self.vocab_size = len(vocab)
        self.k = len(list(vocab.keys())[0])

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
        

    def tokenize(self, sequence: str ) -> list[int]:
        """
        Tokenize the DNA sequence.

        :param sequence: DNA sequence

        :return: list of tokens
        """
        # don't know how token_to_id is made
        ids = []
        for i in range(len(sequence) - self.k + 1):
            token = sequence[i:i+self.k]
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

        for i, sequence in enumerate(tokenized_sequences):
            tokenized_sequences[i] = sequence + self.token_to_id[self.pad_token]*(max_len - len(sequence))

        return tokenized_sequences