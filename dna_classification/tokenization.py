class DNATokenizer:
    def __init__(
        self,
        vocab_size: int,
        k: int = 1,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
    ):
        self.vocab_size = vocab_size
        self.k = k
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.token_to_id = {}

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

        # initialize the tokenizer
        vocab_size = len(mers)
        k = len(mers[0])
        tokenizer = cls(vocab_size, k)
        tokenizer.token_to_id = {mer: i for i, mer in enumerate(mers)}

        return tokenizer

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
