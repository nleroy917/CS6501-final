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