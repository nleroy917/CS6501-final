import os

import torch
import torch.nn as nn

from huggingface_hub import hf_hub_download
from yaml import safe_load, safe_dump

from .const import CHECKPOINT_FILE, VOCAB_FILE, CONFIG_FILE


class DNASequenceClassifier(nn.Module):
    def __init__(
        self,
        model_path: str = None,
        vocab_size: int = None,
        embedding_dim: int = None,
        hidden_dim: int = None,
        num_layers: int = None,
        num_classes: int = None,
        dropout: float = 0.5,
    ):
        super(DNASequenceClassifier, self).__init__()
        if model_path is not None:
            if os.path.exists(model_path):
                self._load_local_model(
                    model_path, CHECKPOINT_FILE, VOCAB_FILE, CONFIG_FILE
                )
            else:
                self._init_from_huggingface(model_path)
                # TODO: init tokenizer when loading from huggingface or local
            self.tokenizer = None
        else:
            self._init_model(
                vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.fc(x[:, -1, :])
        return x

    @classmethod
    def from_pretrained(
        cls,
        path_to_files: str,
        model_file_name: str = CHECKPOINT_FILE,
        universe_file_name: str = VOCAB_FILE,
        config_file_name: str = CONFIG_FILE,
    ) -> "DNASequenceClassifier":
        model_file_path = os.path.join(path_to_files, model_file_name)
        universe_file_path = os.path.join(path_to_files, universe_file_name)
        config_file_path = os.path.join(path_to_files, config_file_name)

        instance = cls()
        instance._load_local_model(
            model_file_path, universe_file_path, config_file_path
        )
        instance.trained = True

        return instance

    def _init_from_huggingface(
        self,
        model_path: str,
        checkpoint_file: str = CHECKPOINT_FILE,
        vocab_file: str = VOCAB_FILE,
        config_file: str = CONFIG_FILE,
    ):
        model_file_path = hf_hub_download(model_path, checkpoint_file)
        vocab_file_path = hf_hub_download(model_path, vocab_file)
        config_file_path = hf_hub_download(model_path, config_file)

        self._load_local_model(model_file_path, vocab_file_path, config_file_path)

    def _load_local_model(
        self,
        path: str,
        checkpoint_file: str = CHECKPOINT_FILE,
        vocab_file: str = VOCAB_FILE,
        config_file: str = CONFIG_FILE,
    ):
        checkpoint_file_path = os.path.join(path, checkpoint_file)
        vocab_file_path = os.path.join(path, vocab_file)
        config_file_path = os.path.join(path, config_file)

        # load config yaml
        with open(config_file_path, "r") as f:
            config = safe_load(f)

        # get model config from yaml
        self._init_model(
            config["vocab_size"],
            config["embedding_dim"],
            config["hidden_dim"],
            config["num_layers"],
            config["num_classes"],
            config["dropout"],
        )

        # load model weights
        self.load_state_dict(torch.load(checkpoint_file_path))

    def _init_model(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
    ):
        # class params
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout

        # model layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def export(self, path: str):
        # save model
        torch.save(self.state_dict(), os.path.join(path, CHECKPOINT_FILE))

        # save config
        config = {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
        }
        with open(os.path.join(path, CONFIG_FILE), "w") as f:
            safe_dump(config, f)
