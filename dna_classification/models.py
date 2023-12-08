import os

from .tokenization import DNATokenizer

import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from yaml import safe_load, safe_dump

from .const import (
    CHECKPOINT_FILE,
    VOCAB_FILE,
    CONFIG_FILE,
    LOSS_FN,
    OPTIMIZER,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_NUM_EPOCHS,
)
from .utils import DNASequenceDataset


class DNASequenceClassifier(nn.Module):
    def __init__(
        self,
        model_path: str = None,
        tokenizer: DNATokenizer = None,
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
                tokenizer,
                vocab_size,
                embedding_dim,
                hidden_dim,
                num_layers,
                num_classes,
                dropout,
            )

    def add_tokenizer(self, tokenizer: DNATokenizer):
        self.tokenizer = tokenizer

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
            DNATokenizer.from_pretrained(vocab_file_path),
            config["vocab_size"],
            config["embedding_dim"],
            config["hidden_dim"],
            config["num_layers"],
            config["num_classes"],
            config["dropout"],
        )

        # load label map - this is needed for prediction
        self.label_map = config["label_map"]

        # load model weights
        self.load_state_dict(torch.load(checkpoint_file_path))

    def _init_model(
        self,
        tokenizer: DNATokenizer,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
    ):
        # class params
        self.tokenizer = tokenizer
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

    def train(
        self,
        data: pd.DataFrame,
        device: torch.device,
        epochs: int = DEFAULT_NUM_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        optimizer_params: dict = {},
        seed: int = 42,
    ):
        """
        Train the model.

        :param pandas.DataFrame data: data to train on
        :param torch.device device: device to train on
        :param int num_epochs: number of epochs to train for
        :param int batch_size: batch size
        :param dict optimizer_params: optimizer parameters
        :param int seed: random seed
        """
        # tokenize the data
        if self.tokenizer is None:
            raise ValueError(
                "Tokenizer is not initialized. Please initialize it with add_tokenizer() or load a pretrained model."
            )

        tokenized_data = self.tokenizer.tokenize_batch(data["sequence"].tolist())
        labels = data["label"].tolist()

        # map id to label
        self.label_map = {i: label for i, label in enumerate(set(labels))}

        x_train, x_test, y_train, y_test = train_test_split(
            tokenized_data, labels, test_size=0.2, random_state=seed
        )

        # create datasets
        train_dataset = DNASequenceDataset(x_train, y_train)
        test_dataset = DNASequenceDataset(x_test, y_test)

        # create dataloaders
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # create loss function
        criterion = LOSS_FN()

        # create optimizer
        optimizer = OPTIMIZER(self.parameters(), lr=DEFAULT_LR, **optimizer_params)

        self.model.train()
        self.model.to(device)

        for epoch in range(num_epochs):
            logs = {}
            total_loss = 0

            for batch in dataloader:
                batch = batch.to(device)
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                optimizer.zero_grad()
                outputs = model(inputs)
                outputs = outputs.transpose(1, 2)
                outputs = outputs[:, :, : targets.size(1)]

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            logs["loss"] = total_loss / len(dataloader)
            logs["perplexity"] = math.exp(logs["loss"])
            plotlosses.update(logs)
            plotlosses.send()

        average_loss = total_loss / len(dataloader)
        final_perp = math.exp(average_loss)
        return final_perp

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
            "label_map": self.label_map,
        }
        with open(os.path.join(path, CONFIG_FILE), "w") as f:
            safe_dump(config, f)

    def predict(self, sequence: str) -> int:
        """
        Predict the label of the DNA sequence.

        :param str sequence: DNA sequence

        :return: predicted label
        """
        if self.tokenizer is None:
            raise ValueError(
                "Tokenizer is not initialized. Please initialize it with add_tokenizer() or load a pretrained model."
            )
        
        tokens = self.tokenizer.tokenize(sequence)
        tokens = torch.tensor(tokens).unsqueeze(0)
        prediction = self(tokens)
        prediction = prediction.argmax(dim=1).item()
        prediction = self.label_map[prediction]

        return prediction



