import os

from .tokenization import DNATokenizer

import torch
import torch.nn as nn
import pandas as pd

from tqdm import tqdm
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
from .utils import DNASequenceDataset, CollateFunction


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
        # check for completely blank init
        if all(
            [
                model_path is None,
                tokenizer is None,
                vocab_size is None,
                embedding_dim is None,
                hidden_dim is None,
                num_layers is None,
                num_classes is None,
            ]
        ):
            pass
        elif model_path is not None:
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
        checkpoint_file_path: str = CHECKPOINT_FILE,
        vocab_file_path: str = VOCAB_FILE,
        config_file_path: str = CONFIG_FILE,
    ):

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
        try:
            self.label_map = config["label_map"]
        except KeyError:
            raise KeyError(
                "Label map not found in config file. Please make sure you have a label map in your config file."
            )

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

    def train_model(
        self,
        data: pd.DataFrame,
        device: torch.device = torch.device("cpu"),
        epochs: int = DEFAULT_NUM_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        optimizer_params: dict = {
            "lr": DEFAULT_LR,
        },
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

        print("Tokenizing data...")
        tokenized_data = self.tokenizer.tokenize_batch(data["sequence"].tolist())
        labels = data["label"].tolist()

        # map id to label
        self.label_map = {i: label for i, label in enumerate(set(labels))}

        print("Splitting data...")
        x_train, x_test, y_train, y_test = train_test_split(
            tokenized_data, labels, test_size=0.2, random_state=seed
        )

        # convert labels to ids
        label_to_id = {label: id for id, label in self.label_map.items()}
        y_train = [label_to_id[label] for label in y_train]
        y_test = [label_to_id[label] for label in y_test]

        # create datasets
        print("Creating datasets...")
        train_dataset = DNASequenceDataset(x_train, y_train)
        test_dataset = DNASequenceDataset(x_test, y_test)

        collate_fn = CollateFunction(self.tokenizer.pad_token_id)

        # create dataloaders
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )

        # create loss function
        criterion = LOSS_FN()

        # create optimizer
        optimizer = OPTIMIZER(self.parameters(), **optimizer_params)

        self.train()
        self.to(device)

        train_losses = []
        validation_losses = []

        print("Training...")
        for epoch in tqdm(range(epochs), total=epochs, desc="Epochs"):
            total_loss = 0
            self.train()
            for batch in tqdm(
                train_dataloader, total=len(train_dataloader), desc="Batches"
            ):
                inputs, labels = batch

                # move to device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                total_loss += loss.item()
                optimizer.step()

            train_losses.append(total_loss / len(train_dataloader))

            # validation
            with torch.no_grad():
                self.eval()
                total_loss = 0
                for batch in test_dataloader:
                    inputs, labels = batch

                    # move to device
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()

                validation_losses.append(total_loss / len(test_dataloader))

            print(f"Epoch {epoch} complete.")
            print(f"Training loss: {train_losses[-1]}")
            print(f"Validation loss: {validation_losses[-1]}")

        return train_losses, validation_losses

    def export(self, path: str):
        """
        Export the model to the given path. This will output
        the model weights, config, and vocab.

        :param str path: path to export the model to
        """
        if not os.path.exists(path):
            os.makedirs(path)

        # save model
        torch.save(self.state_dict(), os.path.join(path, CHECKPOINT_FILE))

        # save config
        config = {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "dropout": self.dropout.p,
            "label_map": self.label_map,
        }

        # export vocab words
        with open(os.path.join(path, VOCAB_FILE), "w") as f:
            f.write("\n".join(self.tokenizer.token_to_id.keys()))

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
        # the final layer needs to be softmaxed
        prediction = nn.functional.softmax(prediction, dim=1)
        prediction = prediction.argmax(dim=1).item()
        prediction = self.label_map[prediction]

        return prediction
