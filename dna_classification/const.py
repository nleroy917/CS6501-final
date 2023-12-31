import torch

CHECKPOINT_FILE = "checkpoint.pt"
VOCAB_FILE = "vocab.txt"
CONFIG_FILE = "config.yaml"

OPTIMIZER = torch.optim.Adam
LOSS_FN = torch.nn.CrossEntropyLoss

DEFAULT_NUM_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 0.001

DATA_FILE_DELIMITER = "\t"

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
