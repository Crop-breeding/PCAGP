import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Training Parameters
BATCH_SIZE = 32
LR = 0.0001
WEIGHT_DECAY = 1e-4
EPOCHS = 200
