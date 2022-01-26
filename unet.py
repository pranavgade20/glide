import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
