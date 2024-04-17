import torch
from torch import nn
from torch.utils.data import DataLoader

class SleepNetwork(nn.Module):
    def __init__(self):
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(10,10),
            nn.ReLU(),
            nn.Linear(10,10),
            nn.ReLU,
            nn.Linear(10,10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits