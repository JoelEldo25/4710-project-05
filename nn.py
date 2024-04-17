import torch
from torch import nn
from sleepdataset import SleepDataset
from torch.utils.data import DataLoader

dataset = SleepDataset("ss.csv")
class SleepNetwork(nn.Module):
    def __init__(self):
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(6,10),
            nn.ReLU(),
            nn.Linear(10,10),
            nn.ReLU,
            nn.Linear(10,1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits

def train(dataloader, model, loss_function, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_function(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            print(f"loss: {loss:>7f}  Batch {batch:>5d}")

def test(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
