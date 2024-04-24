import torch
from torch import nn
from sleepdatasetcategory import SleepDatasetCategorical
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

batch_size=64
learning_rate=0.0001
epochs=10000
train_size=300
test_size=73

val_loss = []
train_loss = []
dataset = SleepDatasetCategorical("ss.csv")
train_dataset,test_dataset=torch.utils.data.random_split(dataset,(train_size,test_size))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


class SleepNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(8,20),
            nn.ReLU(),
            nn.Linear(20,20),
            nn.ReLU(),
            nn.Linear(20,10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits

def train(dataloader, model, loss_function, optimizer):
    model.train()
    correct = 0
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_function(pred, y)
        correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    correct /= size
    train_loss.append(1-correct)

def test(dataloader, model, loss_fn, iter):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    val_loss.append(1-correct)
    if (t+1) %1000 == 0:
        print(f"Test Error: Avg loss: {test_loss:>8f} \nAccuracy: {(100*correct):>0.1f}%")


model = SleepNetwork()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(epochs):
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn, t)

line1, = plt.plot(val_loss, label='Validation Loss')
line2, = plt.plot(train_loss, label='Training Loss')
plt.legend(handles=[line1, line2])
plt.ylabel('Inaccuracy')
plt.xlabel('Epoch')
plt.show()