import torch
from torch import nn
from sleepdataset import SleepDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

batch_size=64
learning_rate=0.0001
epochs=10000
train_size=300
test_size=73

val_loss = []
train_loss = []
dataset = SleepDataset("data/ss.csv")
train_dataset,test_dataset=torch.utils.data.random_split(dataset,(train_size,test_size))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

class SleepNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(8,20, bias=True),
            nn.ReLU(),
            nn.Linear(20, 20, bias=True),
            nn.ReLU(),
            nn.Linear(20,1, bias=True),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits

def train(dataloader, model, loss_function, optimizer):
    model.train()
    avgloss = 0
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_function(pred, y)
        avgloss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avgloss /= batch_size
    train_loss.append(avgloss)


def test(dataloader, model, loss_fn, iter):
    model.eval()
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    val_loss.append(test_loss)
    if(iter+1) % 1000 == 0:
        print(f"Test Error: Avg loss: {test_loss:>8f} \n")


model = SleepNetwork()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(epochs):
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn, t)


line1, = plt.plot(val_loss, label='Validation Loss')
line2, = plt.plot(train_loss, label='Training Loss')
plt.legend(handles=[line1, line2])
plt.ylabel('Average Test Loss')
plt.yscale("log")
plt.xlabel('Epoch')
plt.show()