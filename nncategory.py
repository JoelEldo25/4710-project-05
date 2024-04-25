import torch
from torch import nn
from sleepdatasetcategory import SleepDatasetCategorical
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

batch_size=64
learning_rate=0.001
epochs = 4001
train_size=300
test_size=73

x_vals = []
val_acc = []
train_acc = []
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

def train(dataloader, model, loss_function, optimizer, iter):
    model.train()
    if iter % 10 == 0:
        correct = 0
        avgloss = 0
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
            loss = loss_function(pred, y)
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            avgloss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        correct /= size
        avgloss /= num_batches
        train_loss.append(avgloss)
        train_acc.append(1-correct)
    else:
        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
            loss = loss_function(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


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
    val_acc.append(1-correct)
    val_loss.append(test_loss)
    x_vals.append(iter+1)
    if t %1000 == 0:
        print(f"Test Error: Avg loss: {test_loss:>8f} \nAccuracy: {(100*correct):>0.1f}%")


model = SleepNetwork()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(epochs):
    train(train_dataloader, model, loss_fn, optimizer, t)
    if t%10 == 0:
        test(test_dataloader, model, loss_fn, t)

fig, (ax1, ax2)  = plt.subplots(2)
line1, = ax1.plot(x_vals, val_acc, label='Validation Inaccuracy')
line2, = ax1.plot(x_vals, train_acc, label='Training Inaccuracy')
plt.legend(handles=[line1, line2])
plt.ylabel('Inaccuracy')
plt.xlabel('Epoch')

line3, = ax2.plot(x_vals, val_loss, label='Validation Loss')
line4, = ax2.plot(x_vals, train_loss, label='Training Loss')
plt.legend(handles=[line3, line4])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()