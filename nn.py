import torch
from torch import nn
from sleepdataset import SleepDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

batch_size=64
learning_rate=0.001
epochs=4001
train_size=300
test_size=73

x_vals = []
val_loss = []
train_loss = []
dataset = SleepDataset("data/ss.csv")


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

def train(dataloader, model, loss_function, optimizer, iter):
    model.train()
    if iter%10 == 0:
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
    else:
        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
            loss = loss_function(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


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
    x_vals.append(iter + 1)
    if iter % 1000 == 0:
        print(f"Test Error: Avg loss: {test_loss:>8f} \n")


train_dataset,test_dataset=torch.utils.data.random_split(dataset,(train_size,test_size))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = SleepNetwork()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(epochs):
    train(train_dataloader, model, loss_fn, optimizer, t)
    if t % 10 == 0:
        test(test_dataloader, model, loss_fn, t)

line1, = plt.plot(x_vals, val_loss, label='Validation Loss')
line2, = plt.plot(x_vals, train_loss, label='Training Loss')
plt.legend(handles=[line1, line2])
plt.ylabel('Average Test Loss')
plt.yscale("log")
plt.xlabel('Epoch')
plt.show()

print("Full data")
full_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test(full_dataloader, model, loss_fn, 0)

for i in range(8):
    dataset = SleepDataset("data/ss.csv")
    dataset.shuffle(i)
    var_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("Column",i)
    test(var_dataloader, model, loss_fn, 0)



