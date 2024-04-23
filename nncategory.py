import torch
from torch import nn
from sleepdatasetcategory import SleepDatasetCategorical
from torch.utils.data import DataLoader

batch_size=64
learning_rate=0.0001
epochs=10000
train_size=300
test_size=73

dataset = SleepDatasetCategorical("ss.csv")
train_dataset,test_dataset=torch.utils.data.random_split(dataset,(train_size,test_size))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

class SleepNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(6,20),
            nn.ReLU(),
            nn.Linear(20,20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20,10),
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
        #print(f"loss: {loss:>7f}  Batch {batch:>5d}")

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
    print(f"Test Error: Avg loss: {test_loss:>8f} \nAccuracy: {(100*correct):>0.1f}%")


model = SleepNetwork()
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(epochs):
    #print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    if (t+1) % 1000 == 0:
        test(test_dataloader, model, loss_fn, t)

print("Done!")