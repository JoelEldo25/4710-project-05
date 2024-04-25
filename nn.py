import torch
import sys
from torch import nn
from sleepdataset import SleepDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import statistics

batch_size=64
learning_rate=0.001
epochs=4001
train_size=300
test_size=73
times = int(sys.argv[1]) if len(sys.argv) > 1 else 1

labels = ["Gender", "Age", "Sleep Duration","Physical Activity Level", "Stress Level","BMI Category","Blood Pressure","Heart Rate","Daily Steps","Sleep Disorder"]
x_vals = [i+1 for i in range(0, epochs, 10)]
val_loss = []
train_loss = []
val_agg = []
train_agg = []
cols = [[],[],[],[],[],[],[],[],[],[]]
dataset = SleepDataset("data/ss.csv")


class SleepNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(10,20, bias=True),
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
    return test_loss


for i in range(times):
    train_dataset,test_dataset=torch.utils.data.random_split(dataset,(train_size,test_size))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = SleepNetwork()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        train(train_dataloader, model, loss_fn, optimizer, t)
        if t % 10 == 0:
            val_loss.append(test(test_dataloader, model, loss_fn, t))

    val_agg.append(val_loss)
    train_agg.append(train_loss)
    val_loss = []
    train_loss = []

    for i in range(10):
        dataset.shuffle(i)
        var_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        loss = test(var_dataloader, model, loss_fn, 1)
        cols[i].append(loss)
        dataset.reset()

    val_loss = []
    train_loss = []
    print("Done with iteration", i + 1)

val_loss = [statistics.mean([val_agg[time][epoch] for time in range(times)]) for epoch in range(int(epochs/10+1))]
train_loss = [statistics.mean([train_agg[time][epoch] for time in range(times)]) for epoch in range(int(epochs/10+1))]

line1, = plt.plot(x_vals, val_loss, label='Validation Loss')
line2, = plt.plot(x_vals, train_loss, label='Training Loss')
plt.legend(handles=[line1, line2])
plt.ylabel('Average Loss')
plt.yscale("log")
plt.xlabel('Epoch')
plt.show()

if times > 1:
    val_stdev = [statistics.stdev([val_agg[time][epoch] for time in range(times)]) for epoch in range(int(epochs/10+1))]
    train_stdev = [statistics.stdev([train_agg[time][epoch] for time in range(times)]) for epoch in range(int(epochs/10+1))]
    line1, = plt.plot(x_vals, val_stdev, label='Validation Loss stdev')
    line2, = plt.plot(x_vals, train_stdev, label='Training Loss stdev')
    plt.legend(handles=[line1, line2])
    plt.ylabel('Stdev of average losses')
    plt.yscale("log")
    plt.xlabel('Epoch')
    plt.show()

for i in range(10):
    print("Column",i)
    print("Mean Loss for shuffled",labels[i],"-",statistics.mean(cols[i]))
    if(times > 1):
        print("Stdev of Loss for column",i,"-",statistics.stdev(cols[i]))
        print("Stdev of Loss for column",i,"-",statistics.stdev(cols[i]))


