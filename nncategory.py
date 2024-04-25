import torch
from torch import nn
from sleepdatasetcategory import SleepDatasetCategorical
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import statistics

batch_size=64
learning_rate=0.001
epochs = 4001
train_size=300
test_size=73
times = int(sys.argv[1]) if len(sys.argv) > 1 else 1

labels = ["Gender", "Age", "Sleep Duration","Physical Activity Level", "Stress Level","BMI Category","Blood Pressure","Heart Rate","Daily Steps","Sleep Disorder"]
x_vals = [i+1 for i in range(0, epochs, 10)]
val_acc = []
train_acc = []
val_acc_agg = []
train_acc_agg = []
val_loss = []
train_loss = []
val_agg = []
train_agg = []
cols_acc = [[],[],[],[],[],[],[],[],[],[]]
cols_loss = [[],[],[],[],[],[],[],[],[],[]]



class SleepNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(10,20),
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
    return test_loss, 1-correct

dataset = SleepDatasetCategorical("ss.csv")
for i in range(times):
    train_dataset,test_dataset=torch.utils.data.random_split(dataset,(train_size,test_size))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = SleepNetwork()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        train(train_dataloader, model, loss_fn, optimizer, t)
        if t%10 == 0:
            loss, inacc = test(test_dataloader, model, loss_fn, t)
            val_loss.append(loss)
            val_acc.append(inacc)

    val_agg.append(val_loss)
    val_acc_agg.append(val_acc)
    val_loss = []
    val_acc = []
    train_agg.append(train_loss)
    train_acc_agg.append(train_acc)
    train_loss = []
    train_acc = []
    for i in range(10):
        dataset.shuffle(i)
        var_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        loss,acc = test(var_dataloader, model, loss_fn, 0)
        cols_acc[i].append(acc)
        cols_loss[i].append(loss)
        dataset.reset()

    val_loss = []
    val_acc = []
    print("Done with iteration",i+1)


val_acc = [statistics.mean([val_acc_agg[time][epoch] for time in range(times)]) for epoch in range(int(epochs/10+1))]
train_acc = [statistics.mean([train_acc_agg[time][epoch] for time in range(times)]) for epoch in range(int(epochs/10+1))]

line1, = plt.plot(x_vals, val_acc, label='Validation Inaccuracy')
line2, = plt.plot(x_vals, train_acc, label='Training Inaccuracy')
plt.legend(handles=[line1, line2])
plt.ylabel('% incorrect')
plt.xlabel('Epoch')
plt.show()

if times > 1:
    val_acc = [statistics.stdev([val_acc_agg[time][epoch] for time in range(times)]) for epoch in range(int(epochs/10+1))]
    train_acc = [statistics.stdev([train_acc_agg[time][epoch] for time in range(times)]) for epoch in range(int(epochs/10+1))]

    line1, = plt.plot(x_vals, val_acc, label='Validation Inaccuracy Stdev')
    line2, = plt.plot(x_vals, train_acc, label='Training Inaccuracy Stdev')
    plt.legend(handles=[line1, line2])
    plt.ylabel('% incorrect stdev')
    plt.xlabel('Epoch')
    plt.show()



val_loss = [statistics.mean([val_agg[time][epoch] for time in range(times)]) for epoch in range(int(epochs/10+1))]
train_loss = [statistics.mean([train_agg[time][epoch] for time in range(times)]) for epoch in range(int(epochs/10+1))]

line3, = plt.plot(x_vals, val_loss, label='Validation Loss')
line4, = plt.plot(x_vals, train_loss, label='Training Loss')
plt.legend(handles=[line3, line4])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

if times > 1:
    val_loss = [statistics.stdev([val_agg[time][epoch] for time in range(times)]) for epoch in range(int(epochs / 10 + 1))]
    train_loss = [statistics.stdev([train_agg[time][epoch] for time in range(times)]) for epoch in range(int(epochs / 10 + 1))]
    line3, = plt.plot(x_vals, val_loss, label='Validation Loss Stdev')
    line4, = plt.plot(x_vals, train_loss, label='Training Loss')
    plt.legend(handles=[line3, line4])
    plt.ylabel('Loss Stdev')
    plt.xlabel('Epoch')
    plt.show()

for i in range(10):
    print("Column",labels[i])
    print("Mean Loss -",statistics.mean(cols_loss[i]))
    if(times > 1):
        print("Stdev of Loss -",statistics.stdev(cols_loss[i]))
    print("Mean accuracy -",statistics.mean(cols_acc[i]))
    if (times > 1):
        print("Stdev of accuracy -",statistics.stdev(cols_acc[i]))
    print()



