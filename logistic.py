import torch
import os
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sleepdataset import SleepDataset

class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        return out


file_path = os.path.abspath('ss.csv')
dataset = SleepDataset(file_path)
batchsize = 64
input_size = len(dataset[0][0])

dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)

model = LogisticRegression(input_size)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

num_epochs = 10000
for epoch in range(num_epochs):
    for input, labels in dataloader:
        outputs = model(input)
        scaled_outputs = 9 * outputs + 1
        discrete_outputs = torch.round(scaled_outputs)
        loss = criterion(discrete_outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if(epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')