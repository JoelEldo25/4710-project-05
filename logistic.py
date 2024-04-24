import torch
import os
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset


class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        return out

input_size = 4
model = LogisticRegression(input_size)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

file_path = os.path.abspath('ss.csv')
data = pd.read_csv(file_path)
X = data[['Age', 'Sleep Duration', 'Physical Activity Level', 'Stress Level']].values
y = data['Quality of Sleep'].values


X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1,1)

dataset = TensorDataset(X_tensor, y_tensor)
batchsize = 64
dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)

input_size = X_tensor.shape[1]
model = LogisticRegression(input_size)

num_epochs = 100
for epoch in range(num_epochs):
    for input, labels in dataloader:
        outputs = model(input)
        scaled_outputs = 10 * outputs
        discrete_outputs = torch.round(scaled_outputs)
        loss = criterion(discrete_outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')