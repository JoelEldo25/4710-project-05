import csv

import torch
from torch.utils.data import Dataset
from torch import tensor

class SleepDatasetCategorical(Dataset):
    def __init__(self, filename):
        reader = csv.reader(open(filename))
        self.data = []
        self.targets = []
        next(reader)
        for row in reader:
            datum = [row[2]] + [row[4]] + row[6:8] + [row[10]] + [float(row[11])/1000]
            target = [0,0,0,0,0,0,0,0,0,0]
            target[int(row[5])] = 1.0
            self.targets.append(target)
            for i in range(len(datum)):
                datum[i] = float(datum[i])
            self.data.append(datum)
        self.data = tensor(self.data)
        self.targets = tensor(self.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


