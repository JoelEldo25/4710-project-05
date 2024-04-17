import csv
from torch.utils.data import Dataset


class SleepDataset(Dataset):
    def __init__(self, filename):
        reader = csv.reader(open(filename))
        self.data = []
        for row in reader:
            self.data.append(row)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attributes = self.data[index][:5] + self.data[index][6:]
        return attributes, self.data[index][5]

