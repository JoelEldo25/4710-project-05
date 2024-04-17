import csv
from torch.utils.data import Dataset


class SleepDataset(Dataset):
    def __init__(self, filename):
        reader = csv.reader(open(filename))
        self.data = []
        next(reader)
        for row in reader:
            datum = [row[2]] + row[4:8] + row[10:12]
            for i in range(len(datum)):
                datum[i] = float(datum[i])
            self.data.append(datum)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attributes = self.data[index][:2] + self.data[index][3:]
        return attributes, self.data[index][2]
