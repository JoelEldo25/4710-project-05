import csv
from torch.utils.data import Dataset


class SleepDataset(Dataset):
    def __init__(self, file):
        reader = csv.reader(open('data.csv'))
        for row in reader:
            print(row)

