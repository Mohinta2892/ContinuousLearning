import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

# Needs to overwrite: __len__ and __getitem__
class BinaryData(Dataset):

    def __init__(self, csv_path):
        self.label = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        sample = torch.tensor(self.label.iloc[index, 0:3]).int()
        label = torch.tensor(self.label.iloc[index, 3]).int()
        return sample, label

    def get_sample(self):
        return torch.tensor([self.label.iloc[x, 0:3] for x in range(4)])

# y = f(x) = x1 ^ x2
def getAndData(inputSize):
    x1 = np.random.randint(2, size=inputSize)
    x2 = np.random.randint(2, size=inputSize)

    y = x1 & x2
    return x1, x2, y

# y = f(x) = x1 ^ x2
def getOrData(inputSize):
    x1 = np.random.randint(2, size=inputSize)
    x2 = np.random.randint(2, size=inputSize)

    # y = f(x) = x1 ^ x2
    y = x1 | x2
    return x1, x2, y
