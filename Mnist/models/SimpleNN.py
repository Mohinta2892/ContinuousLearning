import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):

    def __init__(self, outputSize, hiddenSize=64):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, hiddenSize)
        self.fc2 = nn.Linear(hiddenSize, hiddenSize)
        self.fc3 = nn.Linear(hiddenSize, hiddenSize)
        self.fc4 = nn.Linear(hiddenSize, outputSize)
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)
    