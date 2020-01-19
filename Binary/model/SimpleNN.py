import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):

    def __init__(self, inputSize, hiddenLayer, outputSize):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(inputSize, hiddenLayer)
        self.fc2 = nn.Linear(hiddenLayer, outputSize)


    def forward(self, x):
        x = F.relu(self.fc1(x.float()))
        x = self.fc2(x)
        return x