import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):

    def __init__(self, inputSize, hiddenLayer, outputSize):
        super(SimpleNN, self).__init__()
        
        self.emb = nn.Embedding(3,2)
        self._emb_init(self.emb)

        self.fc1 = nn.Linear(inputSize, hiddenLayer)
        self.fc2 = nn.Linear(hiddenLayer, outputSize)


    def forward(self, x):
        # x = torch.sum(x, dim=1)
        # x = self.emb(x)
        x = F.relu(self.fc1(x.float()))         # Activation on the hidden layer
        x = self.fc2(x)                 # No activation on the output layer

        return F.log_softmax(x, dim=1)  # Return either a 0 or 1

    def _emb_init(self, x):
        x = x.weight.data
        sc = 2/(x.size(1) + 1)
        x.uniform_(-sc,sc)