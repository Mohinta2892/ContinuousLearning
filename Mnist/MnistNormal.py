from models.SimpleNN import SimpleNN
import os.path
import torch
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm
import random

from models.utilities import *
from data.MnistData import getDataLoaders

device = getDevice()
EPOCHS = 20

trainsetA, testsetA = getDataLoaders([0])
trainsetB, testsetB = getDataLoaders([1])
trainsetC, testsetC = getDataLoaders([2])
# trainsetD, testsetD = getDataLoaders([3])

model = SimpleNN(outputSize=4, hiddenSize=40).to(device)
optimizer = optim.SGD(model.parameters(), lr=1e-4)

# Normal
##############################################################

# Task A
loss = train(model, trainsetA, device)
accuracyA = test(model, testsetA, device)
print("Accuracy for task A: ", accuracyA)

# Task B
loss = train(model, trainsetB, device)
accuracyA = test(model, testsetA, device)
accuracyB = test(model, testsetB, device)
print("Accuracy for task A: ", accuracyA)
print("Accuracy for task B: ", accuracyB)

# Task C
loss = train(model, trainsetC, device)
accuracyA = test(model, testsetA, device)
accuracyB = test(model, testsetB, device)
accuracyC = test(model, testsetC, device)
print("Accuracy for task A: ", accuracyA)
print("Accuracy for task B: ", accuracyB)
print("Accuracy for task C: ", accuracyC)