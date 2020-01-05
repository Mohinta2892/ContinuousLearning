from models.mnist.SimpleNN import SimpleNN
import os.path
import torch
import torch.optim as optim

from tqdm import tqdm
import random

from models.mnist.utilities import *
from data.MnistData import getDataLoaders

RETRAIN_MODEL = False

device = getDevice()
EPOCHS = 3

trainsetA, testsetA = getDataLoaders([0])
trainsetB, testsetB = getDataLoaders([1])
trainsetC, testsetC = getDataLoaders([2])
# trainsetD, testsetD = getDataLoaders([3])

model = SimpleNN(outputSize=4, hiddenSize=40).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.7e-3)
# optimizer = optim.RMSprop(model.parameters(), lr=1e-3)

# EWC
##############################################################

# Task A
loss = train_ewc(model, trainsetA, device, 1, [], optimizer, EPOCHS=EPOCHS)
accuracyA = test(model, testsetA, device)
print("Accuracy for task A: ", accuracyA)

# Task B
loss = train_ewc(model, trainsetB, device, 2, [trainsetA], optimizer, EPOCHS=EPOCHS)
accuracyA = test(model, testsetA, device)
accuracyB = test(model, testsetB, device)
print("Accuracy for task A: ", accuracyA)
print("Accuracy for task B: ", accuracyB)

# Task C
loss = train_ewc(model, trainsetC, device, 3, [trainsetA, trainsetB], optimizer, EPOCHS=EPOCHS)
accuracyA = test(model, testsetA, device)
accuracyB = test(model, testsetB, device)
accuracyC = test(model, testsetC, device)
print("Accuracy for task A: ", accuracyA)
print("Accuracy for task B: ", accuracyB)
print("Accuracy for task C: ", accuracyC)


# Normal
##############################################################

# # Task A
# loss = train(model, trainsetA, device)
# accuracyA = test(model, testsetA, device)
# print("Accuracy for task A: ", accuracyA)

# # Task B
# loss = train(model, trainsetB, device)
# accuracyA = test(model, testsetA, device)
# accuracyB = test(model, testsetB, device)
# print("Accuracy for task A: ", accuracyA)
# print("Accuracy for task B: ", accuracyB)

# # Task C
# loss = train(model, trainsetC, device)
# accuracyA = test(model, testsetA, device)
# accuracyB = test(model, testsetB, device)
# accuracyC = test(model, testsetC, device)
# print("Accuracy for task A: ", accuracyA)
# print("Accuracy for task B: ", accuracyB)
# print("Accuracy for task C: ", accuracyC)