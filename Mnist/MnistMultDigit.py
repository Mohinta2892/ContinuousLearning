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
EPOCHS = 5

trainsetA, testsetA = getDataLoaders([0,2])
trainsetB, testsetB = getDataLoaders([1,3])
# trainsetC, testsetC = getDataLoaders([2])
# trainsetD, testsetD = getDataLoaders([3])

model = SimpleNN(outputSize=4, hiddenSize=40).to(device)
optimizer = optim.SGD(model.parameters(), lr=7e-4)
# optimizer = optim.RMSprop(model.parameters(), lr=3e-4)

# EWC
##############################################################

# Task A
lossA, accA = train_and_test(model, trainsetA, [testsetA], device, 1, [], optimizer, EPOCHS)

# Task B
lossB, accB = train_and_test(model, trainsetB, [testsetA, testsetB], device, 2, [trainsetA], optimizer, EPOCHS, importance=5000)

# Task A again
lossC, accC = train_and_test(model, trainsetA, [testsetA, testsetB], device, 2, [trainsetA, trainsetB], optimizer, EPOCHS, importance=5000)

# Task C
# lossC, accC = train_and_test(model, trainsetC, [testsetA, testsetB, testsetC], device, 3, [trainsetA, trainsetB], optimizer, EPOCHS)

# loss = np.array([lossA, lossB, lossC])
# acc = np.array([accA[0] + accB[0] + accC[0], accB[1] + accC[1], accC[2]])

loss = np.array([lossA, lossB])
acc = np.array([accA[0] + accB[0], accB[1]])

np.save("loss.npy", loss)
np.save("acc.npy", acc)

# loss = np.load("loss.npy", allow_pickle=True)
# acc = np.load("acc.npy", allow_pickle=True)

loss = loss.reshape(-1)

fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(121)
ax1.title.set_text('Loss')
ax1.set_xlabel('time')
ax1.set_ylabel('loss')

ax2 = fig.add_subplot(122)
ax2.title.set_text('Accuracy')
ax2.set_xlabel('time')
ax2.set_ylabel('accuracy')

ax1.plot(np.arange(loss.shape[0]), loss)

size = len(acc[0])

ax2.plot(np.arange(size), acc[0], label='task1')
ax2.plot(np.arange(size - len(acc[1]), size), acc[1], label='task2')
# ax2.plot(np.arange(size - len(acc[2]), size), acc[2], label='task3')
ax2.legend(loc="lower left")

plt.savefig('loss_multiple.png')
