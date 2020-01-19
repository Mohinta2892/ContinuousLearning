from models.SimpleNN import SimpleNN
import os.path
import torch
import torch.optim as optim
from torchvision import datasets

import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm
import random

from models.utilities import *
from data.MnistData import getDataLoaders
from data.PermutedMnistData import PermutedMnistData

device = getDevice()
EPOCHS = 10
batch_size = 128
hidden_size = 200
importance = 1000

def get_permute_mnist(tasks, batchSize):
    train_loader = []
    test_loader = []
    idx = list(range(28*28))

    for i in range(tasks):
        train_loader.append(torch.utils.data.DataLoader(
            PermutedMnistData(train=True, permute_idx=idx),
            batch_size=batchSize
        ))

        test_loader.append(torch.utils.data.DataLoader(
            PermutedMnistData(train=False, permute_idx=idx),
            batch_size=batchSize
        ))

        random.shuffle(idx)       

    return train_loader, test_loader               

# train_loader, test_loader = get_permute_mnist(5, batch_size)

# model = SimpleNN(outputSize=10, hiddenSize=hidden_size).to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# # optimizer = optim.RMSprop(model.parameters(), lr=1e-3)

# # EWC
# ##############################################################

# # Task A
# lossA, accA = train_and_test(model, train_loader[0], test_loader[0:1], device, 1, [], optimizer, EPOCHS)

# # Task B
# lossB, accB = train_and_test(model, train_loader[1], test_loader[0:2], device, 2, train_loader[0:1], optimizer, EPOCHS, importance=importance)

# # Task C
# lossC, accC = train_and_test(model, train_loader[2], test_loader[0:3], device, 3, train_loader[0:2], optimizer, EPOCHS, importance=importance)

# # Task D
# lossD, accD = train_and_test(model, train_loader[3], test_loader[0:4], device, 4, train_loader[0:3], optimizer, EPOCHS, importance=importance)

# # Task E
# lossE, accE = train_and_test(model, train_loader[4], test_loader[0:5], device, 5, train_loader[0:4], optimizer, EPOCHS, importance=importance)

# loss = np.array([lossA, lossB, lossC, lossD, lossE])
# acc = np.array([accA[0] + accB[0] + accC[0] + accD[0] + accE[0], accB[1] + accC[1] + accD[1] + accE[1], accC[2] + accD[2] + accE[2], accD[3] + accE[3], accE[4]])
# np.save("loss.npy", loss)
# np.save("acc.npy", acc)

loss = np.load("loss.npy", allow_pickle=True)
acc = np.load("acc.npy", allow_pickle=True)

loss = loss.reshape(-1)

fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(121)
ax1.title.set_text('Loss')
ax1.set_xlabel('time')
ax1.set_ylabel('loss')
ax1.grid()

ax2 = fig.add_subplot(122)
ax2.title.set_text('Accuracy')
ax2.set_xlabel('time')
ax2.set_ylabel('accuracy')
ax2.set_ylim([0, 1])
ax2.grid()

ax1.plot(np.arange(loss.shape[0]/5 * 3), loss[0: 10*3])

size = len(acc[0])

ax2.plot(np.arange(size), acc[0], label='task1')
ax2.plot(np.arange(size - len(acc[1]), size), acc[1], label='task2')
ax2.plot(np.arange(size - len(acc[2]), size), acc[2], label='task3')
# ax2.plot(np.arange(size - len(acc[3]), size), acc[3], label='task4')
# ax2.plot(np.arange(size - len(acc[4]), size), acc[4], label='task5')
ax2.legend(loc="lower left")

plt.savefig('loss_permuted.png')
