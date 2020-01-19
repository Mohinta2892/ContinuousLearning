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
from data.PermutedMnistData import PermutedMnistData

device = getDevice()
EPOCHS = 50

def get_permute_mnist(tasks, batchSize):
    train_loader = {}
    test_loader = {}
    idx = list(range(28*28))

    for i in range(tasks):
        train_loader[i] = torch.utils.data.DataLoader(
            PermutedMnistData(train=True, permute_idx=idx),
            batch_size=batchSize,
            num_workers=4
        )

        test_loader[i] = torch.utils.data.DataLoader(
            PermutedMnistData(train=False, permute_idx=idx),
            batch_size=batchSize
        )

        random.shuffle(idx)       

    return train_loader, test_loader               


train_loader, test_loader = get_permute_mnist(5, 128)

model = SimpleNN(outputSize=10, hiddenSize=200).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# optimizer = optim.RMSprop(model.parameters(), lr=1e-3)

# EWC
##############################################################

# Task A
lossA, accA = train_and_test(model, train_loader[0], [test_loader[0]], device, 1, [], optimizer, EPOCHS)

# # Task B
# lossB, accB = train_and_test(model, trainsetB, [testsetA, testsetB], device, 2, [trainsetA], optimizer, EPOCHS, importance=3000)

# # Task C
# lossC, accC = train_and_test(model, trainsetC, [testsetA, testsetB, testsetC], device, 3, [trainsetA, trainsetB], optimizer, EPOCHS, importance=4000)

# loss = np.array([lossA, lossB, lossC])
# acc = np.array([accA[0] + accB[0] + accC[0], accB[1] + accC[1], accC[2]])
# np.save("loss.npy", loss)
# np.save("acc.npy", acc)

# loss = np.load("loss.npy", allow_pickle=True)
# acc = np.load("acc.npy", allow_pickle=True)

# loss = loss.reshape(-1)

# fig = plt.figure(figsize=(12,5))
# ax1 = fig.add_subplot(121)
# ax1.title.set_text('Loss')
# ax1.set_xlabel('time')
# ax1.set_ylabel('loss')

# ax2 = fig.add_subplot(122)
# ax2.title.set_text('Accuracy')
# ax2.set_xlabel('time')
# ax2.set_ylabel('accuracy')

# ax1.plot(np.arange(loss.shape[0]), loss)

# size = len(acc[0])

# ax2.plot(np.arange(size), acc[0], label='task1')
# ax2.plot(np.arange(size - len(acc[1]), size), acc[1], label='task2')
# ax2.plot(np.arange(size - len(acc[2]), size), acc[2], label='task3')
# ax2.legend(loc="lower left")

# plt.savefig('loss_single.png')
