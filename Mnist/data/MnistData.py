import torch
import torchvision
from torchvision import transforms, datasets

import random
import numpy as np
import matplotlib.pyplot as plt


def getDataLoaders(targets, batch_size=10):
    train = datasets.MNIST('data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test = datasets.MNIST('data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    idx_train = np.isin(train.targets, targets)
    idx_test = np.isin(test.targets, targets)

    train = torch.utils.data.dataset.Subset(train, np.where(idx_train==True)[0])
    test = torch.utils.data.dataset.Subset(test, np.where(idx_test==True)[0])

    trainset = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    testset = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    # for data in trainset:
    #     x, y = data
    #     plt.imshow(x[0].view(28,28))
    #     plt.show()
    #     print(1)

    return trainset, testset

def getSamples(dataset, sample_size):
    sample_idx = random.sample(range(len(dataset)), sample_size)

    imgs = []
    for idx in sample_idx:
        imgs.append(dataset.dataset[idx])

    return imgs
