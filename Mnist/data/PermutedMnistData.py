import random
import torch
import numpy as np
from torchvision import datasets

import matplotlib.pyplot as plt

class PermutedMnistData(datasets.MNIST):

    def __init__(self, root="data/", train=True, permute_idx=None):
        super(PermutedMnistData, self).__init__(root, train, download=True)

        assert len(permute_idx) == 28*28

        self.data = torch.stack([img.float().view(-1)[permute_idx] for img in self.data])

        for i, img in enumerate(self.data):
            if i == 2: break;
            print (img)

            plt.imshow(img.view(28,28))
            plt.show()   

            plt.savefig('permutedMnist.png')     

            img = img.float().view(-1)
            img = img[permute_idx]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        return img, target

    def get_sample(self, sample_size):
        sample_idx = random.simple( range(len(self)), sample_size )
        return [img for img in self.data[sample_idx]]