import random
import torch
from torchvision import datasets

class PermutedMnistData(datasets.MNIST):

    def __init__(self, root="data/", train=True, permute_idx=None):
        super(PermutedMnistData, self).__init__(root, train, download=True)

        assert len(permute_idx) == 28*28

        self.data = torch.stack([img.float().view(-1)[permute_idx] // 255 for img in self.data])

    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]

        return img, target

    def get_sample(self, sample_size):
        sample_idx = random.simple( range(len(self)), sample_size )
        return [img for img in self.data[sample_idx]]