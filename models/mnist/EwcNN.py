import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
import random

class EWC(object):
    def __init__(self, model: nn.Module, dataset: list, device='cpu'):

        self.model = model
        self.dataset = dataset
        self.device = device

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = p.data

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data

        self.model.eval()

        for input in self.dataset:
            X, y = input               
            X, y = X.to(self.device), torch.tensor([y]).to(self.device)

            self.model.zero_grad()

            output = self.model(X.view(-1,784))
            pred = output.max(1)[-1]
            # loss = F.binary_cross_entropy_with_logits(output, label)
            loss = F.nll_loss(output, pred)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()

        # TODO: return (self.lamda/2)*sum(losses)
        return loss