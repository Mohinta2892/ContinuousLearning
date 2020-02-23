import torch
import torch.nn as nn
import torch.nn.functional as F

from DQN_Maze import Net, DQN
from copy import deepcopy
import random

class EWC(object):
    def __init__(self, dqn: DQN, dataset: list, device='cpu'):

        self.dqn = dqn
        self.model = dqn.eval_model
        self.dataset = dataset
        self.device = device
        self.loss_func = nn.MSELoss()

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

        for state in self.dataset:

            self.model.zero_grad()

            output = self.dqn.eval_model(state)
            pred = self.dqn.target_model(state)
            
            loss = self.loss_func(output, pred)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        # precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()

        # TODO: return (self.lamda/2)*sum(losses)
        return loss