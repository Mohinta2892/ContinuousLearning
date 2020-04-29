import torch
import torch.nn as nn
import torch.nn.functional as F

from DQN import Net, DQN
from copy import deepcopy
import random
from Transition import Transition

class SI(object):
    def __init__(self, dqn: DQN, c, epsilon, device='cpu'):
        self.dqn = dqn
        self.model = dqn.eval_model
        self.dataset = dqn.old_memory

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        
        self.SI_prev = {}
        self.SI_omega = {}

        # Register starting params
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.SI_prev[n] = p.data.clone()

        self.c = c 
        self.epsilon = epsilon

    def refresh_W(self):
        self.W = {}
        self.p_old = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.W[n] = p.data.clone().zero_()
                self.p_old[n] = p.data.clone()

    def update_W(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                if p.grad is not None:
                    self.W[n].add_(-p.grad*(p.detach()-self.p_old[n]))
                self.p_old[n] = p.detach().clone()

    def update_omega(self):
        for n, p in self.model.named_parameters():

            if p.requires_grad:
                n = n.replace('.', '__')

                # Find/calculate new values for quadratic penalty on parameters
                p_prev = self.SI_prev[n]
                p_current = p.detach().clone()
                p_change = p_current - p_prev

                omega_add = self.W[n] / (p_change**2 + self.epsilon)

                try:
                    omega = self.SI_omega[n]
                except KeyError:
                    omega = p.detach().clone().zero_()

                omega_new = omega + omega_add

                self.SI_prev[n] = p_current
                self.SI_omega[n] = omega_new

    def penalty(self):
        try:
            losses = []

            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    # Retrieve previous parameter values and their normalized path integral (i.e., omega)
                    n = n.replace('.', '__')
                    prev_values = self.SI_prev[n]
                    omega = self.SI_omega[n]

                    # Calculate SI's surrogate loss, sum over all parameters
                    losses.append((omega * (p-prev_values)**2).sum())

            return sum(losses)

        except KeyError:
            # SI-loss is 0 if there is no stored omega yet
            return torch.tensor(0., device=self._device())


