import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import random
import numpy as np

from ReplayMemory import ReplayMemory
from Transition import Transition

# HIDDEN_SIZE = 82
HIDDEN_SIZE = 200
EPSILON_MAX = 0.3
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DQN(object):

    def __init__(self, gamma, memory_size, target_update_counter, batch_size, num_of_states, num_of_actions, ewc_importance=28, si_importance=30):
        self.num_of_states = num_of_states
        self.num_of_actions = num_of_actions

        self.eval_model = Net(self.num_of_states, HIDDEN_SIZE, self.num_of_actions)
        self.target_model = Net(self.num_of_states, HIDDEN_SIZE, self.num_of_actions)

        self.optimizer = torch.optim.Adam(self.eval_model.parameters(), lr=0.001)
        # self.optimizer = torch.optim.SGD(self.eval_model.parameters(), lr=0.01)
        self.loss_func = nn.MSELoss()
        # self.loss_func = nn.MS
        # self.loss_func = nn.SmoothL1Loss()

        self.memory_size = memory_size
        self.memory = ReplayMemory(memory_size)
        self.old_memory = []
        self.learned_tasks = 0
        self.target_udpate_counter = target_update_counter
        self.learn_step_counter = 0
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = EPSILON_MAX
        self.ewc_importance = ewc_importance
        self.si_importance = si_importance

    def choose_action(self, state):
        # From np.array to torch
        state = torch.FloatTensor(state)

        sample = random.random()

        action = self.eval_model(state).argmax().view(1,1)

        if sample > self.epsilon:
            return action
        else:
            return torch.tensor([[random.randrange(self.num_of_actions)]], dtype=torch.long)

    def store_transition(self, state, action, reward, next_state):
        self.memory.push(state, action, next_state, reward)

    def decay_epsilon(self):
        self.epsilon *= EPSILON_DECAY
        self.epsilon = max(EPSILON_MIN, self.epsilon)
        # if(episode % 20 == 0):
        # print(f"Epsilon is {self.epsilon}")

    def reset_training(self):
        self.learn_step_counter = 0
        self.epsilon = EPSILON_MAX

        # Save a batch of old memories
        transitions = self.memory.sample(self.batch_size)
        self.old_memory = self.old_memory + transitions

        self.memory = ReplayMemory(self.memory_size)

    def learn(self, ewc=None, si=None):
        if len(self.memory) < self.batch_size:
            return

        # Should we update our target model?
        if self.learn_step_counter % self.target_udpate_counter == 0:
            self.target_model.load_state_dict(self.eval_model.state_dict())

        transitions = self.memory.sample(self.batch_size) 
        batch = Transition(*zip(*transitions))

        state_batch  = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        self.optimizer.zero_grad()

        q_eval = self.eval_model(state_batch).gather(1, action_batch)
        q_next = self.target_model(next_state_batch).detach().max(1)[0].view(self.batch_size, 1)

        new_q = (q_next * self.gamma) + reward_batch

        loss = self.loss_func(q_eval, new_q)
        if ewc is not None:
            penalty = ewc.penalty(self.eval_model)
            updated_loss = loss + self.ewc_importance * penalty
            # print(f"Loss: {loss}, Penalty: {penalty}, Updated loss: {updated_loss}")
            loss = updated_loss

        if si is not None:
            penalty = si.penalty()
            updated_loss = loss + self.si_importance * penalty
            # print(f"Loss: {loss}, Penalty: {penalty}, Updated loss: {updated_loss}")
            loss = updated_loss

        loss.backward()
        self.optimizer.step()
        self.learn_step_counter += 1

    def save(self, path):
        torch.save(self.eval_model.state_dict(), path)
