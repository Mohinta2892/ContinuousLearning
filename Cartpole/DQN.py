import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import random
import numpy as np

from ReplayMemory import ReplayMemory
from Transition import Transition

HIDDEN_SIZE = 200

EPSILON_MAX = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.99

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

    def __init__(self, gamma, memory_size, target_update_counter, batch_size, env):
        self.num_of_states = env.observation_space.shape[0]
        self.num_of_actions = env.action_space.n

        self.eval_model = Net(self.num_of_states, HIDDEN_SIZE, self.num_of_actions)
        self.target_model = Net(self.num_of_states, HIDDEN_SIZE, self.num_of_actions)

        self.optimizer = torch.optim.Adam(self.eval_model.parameters(), lr=0.01)
        self.loss_func = nn.MSELoss()

        self.memory_size = memory_size
        self.memory = ReplayMemory(memory_size)
        self.target_udpate_counter = target_update_counter
        self.learn_step_counter = 0
        self.steps_done = 0
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = EPSILON_MAX
        self.env = env

    def choose_action(self, state):
        # From np.array to torch
        state = torch.FloatTensor(state)

        sample = random.random()

        # eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)

        if sample > self.epsilon:
            return self.eval_model(state).argmax().view(1,1)
        else:
            return torch.tensor([[random.randrange(self.num_of_actions)]], dtype=torch.long)

    def store_transition(self, state, action, reward, next_state):
        self.memory.push(state, action, next_state, reward)

    def decay_epsilon(self):
        self.epsilon *= EPSILON_DECAY
        self.epsilon = max(EPSILON_MIN, self.epsilon)
        # print(f"Epsilon is {self.epsilon}")

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        # Should we update our target model?
        if self.learn_step_counter % self.target_udpate_counter == 0:
            self.target_model.load_state_dict(self.eval_model.state_dict())

        self.learn_step_counter += 1

        # Gives 128 transitions which we combine into one
        transitions = self.memory.sample(self.batch_size) # transitions.len = 128
        batch = Transition(*zip(*transitions))  # batch = transition, each element.len = 128

        state_batch  = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        q_eval = self.eval_model(state_batch).gather(1, action_batch)
        q_next = self.target_model(next_state_batch).detach().max(1)[0].view(self.batch_size, 1)

        new_q = (q_next * self.gamma) + reward_batch

        loss = self.loss_func(q_eval, new_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.eval_model.state_dict(), path)
