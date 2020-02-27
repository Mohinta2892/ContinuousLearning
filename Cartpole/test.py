import visualise
import gym
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from DQN import DQN
from DQN import Net


# Init env
env = gym.make('CartPole-v0')
num_of_states = env.observation_space.shape[0]
num_of_actions = env.action_space.n

# Init and load Net
net = Net(num_of_states, 200, num_of_actions)
net.load_state_dict(torch.load("models/eval_model_stable_short_smart.pth"))
net.eval()


state = env.reset()
state = torch.FloatTensor(state).view(-1, 4)

while True:
    env.render()
    action = net(state).argmax().item()
    next_state, reward, done, info = env.step(action)

    state = torch.FloatTensor(next_state).view(-1, 4)


