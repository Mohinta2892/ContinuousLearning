import visualise
import gym
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utilities import extractState, train
from CustomEnv import EmptyEnv
from DQN_Maze import DQN
from visualise import Visualizer


# Init env
env_right = EmptyEnv(size=8, goal_position=[6, 1])
env_down = EmptyEnv(size=8, goal_position=[1, 6])
env_diagonal = EmptyEnv(size=8, goal_position=[6, 6])

env = env_right
env_shape = env.observation_space['image'].shape

env_action_num = 3
env_state_num = env_shape[0] * env_shape[1] + 1

# Init and load Net
net = Net(num_of_states, 200, num_of_actions)
net.load_state_dict(torch.load("models/cartpole/eval_model_stable_short_smart.pth"))
net.eval()


state = env.reset()
state = torch.FloatTensor(state).view(-1, 4)

while True:
    env.render()
    action = net(state).argmax().item()
    next_state, reward, done, info = env.step(action)

    state = torch.FloatTensor(next_state).view(-1, 4)
