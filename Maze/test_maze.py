import visualise
import gym
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utilities import extractState, train, test
from CustomEnv import EmptyEnv
from DQN_Maze import DQN, Net
from visualise import Visualizer


# Init env
env_right = EmptyEnv(size=8, goal_position=[6, 1])
env_down = EmptyEnv(size=8, goal_position=[1, 6])
env_diagonal = EmptyEnv(size=8, goal_position=[6, 6])

env = env_right
env_shape = env.observation_space['image'].shape

num_of_actions = 3
num_of_states = env_shape[0] * env_shape[1] + 1

# Init and load Net
net = Net(num_of_states, 200, num_of_actions)
net.load_state_dict(torch.load("models/maze/maze_model_combined_reverse.pth"))
net.eval()

test(net, env_right)
test(net, env_down)
test(net, env_diagonal)

