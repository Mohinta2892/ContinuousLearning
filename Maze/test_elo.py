import visualise
import gym
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utilities import extractState, train, test
from EmptyEnv import EmptyEnv
from EgoEnv import EgoEnv

from DQN import DQN, Net
from visualise import Visualizer

NAME = "ego_alo_stable"

# Init env
ego = EgoEnv(False)
alo = EgoEnv(True)

env_shape = ego.observation_space['image'].shape

num_of_actions = 3
num_of_states = ego.getStateSize()

# Init and load Net
net = Net(num_of_states, 200, num_of_actions)
# net.load_state_dict(torch.load("models/maze/maze_model_both.pth"))
net.load_state_dict(torch.load(f"models/{NAME}.pth"))
net.eval()

test(net, ego)
test(net, alo)

