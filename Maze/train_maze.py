import gym
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utilities import extractState, train
from CustomEnv import EmptyEnv
from DQN_Maze import DQN
from visualise import Visualizer

import numpy as np

BATCH_SIZE = 32
GAMMA = 0.9
TARGET_UPDATE = 10
MEMORY_SIZE = 10_000
EPISODES = 200
CONSOLE_UPDATE_RATE = 1
# device = getDevice()

env_right = EmptyEnv(size=8, goal_position=[6, 1])
env_down = EmptyEnv(size=8, goal_position=[1, 6])
env_diagonal = EmptyEnv(size=8, goal_position=[6, 6])

env = env_right

state = env.reset()
env_shape = env.observation_space['image'].shape

# Turn Left, Turn Right, Move Forward
env_action_num = 3
env_state_num = env_shape[0] * env_shape[1] + 1

dqn = DQN(GAMMA, MEMORY_SIZE, TARGET_UPDATE, BATCH_SIZE, env_state_num, env_action_num)

visualizer = Visualizer()
episode_durations = []


train(dqn, env_right, episode_durations, EPISODES, CONSOLE_UPDATE_RATE, visualizer)
train(dqn, env_down, episode_durations, EPISODES, CONSOLE_UPDATE_RATE, visualizer)

dqn.save("maze_model.pth")
visualizer.save_plot("maze.png")



