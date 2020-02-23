import gym
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utilities import extractState, train, train_ewc, test
from CustomEnv import EmptyEnv
from DQN_Maze import DQN
from visualise import Visualizer

import numpy as np

NAME = "type2_right_down_ewc"

BATCH_SIZE = 64
GAMMA = 0.9
TARGET_UPDATE = 30
MEMORY_SIZE = 20_000
EPISODES = 150
CONSOLE_UPDATE_RATE = 50

env_right = EmptyEnv(size=8, goal_position=[6, 1])
env_right_reverse = EmptyEnv(size=8, goal_position=[1, 6], agent_start_pos=(6, 6))

env_down = EmptyEnv(size=8, goal_position=[1, 6])
env_diagonal = EmptyEnv(size=8, goal_position=[6, 6])
env_diagonal_reverse = EmptyEnv(size=8, goal_position=[1, 1], agent_start_pos=[6, 6])

env = env_right

state = env.reset()
env_shape = env.observation_space['image'].shape

# Turn Left, Turn Right, Move Forward
env_action_num = 3
env_state_num = env.getStateSize()

dqn = DQN(GAMMA, MEMORY_SIZE, TARGET_UPDATE, BATCH_SIZE, env_state_num, env_action_num)

visualizer = Visualizer()
episode_durations = []

train(dqn, env_right, episode_durations, 500, CONSOLE_UPDATE_RATE, visualizer, task=1)
test(dqn.eval_model, env_right)

# train(dqn, env_diagonal_reverse, episode_durations, 500, CONSOLE_UPDATE_RATE, visualizer)
train_ewc(dqn, env_down, env_diagonal, episode_durations, 500, CONSOLE_UPDATE_RATE, visualizer, task=2)
test(dqn.eval_model, env_right)
test(dqn.eval_model, env_down)

dqn.save(f"models/maze/maze_model_{NAME}.pth")
visualizer.save_plot(f"images/maze/maze_model_{NAME}.png")



