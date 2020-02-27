import gym
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utilities import extractState, train, train_ewc, test
from EmptyEnv import EmptyEnv
from DQN import DQN
from visualise import Visualizer

import numpy as np

NAME = "test"

BATCH_SIZE = 32
GAMMA = 0.9
TARGET_UPDATE = 25
MEMORY_SIZE = 10_000
EPISODES = 150
CONSOLE_UPDATE_RATE = 500

env_right = EmptyEnv(size=8, goal_position=[6, 1])
env_right_reverse = EmptyEnv(size=8, goal_position=[1, 6], agent_start_pos=(6, 6))
env_down = EmptyEnv(size=8, goal_position=[1, 6])
env_diagonal = EmptyEnv(size=8, goal_position=[6, 6])
env_diagonal_reverse = EmptyEnv(size=8, goal_position=[1, 1], agent_start_pos=[6, 6])

env = env_right

# Turn Left, Turn Right, Move Forward
env_action_num = 3
env_state_num = env.getStateSize()

dqn = DQN(GAMMA, MEMORY_SIZE, TARGET_UPDATE, BATCH_SIZE, env_state_num, env_action_num, ewc_importance=1000)

visualizer = Visualizer("DQN Training for Maze")
episode_durations = []

train(dqn, env_down, episode_durations, 200, 500, visualizer, task=1)
test(dqn.eval_model, env_down)

# train(dqn, env_right, episode_durations, 500, CONSOLE_UPDATE_RATE, visualizer, task=2)
train_ewc(dqn, env_right, env_down, episode_durations, 200, 10, visualizer, task=2)
test(dqn.eval_model, env_down)
test(dqn.eval_model, env_right)

dqn.save(f"models/{NAME}.pth")
visualizer.plot_durations(episode_durations)
visualizer.save_plot(f"images/{NAME}.png")



