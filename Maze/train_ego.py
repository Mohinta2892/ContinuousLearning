import gym
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utilities import extractState, train, train_ewc, test
from EmptyEnv import EmptyEnv
from EgoEnv import EgoEnv
from DQN import DQN
from visualise import Visualizer

import numpy as np

NAME = "test"

BATCH_SIZE = 64
GAMMA = 0.9
TARGET_UPDATE = 25
MEMORY_SIZE = 15_000
EPISODES = 150
CONSOLE_UPDATE_RATE = 500

env_ego = EgoEnv(False)
env_alo = EgoEnv(True)

# Turn Left, Turn Right, Move Forward
env_action_num = 3
env_state_num = env_ego.getStateSize()

dqn = DQN(GAMMA, MEMORY_SIZE, TARGET_UPDATE, BATCH_SIZE, env_state_num, env_action_num, ewc_importance=1500)

visualizer = Visualizer("DQN Training for Maze")
episode_durations = []

train(dqn, env_ego, episode_durations, 200, 100, visualizer, task=1)
test(dqn.eval_model, env_ego)

# train(dqn, env_right, episode_durations, 500, CONSOLE_UPDATE_RATE, visualizer, task=2)
train_ewc(dqn, env_alo, env_ego, episode_durations, 200, 10, visualizer, task=2)
test(dqn.eval_model, env_ego)
test(dqn.eval_model, env_alo)

dqn.save(f"models/{NAME}.pth")
visualizer.plot_durations(episode_durations)
visualizer.save_plot(f"images/{NAME}.png")



