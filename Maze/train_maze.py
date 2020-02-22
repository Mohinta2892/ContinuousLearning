import gym
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utilities import extractState, train
from CustomEnv import EmptyEnv
from DQN_Maze import DQN, Net
from visualise import Visualizer

import numpy as np

EPISODES = 100

env_right = EmptyEnv(size=8, goal_position=[6, 1])
env_down = EmptyEnv(size=8, goal_position=[1, 6])
env_diagonal = EmptyEnv(size=8, goal_position=[6, 6])

env = env_right
env_shape = env.observation_space['image'].shape

# Turn Left, Turn Right, Move Forward
env_action_num = 3
env_state_num = env_shape[0] * env_shape[1] + 1

net = Net(env_state_num, 200, env_action_num)
net.load_state_dict(torch.load("models/maze/maze_model_both.pth"))
net.eval()

visualizer = Visualizer()
episode_durations = []

for episode in range(EPISODES):

    state = env.reset()
    state = extractState(state)
    steps = 0

    while True:
        env.render()
        action = net(state).argmax().item()
        next_state, reward, done, info = env.step(action)

        if done:
            break;

        state = extractState(next_state)
        steps += 1

    episode_durations.append(steps)
    visualizer.plot_durations(episode_durations)

visualizer.save_plot("maze.png")



