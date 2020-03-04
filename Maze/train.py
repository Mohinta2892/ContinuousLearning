import gym
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utilities import train, test
from environments.EmptyEnv import EmptyEnv
from environments.CrossEnv import CrossEnv
from DQN import DQN
from visualise import visualise_data

import numpy as np

NAME = "test"

BATCH_SIZE = 32
GAMMA = 0.9
TARGET_UPDATE = 25
MEMORY_SIZE = 10_000
EPISODES = 150
DISPLAY_FREQUENCY = 50
TEST_FREQUENCY = 2

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

episode_durations = []
test_durations = []

ep_dur, test_dur = train(dqn, [env_down], 200, TEST_FREQUENCY, DISPLAY_FREQUENCY, usingEWC=False)
episode_durations.append(ep_dur)
test_durations.append([])
for index, test in enumerate(test_dur):
    test_durations[index] += test

print(f"##################################################")
print(f"##################################################")
print(f"################# Starting task 2 ################")
print(f"##################################################")
print(f"##################################################")


ep_dur, test_dur = train(dqn, [env_down, env_right], 200, TEST_FREQUENCY, DISPLAY_FREQUENCY, usingEWC=True)
episode_durations.append(ep_dur)
test_durations.append([])
for index, test in enumerate(test_dur):
    test_durations[index] += test

np.save(f"data/{NAME}_episode_durations", np.array(episode_durations))
np.save(f"data/{NAME}_test_durations", np.array(test_durations))
dqn.save(f"models/{NAME}.pth")

visualise_data(NAME, TEST_FREQUENCY)



