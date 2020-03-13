import gym
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utilities import train, test
from environments.EmptyEnv import EmptyEnv
from environments.CrossEnv import CrossEnv
from environments.TShapedEnv import TShapedEnv
from environments.Trials import TrialEnv
from DQN import DQN
from visualise import visualise_data

import numpy as np

NAME = "test"

BATCH_SIZE = 64
GAMMA = 0.9
TARGET_UPDATE = 10
MEMORY_SIZE = 5_000
EPISODES = 500
DISPLAY_FREQUENCY = 500
TEST_FREQUENCY = 5

# env_ego = TShapedEnv(False)
# env_alo = TShapedEnv(True)

env_ego = TrialEnv(TrialEnv.NORTH, TrialEnv.WEST)
env_alo = TrialEnv(TrialEnv.SOUTH, TrialEnv.WEST)

# env_ego = TrialEnv(TrialEnv.NORTH, TrialEnv.WEST)
# env_alo = TrialEnv(TrialEnv.NORTH, TrialEnv.EAST)

# env_ego = CrossEnv(False)
# env_alo = CrossEnv(True)

# Turn Left, Turn Right, Move Forward
env_action_num = 3
env_state_num = env_ego.getStateSize()

dqn = DQN(GAMMA, MEMORY_SIZE, TARGET_UPDATE, BATCH_SIZE, env_state_num, env_action_num, ewc_importance=2000)

episode_durations = []
test_durations = []

ep_dur, test_dur = train(dqn, [env_ego], EPISODES, TEST_FREQUENCY, DISPLAY_FREQUENCY, usingEWC=False)
episode_durations.append(ep_dur)
test_durations.append([])
for index, test in enumerate(test_dur):
    test_durations[index] += test

print(f"##################################################")
print(f"##################################################")
print(f"################# Starting task 2 ################")
print(f"##################################################")
print(f"##################################################")


ep_dur, test_dur = train(dqn, [env_ego, env_alo], EPISODES, TEST_FREQUENCY, DISPLAY_FREQUENCY, usingEWC=True)
episode_durations.append(ep_dur)
test_durations.append([])
for index, test in enumerate(test_dur):
    test_durations[index] += test

np.save(f"data/{NAME}_episode_durations", np.array(episode_durations))
np.save(f"data/{NAME}_test_durations", np.array(test_durations))
dqn.save(f"models/{NAME}.pth")
visualise_data(NAME, TEST_FREQUENCY, [7, 7])


