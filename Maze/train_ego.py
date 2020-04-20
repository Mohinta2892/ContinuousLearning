import gym
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utilities import train, test, showEnv
from environments.EmptyEnv import EmptyEnv
from environments.CrossEnv import CrossEnv
from environments.TShapedEnv import TShapedEnv
from environments.Trials import TrialEnv
from DQN import DQN
from visualise import visualise_data

import numpy as np

NAME = "CrossEnv"

BATCH_SIZE = 64
GAMMA = 0.9
TARGET_UPDATE = 10
MEMORY_SIZE = 5_000
EPISODES = 500
DISPLAY_FREQUENCY = 500
TEST_FREQUENCY = 5

# env_ego = TShapedEnv(False)
# env_alo = TShapedEnv(True)

# env_allo_1 = TrialEnv(TrialEnv.SOUTH, TrialEnv.WEST)
# env_allo_2 = TrialEnv(TrialEnv.NORTH, TrialEnv.WEST)
# env_allo_3 = TrialEnv(TrialEnv.NORTH, TrialEnv.EAST)
# env_allo_4 = TrialEnv(TrialEnv.SOUTH, TrialEnv.EAST)

# env_ego = TrialEnv(TrialEnv.NORTH, TrialEnv.WEST)
# env_alo = TrialEnv(TrialEnv.NORTH, TrialEnv.EAST)

env_ego = CrossEnv(False)
env_alo = CrossEnv(True)

# showEnv(env_ego)
showEnv(env_alo)

# Turn Left, Turn Right, Move Forward
env_action_num = 3
env_state_num = env_ego.getStateSize()

dqn = DQN(GAMMA, MEMORY_SIZE, TARGET_UPDATE, BATCH_SIZE, env_state_num, env_action_num, ewc_importance=3000)

episode_durations = []
test_durations = []

ep_dur, test_dur = train(dqn, [env_ego], EPISODES, TEST_FREQUENCY, DISPLAY_FREQUENCY, usingEWC=False)
episode_durations.append(ep_dur)
test_durations.append([])
for index, testS in enumerate(test_dur):
    test_durations[index] += testS

print(f"##################################################")
print(f"##################################################")
print(f"################# Starting task 2 ################")
print(f"##################################################")
print(f"##################################################")


ep_dur, test_dur = train(dqn, [env_ego, env_alo], EPISODES, TEST_FREQUENCY, DISPLAY_FREQUENCY, usingEWC=True)
episode_durations.append(ep_dur)
test_durations.append([])
for index, testS in enumerate(test_dur):
    test_durations[index] += testS

# print(f"##################################################")
# print(f"##################################################")
# print(f"################# Starting task 3 ################")
# print(f"##################################################")
# print(f"##################################################")


# ep_dur, test_dur = train(dqn, [env_allo_1, env_allo_2, env_allo_3], EPISODES, TEST_FREQUENCY, DISPLAY_FREQUENCY, usingEWC=True)
# episode_durations.append(ep_dur)
# test_durations.append([])
# for index, testS in enumerate(test_dur):
#     test_durations[index] += testS


# print(f"##################################################")
# print(f"##################################################")
# print(f"################# Starting task 4 ################")
# print(f"##################################################")
# print(f"##################################################")


# ep_dur, test_dur = train(dqn, [env_allo_1, env_allo_2, env_allo_3, env_allo_4], EPISODES, TEST_FREQUENCY, DISPLAY_FREQUENCY, usingEWC=True)
# episode_durations.append(ep_dur)
# test_durations.append([])
# for index, testS in enumerate(test_dur):
#     test_durations[index] += testS

# test(dqn.eval_model, env_allo_1)
# test(dqn.eval_model, env_allo_1)
# test(dqn.eval_model, env_allo_3)
# test(dqn.eval_model, env_allo_4)

np.save(f"data/{NAME}_episode_durations", np.array(episode_durations))
np.save(f"data/{NAME}_test_durations", np.array(test_durations))
dqn.save(f"models/{NAME}.pth")
visualise_data(NAME, TEST_FREQUENCY, [7, 7])


