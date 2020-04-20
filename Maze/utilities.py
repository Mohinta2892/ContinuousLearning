import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm

from Transition import Transition
from PIL import Image
from EWC import EWC
from DQN import DQN

def showEnv(env):
    state = env.reset()
    state = env.extractState()
    steps_taken = 0
    isFinished = False

    while not isFinished:
        env.render()

    return steps_taken

def getDevice(forceCPU = False):
    if not forceCPU and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    return device

def extractState(state):
    state = np.append(state['image'][:,:,0].flatten(), state['direction']);
    return torch.FloatTensor([state])

def test(net, env, should_render=True):

    state = env.reset()
    state = env.extractState()
    steps_taken = 0
    isFinished = False

    while not isFinished:
        if should_render:
            env.render()

        with torch.no_grad():
            action = net(state).argmax().item()

        next_state, reward, done, info = env.step(action)

        steps_taken += 1
        isFinished = done
        state = env.extractState()

    return steps_taken


def train(dqn, envs, EPISODES, TEST_FREQUENCY, DISPLAY_FREQUENCY, usingEWC=False):

    firstTimeTraining = len(envs) == 1
    episode_durations = []
    test_durations = []
    for _ in envs:
        test_durations.append([])

    # Reset the DQN if the network is not new
    if not firstTimeTraining:
        dqn.reset_training()

    # Get the env we're training (the current env should be the last one in the list)
    env = envs[-1]

    if usingEWC:
        # Calculate Fisher by creating the EWC object
        ewc = EWC(dqn)

    for episode in range(EPISODES):
        state = env.reset()
        state = env.extractState()

        steps = 0
        isFinished = False

        while not isFinished:
            if episode > 0 and episode % DISPLAY_FREQUENCY == 0:
                env.render()

            action = dqn.choose_action(state)

            next_state, reward, done, info = env.step(action.item())
            reward = torch.FloatTensor([[reward]])
            next_state = env.extractState()

            dqn.store_transition(state, action, reward, next_state)

            # Update DQN
            if usingEWC:
                dqn.learn(ewc)
            else:
                dqn.learn()

            state = next_state
            steps += 1
            isFinished = done

        dqn.decay_epsilon(episode)

        print(f"{episode+1} Episode finished after {steps} steps.")
        episode_durations.append(steps)

        # Collect test data for each env
        if episode % TEST_FREQUENCY == 0:
            for index, test_env in enumerate(envs):
                test_steps = test(dqn.eval_model, test_env, should_render=False)
                test_durations[index].append(test_steps)
                print(f"Testing the {index+1} env took {test_steps} steps.")
            
            print(f"##################################################")

    return episode_durations, test_durations


def trainSmart(dqn: DQN, env, EPISODES, DISPLAY_FREQUENCY, config, usingEWC=True):
    episode_durations = []

    # Reset the DQN if the network is not new
    if dqn.learned_tasks > 0:
        dqn.reset_training()

    if dqn.learned_tasks > 0 and usingEWC:
        # Calculate Fisher by creating the EWC object
        ewc = EWC(dqn)

    # for episode in tqdm(range(EPISODES), desc=f"Training DQN on config {config}", unit='episode'):
    for episode in range(EPISODES):
        state = env.reset()
        state = env.extractState()

        steps = 0
        isFinished = False

        while not isFinished:
            # if episode > 0 and episode % DISPLAY_FREQUENCY == 0:
            #     env.render()

            action = dqn.choose_action(state)

            next_state, reward, done, info = env.step(action.item())
            reward = torch.FloatTensor([[reward]])
            next_state = env.extractState()

            dqn.store_transition(state, action, reward, next_state)

            # Update DQN
            if dqn.learned_tasks > 0 and usingEWC:
                dqn.learn(ewc)
            else:
                dqn.learn()

            state = next_state
            steps += 1
            isFinished = done

        dqn.decay_epsilon(episode)

        episode_durations.append(steps)
        # if episode % DISPLAY_FREQUENCY == 0:
        #     tqdm.write(f"{episode+1} Episode finished after {steps} steps.")

    dqn.learned_tasks += 1
    return episode_durations


def runDQN(dqn: DQN, ewc: EWC, env):
    state = env.reset()
    state = env.extractState()

    steps = 0
    isFinished = False

    while not isFinished:
        # if episode > 0 and episode % DISPLAY_FREQUENCY == 0:
        #     env.render()

        action = dqn.choose_action(state)

        next_state, reward, done, info = env.step(action.item())
        reward = torch.FloatTensor([[reward]])
        next_state = env.extractState()

        dqn.store_transition(state, action, reward, next_state)

        # Update DQN
        if ewc is None:
            dqn.learn()
        else:
            dqn.learn(ewc)

        state = next_state
        steps += 1
        isFinished = done

    # dqn.decay_epsilon()

    return steps