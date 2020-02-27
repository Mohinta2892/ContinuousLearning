import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np

from Transition import Transition
from PIL import Image
from EWC import EWC
from DQN_Maze import DQN

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
    # state = extractState(state)
    state = env.extractState()

    steps_taken = 0

    while True:
        if should_render:
            env.render()

        action = net(state).argmax().item()
        next_state, reward, done, info = env.step(action)
        steps_taken += 1

        if done:
            break

        # state = extractState(next_state)
        state = env.extractState()

    print(f"Testing took {steps_taken} steps.")


def train(dqn, env, episode_durations, EPISODES, CONSOLE_UPDATE_RATE, visualizer, task=1):

    if(task > 1):
        dqn.reset_training()

    for episode in range(EPISODES + 1):
        state = env.reset()
        state = env.extractState()

        # state = extractState(state)
        steps = 0

        while True:
            if episode > 0 and episode % CONSOLE_UPDATE_RATE == 0:
                env.render()

            action = dqn.choose_action(state)

            next_state, reward, done, info = env.step(action.item())
            reward = torch.FloatTensor([[reward]])
            next_state = env.extractState()

            dqn.store_transition(state, action, reward, next_state)
            dqn.learn()                

            state = next_state
            steps += 1
            
            if done: 
                print(f"{episode} Episode finished after {steps} steps.")
                break

        dqn.decay_epsilon(episode)
        episode_durations.append(steps)


def train_ewc(dqn: DQN, env, old_env, episode_durations, EPISODES, CONSOLE_UPDATE_RATE, visualizer, task):
    # Reset the dqn
    dqn.reset_training()

    # Calculate Fisher by creating the EWC object
    ewc = EWC(dqn)

    for episode in range(EPISODES + 1):
        state = env.reset()
        state = env.extractState()

        # state = extractState(state)
        steps = 0
        
        while True:
            # if episode > 0 and episode % CONSOLE_UPDATE_RATE == 0:
                # env.render()

            action = dqn.choose_action(state)

            next_state, reward, done, info = env.step(action.item())
            reward = torch.FloatTensor([[reward]])
            next_state = env.extractState()

            dqn.store_transition(state, action, reward, next_state)
            dqn.learn(ewc)     

            state = next_state
            steps += 1           

            if done: 
                print(f"{episode} Episode finished after {steps} steps.")
                break

        dqn.decay_epsilon(episode)
        episode_durations.append(steps)

        if episode % CONSOLE_UPDATE_RATE == 0:
            test(dqn.eval_model, old_env, should_render=False)
            test(dqn.eval_model, env, should_render=False)