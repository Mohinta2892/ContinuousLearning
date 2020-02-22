import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np

from PIL import Image

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

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

def test(net, env):

    state = env.reset()
    state = extractState(state)

    steps_taken = 0

    while True:
        env.render()
        action = net(state).argmax().item()
        next_state, reward, done, info = env.step(action)
        steps_taken += 1

        if done:
            break

        state = extractState(next_state)

    print(f"Testing took {steps_taken} steps.")


def train(dqn, env, episode_durations, EPISODES, CONSOLE_UPDATE_RATE, visualizer):

    for episode in range(EPISODES):
        state = env.reset()
        state = extractState(state)
        steps = 0

        while True:
            if episode % CONSOLE_UPDATE_RATE == 0:
                env.render()

            action = dqn.choose_action(state)

            next_state, reward, done, info = env.step(action.item())

            reward = torch.FloatTensor([[reward]])
            next_state = extractState(next_state)

            dqn.store_transition(state, action, reward, next_state)

            steps += 1
            
            dqn.learn()                

            if done: 
                if episode % CONSOLE_UPDATE_RATE == 0:
                    print(f"{episode} Episode finished after {steps} steps.")
                break

            state = next_state

        dqn.decay_epsilon()
        episode_durations.append(steps)

        if episode % CONSOLE_UPDATE_RATE == 0:
            visualizer.plot_durations(episode_durations)

        if steps % 10_000 == 0:
            break;



def train_ewc(dqn, env, episode_durations, EPISODES, CONSOLE_UPDATE_RATE, visualizer):

    # Calculate Fisher by creating the EWC object

    # Do you need a seperate ReplayMemory store?

    for episode in range(EPISODES):
        state = env.reset()
        state = extractState(state)
        steps = 0

        while True:
            if episode % CONSOLE_UPDATE_RATE == 0:
                env.render()

            action = dqn.choose_action(state)

            next_state, reward, done, info = env.step(action.item())

            reward = torch.FloatTensor([[reward]])
            next_state = extractState(next_state)

            dqn.store_transition(state, action, reward, next_state)

            steps += 1
            
            dqn.learn()            

            if done: 
                if episode % CONSOLE_UPDATE_RATE == 0:
                    print(f"{episode} Episode finished after {steps} steps.")
                break

            state = next_state

        dqn.decay_epsilon()
        episode_durations.append(steps)

        if episode % CONSOLE_UPDATE_RATE == 0:
            visualizer.plot_durations(episode_durations)

        if steps % 10_000 == 0:
            break