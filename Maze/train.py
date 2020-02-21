import gym
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from DQN import DQN
from visualise import Visualizer

BATCH_SIZE = 32
GAMMA = 0.95
TARGET_UPDATE = 20
MEMORY_SIZE = 15_000
EPISODES = 500
CONSOLE_UPDATE_RATE = 50
# device = getDevice()



env = gym.make('CartPole-v0')
# env = env.unwrapped

dqn = DQN(GAMMA, MEMORY_SIZE, TARGET_UPDATE, BATCH_SIZE, env)

visualizer = Visualizer()
episode_durations = []

for episode in range(EPISODES):

    state = env.reset()
    state = torch.FloatTensor(state).view(-1, 4)
    steps = 0

    while True:
        if episode % CONSOLE_UPDATE_RATE == 0:
            env.render()

        action = dqn.choose_action(state)

        next_state, reward, done, info = env.step(action.item())

        x, x_dot, theta, theta_dot = next_state
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.6
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2
        

        # print(reward)

        if done:
            reward = -2.0

        reward = torch.tensor([[reward]])
        next_state = torch.FloatTensor(next_state).view(-1, 4)

        dqn.store_transition(state, action, reward, next_state)

        steps += 1
        
        dqn.learn()                

        if done: 
            if episode % CONSOLE_UPDATE_RATE == 0:
                print(f"{episode} Episode finished after {steps} steps.")
            break;

        if steps % 10_000 == 0:
            print(f"Stopping as it runs infinitely.")
            break;

        state = next_state

    dqn.decay_epsilon()
    episode_durations.append(steps)

    if episode % CONSOLE_UPDATE_RATE == 0:
        visualizer.plot_durations(episode_durations)
        visualizer.save_plot("cart_pole.png")

    if steps % 10_000 == 0:
        break;

dqn.save("eval_model.pth")
visualizer.save_plot("cart_pole.png")
