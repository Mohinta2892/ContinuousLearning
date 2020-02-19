import visualise
import gym
import gym_minigrid
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utilities import get_screen
from DQN import DQN
from Transition import Transition
from ReplayMemory import ReplayMemory

from itertools import count
import random
import math


BATCH_SIZE = 128
GAMMA = 0.999   # Same as Discount - or how far we look into the future
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10  # How often do we update our target model
device = torch.device("cpu")

# Init env
env = gym.make('CartPole-v0')
env.reset()
init_screen = get_screen(env)
_, _, screen_height, screen_width = init_screen.shape
n_actions = env.action_space.n

# Plot purposes
episode_durations = []

# Creating the two models
policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Creating the replaymemory storage
memory = ReplayMemory(10000)

optimizer = optim.RMSprop(policy_net.parameters())
loss_function = nn.SmoothL1Loss()

steps_done = 0

def select_action(state):
    global steps_done
    steps_done += 1

    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)

    if sample > eps_threshold:
        # Return the prediction
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # Return random action
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    # Gives 128 transitions which we combine into one
    transitions = memory.sample(BATCH_SIZE) # transitions.len = 128
    batch = Transition(*zip(*transitions))  # batch = transition, each element.len = 128

    state_batch  = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    current_q_list = policy_net(state_batch).gather(1, action_batch)

    # If the next_state == None => next_state is a final state, hence we're done
    # Create a mask array that represents if the state is final (False - not final, True - final) 
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    # Get all the non-final states
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    # Get what the previous qs were for the given state
    # Note that the q value is 0 if the state is final (comes from the mask)
    future_q_list = torch.zeros(BATCH_SIZE, device=device)
    future_q_list[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the new Q value.
    # If final state => new_q = rewards
    new_q = (future_q_list * GAMMA) + reward_batch

    # Compute Huber loss
    loss = loss_function(current_q_list, new_q.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    # for param in policy_net.parameters():
    #     param.grad.data.clamp_(-1, 1)

    optimizer.step()


num_episodes = 50
for i_episode in range(num_episodes):

    # Initialize the environment and state
    env.reset()

    last_screen = get_screen(env)
    current_screen = get_screen(env)
    state = current_screen - last_screen

    for t in count():
        # Select and perform an action
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen(env)

        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            visualise.plot_durations(episode_durations)
            break

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())