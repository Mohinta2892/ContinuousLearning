import matplotlib
import matplotlib.pyplot as plt
import torch

import numpy as np

class Visualizer(object):

    def __init__(self, title='DQN Training', average=100):
        super().__init__()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(title)
        ax.set_xlabel('Episodes', fontsize=12)
        ax.set_ylabel('Steps', fontsize=12)
        
        
        # textstr = '\n'.join((f"optimiser={optimiser}",f"lr={lr}"))

        # props = dict(boxstyle='round', alpha=0.5, facecolor='wheat')

        # ax.text(0.77, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        #         verticalalignment='top', bbox=props)


        self.fig = fig
        self.ax = ax

    def plot_durations(self, episode_durations):
        durations_t = torch.tensor(episode_durations, dtype=torch.float)
        
        self.ax.plot(durations_t.numpy(), c='orange', label="steps")

        if len(durations_t) >= 50:
            means = durations_t.unfold(0, 50, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(49), means))
            self.ax.plot(means.numpy(), c='red', label="average steps")

        # plt.pause(0.001)  # pause a bit so that plots are updated
        # self.fig.show()

    def save_plot(self, path):
        self.ax.legend(loc="upper right", fontsize = 'large')
        # self.fig.tight_layout()
        self.fig.savefig(path)


def visualise_data(NAME, TEST_FREQUENCY, min_steps):
    plt.style.use('ggplot')

    episode_durations = np.load(f"data/{NAME}_episode_durations.npy", allow_pickle=True)
    test_durations    = np.load(f"data/{NAME}_test_durations.npy", allow_pickle=True)
    
    accuracy = []
    for task, test_task in enumerate(test_durations):
        task_accuracy = []
        for test in test_task:
            task_accuracy.append(min_steps[task] / test * 100)

        accuracy.append(task_accuracy)
        

    episodes_per_task = len(episode_durations[0])
    # episode_durations = episode_durations.reshape(-1, )

    # fig1, ax1 = plt.subplots(figsize=(12, 7))
    # fig2, ax2 = plt.subplots(figsize=(12, 7))
    fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, figsize=(12, 14))

    ax1.set_title("Steps done per episode during DQN Training of 2 maze tasks")
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Steps")

    episode_X = np.arange(episodes_per_task)
    ax1.plot(episode_X, episode_durations[0], label='Task 1')
    ax1.plot(episode_X + episodes_per_task, episode_durations[1], label='Task 2')
    ax1.legend(loc='upper right')

    ax2.set_title("Accuracy during DQN Training of 2 maze tasks")
    ax2.set_xlabel("Episodes")
    ax2.set_ylabel("Accuracy")

    test_episodes = (episodes_per_task // TEST_FREQUENCY)
    episode_X = np.arange(0, episodes_per_task*2, TEST_FREQUENCY)
    episode_X1 = np.arange(episodes_per_task, episodes_per_task*2, TEST_FREQUENCY)
    ax2.plot(episode_X, accuracy[0], linewidth=3, label='Task 1')
    ax2.plot(episode_X1, accuracy[1], linewidth=3, label='Task 2')
    ax2.legend(loc='upper right')

    # fig1.savefig(f"images/{NAME}_training.png")
    # fig2.savefig(f"images/{NAME}_testing.png")
    fig.tight_layout()
    fig.savefig('temp')
    # plt.show()


visualise_data(NAME="emptyNoEWC", TEST_FREQUENCY=2, min_steps=[6, 5])

