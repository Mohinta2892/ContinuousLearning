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
        self.fig.tight_layout()
        self.fig.savefig(path)
