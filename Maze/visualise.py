import matplotlib
import matplotlib.pyplot as plt
import torch

class Visualizer(object):

    def __init__(self, average=100):
        super().__init__()

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.set_title('DQN Training for CartPole')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Duration')

        self.fig = fig
        self.ax = ax

    def plot_durations(self, episode_durations):
        durations_t = torch.tensor(episode_durations, dtype=torch.float)
        
        self.ax.plot(durations_t.numpy())

        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            self.ax.plot(means.numpy())

        # plt.pause(0.001)  # pause a bit so that plots are updated
        # self.fig.show()

    def save_plot(self, path):
        self.fig.tight_layout()
        self.fig.savefig(path)
