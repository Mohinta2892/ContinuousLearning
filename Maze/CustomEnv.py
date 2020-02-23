from gym_minigrid.minigrid import *
from gym_minigrid.register import register

import torch
import numpy as np

class EmptyEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward

    0 - EMPTY
    1 - WALL 
    2 - REWARD
    3 - AGENT

    """

    def __init__(self, goal_position, size=8, agent_start_pos=(1,1), agent_start_dir=0):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_position = goal_position

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,

            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), self.goal_position[0], self.goal_position[1])

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

        grid_mask = np.array([s != None for s in self.grid.grid])
        self.grid_data = np.zeros([self.grid.width * self.grid.height])
        self.grid_data[grid_mask] = 1

        self.grid_data[self.goal_position[1] * self.grid.width + self.goal_position[0]] = 2

    def getStateSize(self):
        return self.grid.width * self.grid.height + 1

    def extractState(self):
        state = np.copy(self.grid_data)

        # Update agent position and direction
        state[self.agent_pos[1] * self.grid.width + self.agent_pos[0]] = 3
        state = np.append(state, self.agent_dir)
        
        return torch.FloatTensor([state])
        
        