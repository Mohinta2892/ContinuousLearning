from gym_minigrid.minigrid import *
from gym_minigrid.register import register

import torch
import numpy as np

class TrialEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward

    0 - EMPTY
    1 - WALL 
    2 - REWARD
    3 - AGENT

    """

    DIR_UP    = 3
    DIR_DOWN  = 1
    DIR_LEFT  = 2
    DIR_RIGHT = 0

    NORTH = (4, 1)
    SOUTH = (4, 7)
    EAST  = (7, 4)
    WEST  = (1, 4)

    def __init__(self, agent_pos, goal_position, size=9):
        self.agent_start_pos = agent_pos
        self.agent_start_dir = self.DIR_UP if agent_pos == self.SOUTH else self.DIR_DOWN
        self.goal_position = goal_position

        super().__init__(
            grid_size=size,
            max_steps=200,
        )

    def _gen_grid(self, width, height):

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate borders
        self.grid.wall_rect(0, 0, width, height)

        # Generate PLUS
        for x in [1,2,3,5,6,7]:
            self.grid.vert_wall(x, 1, 3)

        for x in [1,2,3,5,6,7]:
            self.grid.vert_wall(x, 5, 3)

        # Generate T-shaped
        if self.agent_start_pos == self.SOUTH:
            self.grid.vert_wall(4, 1, 3)
        else:
            self.grid.vert_wall(4, 5, 3)

        # Place goal
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
        self.grid_data[grid_mask] = -1
        self.grid_data[self.goal_position[1] * self.grid.width + self.goal_position[0]] = 2

    def getStateSize(self):
        return self.grid.width * self.grid.height + 1

    def extractState(self):
        state = np.copy(self.grid_data)

        # Update agent position and direction
        state[self.agent_pos[1] * self.grid.width + self.agent_pos[0]] = 3
        state = np.append(state, self.agent_dir)
        # state = np.append(state, self.agent_pos[0])
        # state = np.append(state, self.agent_pos[1])
        # state = np.append(state, self.agent_dir)

        # state[0]  = self.get_offset()
        # state[1]  = self.get_offset()
        # state[9]  = self.get_offset()
        # state[10] = self.get_offset()

        return torch.FloatTensor([state])
    
    def get_offset(self):
        if self.agent_start_pos == self.NORTH:
            if self.goal_position == self.WEST:
                return 10
            else:
                return -10

        else:
            if self.goal_position == self.WEST:
                return 50
            else:
                return -50
    
        
    def _reward(self):
        """
        Compute the reward to be given upon success
        """

        return 1 - 0.9 * (self.step_count / self.max_steps)