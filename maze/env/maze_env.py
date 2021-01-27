from copy import copy, deepcopy

import numpy as np
from environment import BaseEnvironment


class MazeEnvironment(BaseEnvironment):
    """Implements the environment for an RLGlue environment

    Note:
        env_init, env_start, env_step, env_cleanup, and env_message are required
        methods.
    """

    def env_init(self, env_info={}):
        """Setup for the environment called when the experiment first starts.

        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """
        self.rows, self.cols = self.maze_dim = env_info.get("maze_dim", [7, 7])
        self.obstacles_locs = env_info.get("obstacles", [])
        self.doors_locs = env_info.get("doors", {})
        self.start_state = env_info.get("start_state", [0, 0])
        self.end_state = env_info.get("end_state", [0, 0])
        self.current_state = None
        self.reward_obs_term = [0.0, None, False]

    def env_start(self, keep_history=False):
        """The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """
        self.keep_history = keep_history
        self.current_state = self.start_state
        self.reward_obs_term[1] = self.get_observation(self.current_state)
        self.obstacles = deepcopy(self.obstacles_locs)
        self.doors = deepcopy(self.doors_locs)
        self.door_open = False
        self.door_timer = None

        if self.keep_history:
            # initialize a gridview of the environment
            self.update_grid()
            self.history = [copy(self.grid)]
        return self.reward_obs_term[1], False

    def out_of_bounds(self, row, col):
        """check if current state is within the gridworld and return bool"""
        if row < 0 or row > self.maze_dim[0]-1 or col < 0 or col > self.maze_dim[1]-1:
            return True
        else:
            return False

    def is_obstacle(self, row, col):
        """check if there is an obstacle at (row, col)"""
        if [row, col] in self.obstacles:
            return True
        else:
            return False

    def get_observation(self, state):
        return state[0] * self.maze_dim[1] + state[1]

    def get_state_features(self, state):
        return [self.get_observation(state), tuple(state) in self.doors]

    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """

        reward = 0.0
        # reward = -1.0
        is_terminal = False
        row, col = self.current_state

        if self.keep_history:
            self.grid[row, col] = 0

        # update current_state with the action (also check validity of action)
        if action == 0:  # left
            if not (self.out_of_bounds(row, col-1) or self.is_obstacle(row, col-1)):
                self.current_state = [row, col-1]
        elif action == 1:  # right
            if not (self.out_of_bounds(row, col+1) or self.is_obstacle(row, col+1)):
                self.current_state = [row, col+1]
        elif action == 2:  # up	
            if not (self.out_of_bounds(row+1, col) or self.is_obstacle(row+1, col)):	
                self.current_state = [row+1, col]
        elif action == 3:  # down	
            if not (self.out_of_bounds(row-1, col) or self.is_obstacle(row-1, col)):	
                self.current_state = [row-1, col]        
        # terminate if goal is reached
        if self.current_state == self.end_state:
            reward = 1.0
            # reward = 0.0
            is_terminal = True

        # if self.current_state == [0,0]:
        #     reward = 0.0
        #     is_terminal = True

        self.reward_obs_term = [reward, self.get_state_features(self.current_state), is_terminal]
        if self.keep_history:
            row, col = self.current_state
            self.grid[row, col] = 100
            self.history.append(copy(self.grid))
        return self.reward_obs_term

    def env_cleanup(self):
        """Cleanup done after the environment ends"""
        current_state = None
        self.history = []
        self.grid = np.zeros(self.maze_dim)

    def update_grid(self):
        self.grid = np.zeros(self.maze_dim)
        self.grid[self.current_state[0], self.current_state[1]] = 100
        self.grid[self.end_state[0], self.end_state[1]] = -1
        mask_grid(self.grid, self.obstacles)
        mask_grid(self.grid, list(self.doors.keys()), -1)
        self.grid = np.ma.masked_where(self.grid == -100, self.grid)


def mask_grid(grid, blocks, mask_value=-100):
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if [i, j] in (list(x) for x in blocks):
                grid[i, j] = mask_value
