from enum import Enum
from typing import Tuple

import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from learning.mdp_grid_world.actions import Action


class GridWorldEnvironment:
    
    def __init__(self):
        self.__grid = self.__create_grid()
    
    def __create_grid(self):
        grid = np.zeros((4, 4), dtype=np.int8)
        
        # TODO: try implementing probabilities to of actions in every state like:
        # grid[2, 3] = [reward, probability of performing a selected action]
        grid[0, 0] = 0
        grid[0, 1] = -1
        grid[0, 2] = -1
        grid[0, 3] = -100
        
        grid[1, 0] = -1
        grid[1, 1] = -100
        grid[1, 2] = -1
        grid[1, 3] = -1
        
        grid[2, 0] = -1
        grid[2, 1] = -1
        grid[2, 2] = -100
        grid[2, 3] = -1
        
        grid[3, 0] = -100
        grid[3, 1] = -1
        grid[3, 2] = -1
        grid[3, 3] = 100
        
        return grid
    
    def __calculate_next_position(self, current_position, action):
        next_position = (current_position[0], current_position[1])
        
        if action == Action.UP.value:
            next_position = (current_position[0] - 1, current_position[1])
        elif action == Action.DOWN.value:
            next_position = (current_position[0] + 1, current_position[1])
        elif action == Action.LEFT.value:
            next_position = (current_position[0], current_position[1] - 1)
        elif action == Action.RIGHT.value:
            next_position = (current_position[0], current_position[1] + 1)
            
        # Check if out of bounds
        if not (0 <= next_position[0] < 4 and 0 <= next_position[1] < 4):
            return current_position  # If out of bounds, return the same position
        
        return next_position
    
    @property
    def grid(self) -> np.ndarray:
        return self.__grid
    
    def step(self, current_position: Tuple[int, int], action: Action):
        reward = 0
        done = False
        next_state = self.__calculate_next_position(current_position, action)
        if next_state == current_position:
            reward = -100
            return next_state, reward, done
        reward = self.__grid[next_state]
        if reward == 100:
            done = True
        return next_state, reward, done
    
    
if __name__ == '__main__':
    env = GridWorldEnvironment()
    print(env.grid)
    
    actions = [Action.RIGHT, Action.DOWN, Action.RIGHT, Action.DOWN, Action.RIGHT, Action.DOWN]
    current_position = (0, 0)
    for action in actions:
        current_position, reward, done = env.step(current_position, action)
        print(f"current position: {current_position}, reward: {reward}, done: {done}")