from enum import Enum
from typing import Tuple
import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from learning.mdp_grid_world.actions import Action


class GridWorldEnvironment:
    """
    A 4x4 GridWorld environment for reinforcement learning agents.
    The environment consists of a grid where the agent navigates by taking actions.
    
    Attributes:
        __grid (np.ndarray): A 4x4 numpy array representing state rewards in the environment.
    """

    def __init__(self) -> None:
        """
        Initializes the GridWorld environment by creating a grid with rewards.
        """
        self.__grid: np.ndarray = self.__create_grid()

    def __create_grid(self) -> np.ndarray:
        """
        Creates a 4x4 grid representing the environment.
        Positive values indicate rewards, negative values indicate penalties.

        Returns:
            np.ndarray: A 4x4 numpy array representing the grid with rewards.
        """
        grid = np.zeros((4, 4), dtype=np.int8)

        # Define rewards and penalties
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
        grid[3, 3] = 100  # Goal state

        return grid

    def __calculate_next_position(self, current_position: Tuple[int, int], action: Action) -> Tuple[int, int]:
        """
        Calculates the next position of the agent based on the selected action.

        Args:
            current_position (Tuple[int, int]): The current (row, column) position.
            action (Action): The action to be performed.

        Returns:
            Tuple[int, int]: The new position of the agent.
        """
        x, y = current_position

        if action == Action.UP:
            x -= 1
        elif action == Action.DOWN:
            x += 1
        elif action == Action.LEFT:
            y -= 1
        elif action == Action.RIGHT:
            y += 1

        # Ensure the agent stays within the grid boundaries
        if not (0 <= x < 4 and 0 <= y < 4):
            return current_position  # Stay in the same position if out of bounds

        return (x, y)

    @property
    def grid(self) -> np.ndarray:
        """
        Returns the grid representation of the environment.

        Returns:
            np.ndarray: A 4x4 numpy array containing the environment's rewards.
        """
        return self.__grid

    def step(self, current_position: Tuple[int, int], action: Action) -> Tuple[Tuple[int, int], int, bool]:
        """
        Executes an action in the environment and returns the next state, reward, and done flag.

        Args:
            current_position (Tuple[int, int]): The agent's current position.
            action (Action): The action chosen by the agent.

        Returns:
            Tuple[Tuple[int, int], int, bool]: 
                - The next state (row, column).
                - The reward received after taking the action.
                - A boolean indicating whether the episode is done.
        """
        next_state = self.__calculate_next_position(current_position, action)

        # If the agent stays in the same place (e.g., moves out of bounds), apply penalty
        if next_state == current_position:
            reward = -100  # Harsh penalty for trying to move out of bounds
            done = True
            return next_state, reward, done

        # Get the reward from the grid
        reward = self.__grid[next_state]
        
        if reward == -100:
            done = True  # Game over: agent reached a penalty state
            return next_state, reward, done

        # Check if the episode should end
        done = reward == 100  # Goal reached

        return next_state, reward, done
    
    
if __name__ == '__main__':
    env = GridWorldEnvironment()
    print(env.grid)
    
    actions = [Action.RIGHT, Action.DOWN, Action.RIGHT, Action.DOWN, Action.RIGHT, Action.DOWN]
    current_position = (0, 0)
    for action in actions:
        current_position, reward, done = env.step(current_position, action)
        print(f"current position: {current_position}, reward: {reward}, done: {done}")