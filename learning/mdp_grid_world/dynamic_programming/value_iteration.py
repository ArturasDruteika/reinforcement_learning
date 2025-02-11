from typing import List

import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from learning.mdp_grid_world.grid_world_environment.environment import GridWorldEnvironment, RewardValues
from learning.mdp_grid_world.grid_world_environment.actions import Action


class ValueIteration:
    
    def __init__(
        self,
        environment: GridWorldEnvironment,
        action_state_size: int,
        gamma: float,
        theta: float
    ):
        self.__env = environment
        self.__action_state_size = action_state_size
        self.__gamma = gamma
        self.__theta = theta
        self.__state_values = self.__init_state_values()
        self.__policy_table = None
        
    def __init_state_values(self) -> np.ndarray:
        """
        Initializes state values to zero.

        Returns:
            np.ndarray: A grid of zeros with the same shape as the environment.
        """
        return np.zeros(self.__env.grid.shape)
    
    def __calculate_state_value(
        self, 
        reward, 
        next_state_value,
        done
    ):
        """
        Calculates the state value using the Bellman equation.

        Args:
            next_state (Tuple[int, int]): Coordinates of the next state.
            reward (float): Reward received after the transition.
            next_state_value (float): Value of the next state.
            done (bool): Whether the next state is terminal.

        Returns:
            float: The calculated state value.
        """
        return reward + self.__gamma * next_state_value * (not done)
        
    
    def __update_state_values_iterativly(self):
        rows, cols = self.__env.grid.shape
        
        for row in range(rows):
            for col in range(cols):
                old_value = self.__state_values[row, col]
                state_value = float('-inf')
                for action in list(Action):
                    action_state_value = 0
                    
                    for _, next_state, reward, done in self.__env.grid_transition_dynamics[(row, col)][Action(action)]:
                        action_state_value += self.__calculate_state_value(
                            reward, 
                            self.__state_values[next_state],
                            done
                        )
                        
                    if action_state_value > state_value:
                        state_value = action_state_value
                        
                self.__state_values[row, col] = state_value
                    
    
    @property
    def action_state_size(self) -> int:
        return self.__action_state_size
    
    @property
    def gamma(self) -> float:
        return self.__gamma
    
    @property
    def theta(self) -> float:
        return self.__theta
    
    @property
    def state_values(self) -> np.ndarray:
        return self.__state_values
    
    @property
    def policy_table(self) -> np.ndarray:
        return self.__policy_table
            

if __name__ == '__main__':
    pass