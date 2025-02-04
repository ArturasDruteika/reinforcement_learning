import random
from typing import Tuple, List

import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from learning.mdp_grid_world.grid_world_environment.environment import GridWorldEnvironment

class PolicyIteration:
    
    def __init__(
        self, 
        environment: GridWorldEnvironment,
        action_state_size: int,
        gamma: float,
        theta: float
    ) -> None:
        self.__env = environment
        self.__action_state_size = action_state_size
        self.__gamma = gamma
        self.__theta = theta
        self.__actions = [action for action in range(action_state_size)]
        self.__state_values = self.__init_state_values()
        self.__policy_table = self.__init_random_policy()
        
    def __init_state_values(self):
        return np.zeros(self.state_size)
    
    def __init_random_policy(self):
        """ Initializes a random stochastic policy where all actions have equal probability. """
        grid_height, grid_width = self.__env.grid.shape  # Extract grid size
        num_actions = self.__action_state_size  # Number of possible actions
        
        # Initialize a uniform probability distribution over all actions
        policy = np.full((grid_height, grid_width, num_actions), 1 / num_actions)
        
        return policy  # policy[y, x, action] gives P(action | state (x, y))

    
    def __calculate_state_action_subvalue(self, action_prob, transition_prob, reward, next_state_value):
        return action_prob * transition_prob * (reward + self.__gamma * next_state_value)
    
    @property
    def state_size(self) -> Tuple[int, int]:
        return self.__env.grid.shape
    
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
    def actions(self) -> List[int]:
        return self.__actions
    
    @property
    def state_values(self) -> np.ndarray:
        return self.__state_values
    
    @property
    def policy_table(self):
        return self.__policy_table
    
    def evaluate_policy(self):
        while True:
            prev_state_values = self.__state_values.copy()  # Store previous values
            delta = 0  # Track the largest change in state values
            
            for i in range(self.state_size[0]):
                for j in range(self.state_size[1]):
                    current_state = (i, j)
                    current_state_value = 0
                    
                    for action in self.__actions:
                        next_state, reward, _ = self.__env.step(current_state, action)
                        sub_value = self.__calculate_state_action_subvalue(
                            self.__policy_table[i, j, action],  
                            self.__env.transition_prob, reward,  
                            self.__state_values[next_state]
                        )
                        current_state_value += sub_value
                    
                    self.__state_values[i, j] = current_state_value  
            
            # Compute max change in state values across all states
            delta = np.max(np.abs(prev_state_values - self.__state_values))
            # If changes are small, stop
            if delta < self.__theta:
                break


                    
            
    def improve_policy(self):
        pass
    
    
if __name__ == '__main__':
    env = GridWorldEnvironment(transition_prob=0.9)
    policy_iteration = PolicyIteration(env, action_state_size=4, gamma=0.9, theta=0.001)
    policy_iteration.evaluate_policy()