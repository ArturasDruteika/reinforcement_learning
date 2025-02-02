import random
from typing import Tuple, List

import numpy as np

class PolicyIteration:
    
    def __init__(
        self, 
        state_size: Tuple[int, int],
        action_state_size: int,
        gamma: float,
        theta: float
    ) -> None:
        self.__state_size = state_size
        self.__action_state_size = action_state_size
        self.__gamma = gamma
        self.__theta = theta
        self.__actions = [action for action in range(action_state_size)]
        self.__state_values = self.__init_state_values()
        self.__policy = self.__init_random_policy()
        
    def __init_state_values(self):
        return np.zeros(self.__state_size)
    
    def __init_random_policy(self):
        return np.random.choice(self.__actions, size = self.__state_size)
    
    @property
    def state_size(self) -> Tuple[int, int]:
        return self.__state_size
    
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
    def policy(self):
        return self.__policy
    
    def evaluate_policy(self):
        pass
    
    def improve_policy(self):
        pass