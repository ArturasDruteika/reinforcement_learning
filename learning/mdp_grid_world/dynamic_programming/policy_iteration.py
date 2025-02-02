from typing import Tuple

from numpy import np


class PolicyIteration:
    
    def __init__(
        self, 
        state_size: Tuple[int, int],
        action_state_size: int,
        gamma: float
    ) -> None:
        self.__state_size = state_size
        self.__action_state_size = action_state_size
        self.__gamma = gamma
        self.__state_values = self.__init_state_values()
        self.__policy = self.__init_policy()
        
    def __init_state_values(self):
        return np.zeros(self.__state_size)
    
    def __init_random_policy(self):
        pass
    
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
    def state_values(self) -> np.ndarray:
        return self.__state_values
    
    def policy_evaluation(self):
        pass
    
    def evaluate_policy(self):
        pass
    
    def improve_policy(self):
        pass