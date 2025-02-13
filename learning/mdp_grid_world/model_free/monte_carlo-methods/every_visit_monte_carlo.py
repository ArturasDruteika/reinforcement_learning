from typing import List, Tuple, Union
import time
import random

import numpy as np
import rootutils
from tqdm import tqdm

rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from learning.mdp_grid_world.grid_world_environment.environment import GridWorldEnvironment, RewardValues
from learning.mdp_grid_world.grid_world_environment.actions import Action


class EveryVisitMonteCarlo:
    
    def __init__(
        self, 
        state_space_size: Tuple[int, int], 
        action_state_size: int,
        gamma,
        alpha,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01
    ):
        self.__state_space_size = state_space_size
        self.__action_state_size = action_state_size
        self.__gamma = gamma,
        self.__alpha = alpha
        self.__epsilon = epsilon
        self.__epsilon_decay = epsilon_decay
        self.__min_epsilon = min_epsilon
        self.__q_table = self.__initialize_q_table(self.__state_space_size, self.__action_state_size)
    
    @staticmethod   
    def __initialize_q_table(state_space_size: Tuple[int, int], action_space_size: int) -> np.ndarray:
        """
        Initializes the Q-table with small random values.

        Args:
            state_space_size (Tuple[int, int]): The state space dimensions.
            action_space_size (int): The number of possible actions.

        Returns:
            np.ndarray: A 3D Q-table initialized with random values.
        """
        return np.zeros((state_space_size[0], state_space_size[1], action_space_size))
    
    def __update_state_action_value(
        self, 
        state: Tuple[int, int], 
        action: Action, 
        reward: Union[int, float]
    ):
        self.__q_table[state[0], state[1], action] = self.__q_table[state[0], state[1], action] + self.__alpha * (reward - self.__q_table[state[0], state[1], action])
    
    @property
    def state_space_size(self) -> Tuple[int, int]:
        """Returns the state space size."""
        return self.__state_space_size
    
    @property
    def action_state_size(self) -> int:
        """Returns the action-state size."""
        return self.__action_state_size
    
    @property
    def gamma(self) -> float:
        """Returns the discount factor (γ)."""
        return self.__gamma
    
    @property
    def alpha(self) -> float:
        """Returns the learning rate (α)."""
        return self.__alpha
    
    @property
    def epsilon(self) -> float:
        """Returns the exploration rate (ε)."""
        return self.__epsilon
    
    @property
    def epsilon_decay(self) -> float:
        """Returns the rate at which epsilon decays."""
        return self.__epsilon_decay
    
    @property
    def min_epsilon(self) -> float:
        """Returns the minimum possible value for epsilon."""
        return self.__min_epsilon
    
    @property
    def q_table(self) -> np.ndarray:
        """Returns the Q-table."""
        return self.__q_table
    

    def choose_action(self, current_state: Tuple[int, int]) -> Action:
        """
        Chooses an action using an ε-greedy strategy.

        Args:
            current_state (Tuple[int, int]): The current state.

        Returns:
            Action: The chosen action.
        """
        if np.random.rand() < self.__epsilon:
            # Exploration: Choose a random action
            return random.choice(list(Action))
        else:
            # Exploitation: Choose the best action from Q-table
            action_index: int = np.argmax(self.__q_table[current_state[0], current_state[1]])
            return Action(action_index)
        
    def decay_epsilon(self) -> None:
        """
        Decreases the epsilon value for exploration.
        """
        self.__epsilon = max(self.__epsilon * self.__epsilon_decay, self.__min_epsilon)
    
    def learn(self, episodes_info_list) -> None:
        for episode_info_dict in episodes_info_list:
            state = episode_info_dict['state']
            action = episode_info_dict['action']
            next_state = episode_info_dict['next_state']
            reward = episode_info_dict['reward']
            done = episode_info_dict['done']
            
            self.__update_state_action_value(state, action.value, reward)
            
            if done:
                break
    
    
if __name__ == '__main__':
    env = GridWorldEnvironment(0.9)
    every_visit_monte_carlo = EveryVisitMonteCarlo(
        state_space_size=env.grid.shape, 
        action_state_size=len(list(Action)),
        gamma=0.99,
        alpha=1e-5
    )
    n_episodes = 10_000
    
    start_state = (0, 0)
    current_state = start_state
    reward = 0
    episode_info = {'state': None,
                    'action': None,
                    'next_state': None,
                    'reward': None,
                    'done': False}
    episodes_info_list = []
    
    for episode in tqdm(range(n_episodes)):
        while reward != RewardValues.HOLE_STATE_VALUE and reward != RewardValues.GOAL_STATE_VALUE:
            action = every_visit_monte_carlo.choose_action(current_state)
            next_state, reward, done = env.step(current_state, action)
            episode_info['state'] = current_state
            episode_info['action'] = action
            episode_info['next_state'] = next_state
            episode_info['reward'] = reward
            episode_info['done'] = done
            episodes_info_list.append(episode_info.copy())
            current_state = next_state
        
        every_visit_monte_carlo.learn(episodes_info_list)
        episodes_info_list = []
        every_visit_monte_carlo.decay_epsilon()
        
        
    print(every_visit_monte_carlo.q_table)