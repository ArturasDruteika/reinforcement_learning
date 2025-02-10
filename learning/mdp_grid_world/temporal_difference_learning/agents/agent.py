import random
from typing import List, Tuple
from abc import ABC, abstractmethod

import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from learning.mdp_grid_world.grid_world_environment.actions import Action


class Agent(ABC):
    """
    A SARSA agent that learns an optimal policy for navigating a GridWorld environment.

    Attributes:
        __state_space_size (Tuple[int, int]): The number of states in the environment.
        __action_space_size (int): The number of possible actions.
        __alpha (float): The learning rate for SARSA updates.
        __gamma (float): The discount factor for future rewards.
        __epsilon (float): The exploration rate for the ε-greedy policy.
        __epsilon_decay (float): The rate at which epsilon decreases.
        __min_epsilon (float): The minimum possible value for epsilon.
        __q_table (np.ndarray): The Q-table storing action-value estimates.
    """

    def __init__(
        self,
        state_space_size: Tuple[int, int],
        action_space_size: int,
        learning_rate: float,
        discount_ratio: float,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01
    ):
        """
        Initializes the SARSA agent with a Q-table and learning parameters.

        Args:
            state_space_size (Tuple[int, int]): The state space size.
            action_space_size (int): The number of possible actions.
            learning_rate (float): Learning rate (α).
            discount_ratio (float): Discount factor (γ).
            epsilon (float, optional): Exploration rate. Defaults to 1.0.
            epsilon_decay (float, optional): Rate at which epsilon decays. Defaults to 0.995.
            min_epsilon (float, optional): Minimum epsilon value. Defaults to 0.01.
        """
        self._state_space_size: Tuple[int, int] = state_space_size
        self._action_space_size: int = action_space_size
        self._alpha: float = learning_rate
        self._gamma: float = discount_ratio
        self._epsilon: float = epsilon
        self._epsilon_decay: float = epsilon_decay
        self._min_epsilon: float = min_epsilon
        self._q_table: np.ndarray = self.__initialize_q_table(self._state_space_size, self._action_space_size)

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

    def _decay_epsilon(self) -> None:
        """
        Decreases the epsilon value for exploration.
        """
        self._epsilon = max(self._epsilon * self._epsilon_decay, self._min_epsilon)
        
    @property
    def state_space_size(self) -> Tuple[int, int]:
        """Returns the state space size."""
        return self._state_space_size
    
    @property
    def action_space_size(self) -> int:
        """Returns the action space size."""
        return self._action_space_size
    
    @property
    def learning_rate(self) -> float:
        """Returns the learning rate (α)."""
        return self._alpha
    
    @property
    def discount_ratio(self) -> float:
        """Returns the discount factor (γ)."""
        return self._gamma
    
    @property
    def epsilon(self) -> float:
        """Returns the exploration rate (ε)."""
        return self._epsilon
    
    @property
    def epsilon_decay(self) -> float:
        """Returns the rate at which epsilon decays."""
        return self._epsilon_decay
    
    @property
    def min_epsilon(self) -> float:
        """Returns the minimum possible value for epsilon."""
        return self._min_epsilon

    @property
    def q_table(self) -> np.ndarray:
        """Returns the current Q-table."""
        return self._q_table

    def choose_action(self, current_state: Tuple[int, int]) -> Action:
        """
        Chooses an action using an ε-greedy strategy.

        Args:
            current_state (Tuple[int, int]): The current state.

        Returns:
            Action: The chosen action.
        """
        if np.random.rand() < self._epsilon:
            # Exploration: Choose a random action
            return random.choice(list(Action))
        else:
            # Exploitation: Choose the best action from Q-table
            action_index: int = np.argmax(self._q_table[current_state[0], current_state[1]])
            return Action(action_index)

    @abstractmethod
    def update_q_values(self, *args, **kwargs) -> None:
        pass
