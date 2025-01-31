import random
from typing import List, Tuple
import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from learning.mdp_grid_world.actions import Action


class SarsaAgent:
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
        self.__state_space_size: Tuple[int, int] = state_space_size
        self.__action_space_size: int = action_space_size
        self.__alpha: float = learning_rate
        self.__gamma: float = discount_ratio
        self.__epsilon: float = epsilon
        self.__epsilon_decay: float = epsilon_decay
        self.__min_epsilon: float = min_epsilon
        self.__q_table: np.ndarray = self.__initialize_q_table(self.__state_space_size, self.__action_space_size)

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

    def __decay_epsilon(self) -> None:
        """
        Decreases the epsilon value for exploration.
        """
        self.__epsilon = max(self.__epsilon * self.__epsilon_decay, self.__min_epsilon)
        
    @property
    def state_space_size(self) -> Tuple[int, int]:
        """Returns the state space size."""
        return self.__state_space_size
    
    @property
    def action_space_size(self) -> int:
        """Returns the action space size."""
        return self.__alpha
    
    @property
    def learning_rate(self) -> float:
        """Returns the learning rate (α)."""
        return self.__alpha
    
    @property
    def discount_ratio(self) -> float:
        """Returns the discount factor (γ)."""
        return self.__gamma
    
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
        """Returns the current Q-table."""
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

    def update_q_values(
        self,
        current_state: Tuple[int, int],
        action: Action,
        reward: float,
        next_state: Tuple[int, int],
        next_action: Action,
        done: bool
    ) -> None:
        """
        Updates the Q-values using the SARSA update rule.

        Args:
            current_state (Tuple[int, int]): The current state.
            action (Action): The action taken.
            reward (float): The received reward.
            next_state (Tuple[int, int]): The next state.
            next_action (Action): The next action taken.
            done (bool): Whether the episode has ended.

        Returns:
            None
        """
        current_q: float = self.__q_table[current_state[0], current_state[1], action.value]

        if done:
            target_q = reward  # No future rewards if terminal state
        else:
            next_q = self.__q_table[next_state[0], next_state[1], next_action.value]  # Use SARSA rule
            target_q = reward + self.__gamma * next_q  # Compute target Q-value

        # SARSA update rule
        self.__q_table[current_state[0], current_state[1], action.value] += self.__alpha * (target_q - current_q)

        # Decay epsilon
        self.__decay_epsilon()
