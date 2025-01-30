import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from learning.mdp_grid_world.actions import Action
from typing import List, Tuple


class QLearningAgent:
    """
    A Q-learning agent that learns an optimal policy for navigating a GridWorld environment.

    Attributes:
        __state_space_size (int): Number of states in the environment.
        __action_space_size (int): Number of possible actions.
        __alpha (float): Learning rate for Q-learning updates.
        __gamma (float): Discount factor for future rewards.
        __actions (List[Action]): List of possible actions.
        __q_table (np.ndarray): Q-table storing the action-value estimates.
    """

    def __init__(self, state_space_size: int, action_space_size: int, learning_rate: float, discount_ratio: float):
        """
        Initializes the Q-learning agent with a Q-table and learning parameters.

        Args:
            state_space_size (int): Number of states in the environment.
            action_space_size (int): Number of possible actions.
            learning_rate (float): Learning rate (alpha) for updating Q-values.
            discount_ratio (float): Discount factor (gamma) for considering future rewards.
        """
        self.__state_space_size: int = state_space_size
        self.__action_space_size: int = action_space_size
        self.__alpha: float = learning_rate
        self.__gamma: float = discount_ratio
        self.__actions: List[Action] = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        self.__q_table: np.ndarray = self.__initialize_q_table(self.__state_space_size, self.__action_space_size)

    @staticmethod
    def __initialize_q_table(state_space_size: int, action_space_size: int) -> np.ndarray:
        """
        Initializes the Q-table with small random values.

        Args:
            state_space_size (int): Number of states in the environment.
            action_space_size (int): Number of possible actions.

        Returns:
            np.ndarray: A 3D Q-table initialized with random values.
        """
        return np.random.uniform(0, 1, (state_space_size, state_space_size, action_space_size))

    @property
    def learning_rate(self) -> float:
        """Returns the learning rate (alpha)."""
        return self.__alpha

    @property
    def discount_ratio(self) -> float:
        """Returns the discount factor (gamma)."""
        return self.__gamma

    @property
    def actions(self) -> List[Action]:
        """Returns the list of possible actions."""
        return self.__actions

    @property
    def q_table(self) -> np.ndarray:
        """Returns the current Q-table."""
        return self.__q_table

    def choose_action(self, current_state: Tuple[int, int]) -> Action:
        """
        Chooses the best action based on the Q-values for the current state.

        Args:
            current_state (Tuple[int, int]): The current state in the environment.

        Returns:
            Action: The best action based on the Q-table.
        """
        action_index: int = np.argmax(self.__q_table[current_state[0], current_state[1]])
        return Action(action_index)

    def update_q_values(self, current_state: Tuple[int, int], action: Action, reward: float, next_state: Tuple[int, int], done: bool) -> None:
        """
        Updates the Q-values using the Q-learning update rule.

        Args:
            current_state (Tuple[int, int]): The current state.
            action (Action): The action taken.
            reward (float): The received reward.
            next_state (Tuple[int, int]): The next state reached.
            done (bool): Whether the episode has ended.

        Returns:
            None
        """
        current_q: float = self.__q_table[current_state[0], current_state[1]][action.value]

        if done:
            target_q = reward  # Terminal state has no future rewards
        else:
            max_next_q = max(self.__q_table[next_state[0], next_state[1]])  # Best Q-value of next state
            target_q = reward + self.__gamma * max_next_q  # Compute target

        # Q-learning update rule
        self.__q_table[current_state[0], current_state[1]][action.value] += self.__alpha * (target_q - current_q)
