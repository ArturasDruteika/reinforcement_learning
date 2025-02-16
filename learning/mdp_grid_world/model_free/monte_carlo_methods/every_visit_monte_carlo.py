from typing import List, Tuple, Dict, Union
import random

import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from learning.mdp_grid_world.grid_world_environment.actions import Action


class EveryVisitMonteCarlo:
    """
    Implements the Every-Visit Monte Carlo algorithm for Reinforcement Learning.

    Attributes:
        state_space_size (Tuple[int, int]): The dimensions of the state space.
        action_state_size (int): The number of possible actions.
        gamma (float): Discount factor (γ) for future rewards.
        alpha (float): Learning rate (α) for Q-value updates.
        epsilon (float): Exploration rate (ε) for ε-greedy action selection.
        epsilon_decay (float): Rate at which ε decays per episode.
        min_epsilon (float): Minimum value that ε can decay to.
        q_table (np.ndarray): Q-value table mapping state-action pairs to expected returns.
    """

    def __init__(
        self, 
        state_space_size: Tuple[int, int], 
        action_state_size: int,
        gamma: float,
        alpha: float,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01
    ):
        """
        Initializes the Monte Carlo agent with given hyperparameters.

        Args:
            state_space_size (Tuple[int, int]): The dimensions of the state space.
            action_state_size (int): The number of possible actions.
            gamma (float): Discount factor for future rewards.
            alpha (float): Learning rate for Q-value updates.
            epsilon (float, optional): Initial exploration rate. Defaults to 1.0.
            epsilon_decay (float, optional): Rate of decay for ε. Defaults to 0.995.
            min_epsilon (float, optional): Minimum possible ε value. Defaults to 0.01.
        """
        self.__state_space_size: Tuple[int, int] = state_space_size
        self.__action_state_size: int = action_state_size
        self.__gamma: float = gamma
        self.__alpha: float = alpha
        self.__epsilon: float = epsilon
        self.__epsilon_decay: float = epsilon_decay
        self.__min_epsilon: float = min_epsilon
        self.__q_table: np.ndarray = self.__initialize_q_table(state_space_size, action_state_size)
    
    @staticmethod   
    def __initialize_q_table(state_space_size: Tuple[int, int], action_space_size: int) -> np.ndarray:
        """
        Initializes the Q-table with small random values.

        Args:
            state_space_size (Tuple[int, int]): The state space dimensions.
            action_space_size (int): The number of possible actions.

        Returns:
            np.ndarray: A 3D Q-table initialized with zeros.
        """
        return np.zeros((state_space_size[0], state_space_size[1], action_space_size))
    
    def __update_state_action_value(self, state: Tuple[int, int], action: Action, cumulative_return: float) -> None:
        """
        Updates the Q-value for a given state-action pair using Monte Carlo estimation.

        Args:
            state (Tuple[int, int]): The state to update.
            action (Action): The action taken.
            cumulative_return (float): The computed return (G) from that state-action pair.
        """
        self.__q_table[state[0], state[1], action.value] += self.__alpha * (
            cumulative_return - self.__q_table[state[0], state[1], action.value]
        )
    
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
        """Returns the Q-table containing state-action values."""
        return self.__q_table

    def choose_action(self, current_state: Tuple[int, int]) -> Action:
        """
        Chooses an action using an ε-greedy policy.

        Args:
            current_state (Tuple[int, int]): The current state.

        Returns:
            Action: The action chosen according to the ε-greedy strategy.
        """
        if np.random.rand() < self.__epsilon:
            # Exploration: Choose a random action
            return random.choice(list(Action))
        else:
            # Exploitation: Choose the best action from the Q-table (with tie-breaking)
            best_actions = np.where(self.__q_table[current_state[0], current_state[1]] == 
                                    np.max(self.__q_table[current_state[0], current_state[1]]))[0]
            action_index = np.random.choice(best_actions)  # Break ties randomly
            return Action(action_index)
        
    def decay_epsilon(self) -> None:
        """
        Decays the epsilon value over time to favor exploitation over exploration.
        """
        self.__epsilon = max(self.__epsilon * self.__epsilon_decay, self.__min_epsilon)
    
    def learn(self, episodes_info_list: List[Dict[str, Union[Tuple[int, int], Action, float, bool]]]) -> None:
        """
        Updates the Q-values using the Every-Visit Monte Carlo method.

        Args:
            episodes_info_list (List[Dict[str, Union[Tuple[int, int], Action, float, bool]]]):
                A list of dictionaries, where each dictionary represents a timestep with:
                - 'state' (Tuple[int, int]): The state visited.
                - 'action' (Action): The action taken.
                - 'G' (float): The return G computed from that state-action pair.
                - 'done' (bool): Whether the episode has ended.

        Notes:
            - Every state-action pair in the episode is updated (Every-Visit MC).
            - If First-Visit MC is preferred, we would only update the first occurrence.
        """
        visited_state_actions = set()

        for episode_info_dict in episodes_info_list:
            state = episode_info_dict['state']
            action = episode_info_dict['action']
            cumulative_return = episode_info_dict['G']

            # Every-Visit MC: Updates all occurrences of state-action pairs in an episode
            if (state, action) not in visited_state_actions:
                visited_state_actions.add((state, action))
                self.__update_state_action_value(state, action, cumulative_return)
