from typing import Tuple
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from learning.mdp_grid_world.grid_world_environment.actions import Action
from learning.mdp_grid_world.temporal_difference_learning.agents.agent import Agent


class SarsaAgent(Agent):
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
        super(SarsaAgent, self).__init__(state_space_size, 
                                         action_space_size, 
                                         learning_rate, 
                                         discount_ratio, 
                                         epsilon, epsilon_decay, 
                                         min_epsilon)

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
        current_q: float = self._q_table[current_state[0], current_state[1], action.value]

        if done:
            target_q = reward  # No future rewards if terminal state
        else:
            next_q = self._q_table[next_state[0], next_state[1], next_action.value]  # Use SARSA rule
            target_q = reward + self._gamma * next_q  # Compute target Q-value

        # SARSA update rule
        self._q_table[current_state[0], current_state[1], action.value] += self._alpha * (target_q - current_q)

        # Decay epsilon
        self._decay_epsilon()
