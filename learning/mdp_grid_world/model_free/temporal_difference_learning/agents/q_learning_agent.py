from typing import Tuple
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from learning.mdp_grid_world.grid_world_environment.actions import Action
from learning.mdp_grid_world.model_free.temporal_difference_learning.agents.agent import Agent


class QLearningAgent(Agent):
    """
    A Q-learning agent that learns an optimal policy for navigating a GridWorld environment.

    Attributes:
        __state_space_size Tuple[int, int]: Number of states in the environment.
        __action_space_size (int): Number of possible actions.
        _alpha (float): Learning rate for Q-learning updates.
        _gamma (float): Discount factor for future rewards.
        __actions (List[Action]): List of possible actions.
        _q_table (np.ndarray): Q-table storing the action-value estimates.
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
        Initializes the Q-learning agent with a Q-table and learning parameters.

        Args:
            state_space_size (int): Number of states in the environment.
            action_space_size (int): Number of possible actions.
            learning_rate (float): Learning rate (alpha) for updating Q-values.
            discount_ratio (float): Discount factor (gamma) for considering future rewards.
        """
        super(QLearningAgent, self).__init__(state_space_size, 
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
        done: bool
    ) -> None:
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
        current_q: float = self._q_table[current_state[0], current_state[1], action.value]

        if done:
            target_q = reward  # Terminal state has no future rewards
        else:
            max_next_q = max(self._q_table[next_state[0], next_state[1]])  # Best Q-value of next state
            target_q = reward + self._gamma * max_next_q  # Compute target

        # Q-learning update rule
        self._q_table[current_state[0], current_state[1], action.value] += self._alpha * (target_q - current_q)
        