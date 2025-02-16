from typing import List, Tuple, Dict, Union
import random
import numpy as np
import rootutils
from tqdm import tqdm

rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from learning.mdp_grid_world.grid_world_environment.environment import GridWorldEnvironment, RewardValues
from learning.mdp_grid_world.grid_world_environment.actions import Action
from learning.mdp_grid_world.model_free.monte_carlo_methods.every_visit_monte_carlo import EveryVisitMonteCarlo


class Trainer:
    """
    A trainer class to handle the training and testing of the Every-Visit Monte Carlo agent
    in a GridWorld environment.
    
    Attributes:
        every_visit_monte_carlo (EveryVisitMonteCarlo): The Monte Carlo learning agent.
        environment (GridWorldEnvironment): The grid world environment.
        gamma (float): Discount factor (γ) for future rewards.
        alpha (float): Learning rate (α) for Q-value updates.
        epsilon (float): Exploration rate (ε) for action selection.
        epsilon_decay (float): Rate at which epsilon decays per episode.
        min_epsilon (float): Minimum possible epsilon value.
        n_episodes (int): Number of episodes to train the agent.
        max_steps_per_episode (int): Maximum number of steps per episode.
    """

    def __init__(
        self,
        every_visit_monte_carlo: EveryVisitMonteCarlo,
        environment: GridWorldEnvironment,
        gamma: float = 0.99,
        alpha: float = 1e-4,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
        n_episodes: int = 10_000,
        max_steps_per_episode: int = 1000
    ) -> None:
        """
        Initializes the Trainer class with the Monte Carlo agent and training parameters.

        Args:
            every_visit_monte_carlo (EveryVisitMonteCarlo): The Monte Carlo learning agent.
            environment (GridWorldEnvironment): The environment where training takes place.
            gamma (float, optional): Discount factor for future rewards. Defaults to 0.99.
            alpha (float, optional): Learning rate for Q-value updates. Defaults to 1e-4.
            epsilon (float, optional): Initial exploration rate. Defaults to 0.1.
            epsilon_decay (float, optional): Decay factor for epsilon. Defaults to 0.995.
            min_epsilon (float, optional): Minimum possible epsilon value. Defaults to 0.01.
            n_episodes (int, optional): Number of episodes to train. Defaults to 10,000.
            max_steps_per_episode (int, optional): Max steps per episode. Defaults to 1000.
        """
        self.__every_visit_monte_carlo: EveryVisitMonteCarlo = every_visit_monte_carlo
        self.__env: GridWorldEnvironment = environment
        self.__gamma: float = gamma
        self.__alpha: float = alpha
        self.__epsilon: float = epsilon
        self.__epsilon_decay: float = epsilon_decay
        self.__min_epsilon: float = min_epsilon
        self.__n_episodes: int = n_episodes
        self.__max_steps_per_episode: int = max_steps_per_episode
        
    @property
    def every_visit_monte_carlo(self) -> EveryVisitMonteCarlo:
        """Returns the Monte Carlo learning agent."""
        return self.__every_visit_monte_carlo
    
    @property
    def environment(self) -> GridWorldEnvironment:
        """Returns the environment."""
        return self.__env
    
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
        """Returns the epsilon decay rate."""
        return self.__epsilon_decay
    
    @property
    def min_epsilon(self) -> float:
        """Returns the minimum epsilon value."""
        return self.__min_epsilon
    
    @property
    def n_episodes(self) -> int:
        """Returns the number of training episodes."""
        return self.__n_episodes
    
    @property
    def max_steps_per_episode(self) -> int:
        """Returns the maximum number of steps per episode."""
        return self.__max_steps_per_episode
    
    def train(self) -> None:
        """
        Trains the Monte Carlo agent in the GridWorld environment for a given number of episodes.

        - The agent follows an ε-greedy policy.
        - Each episode runs until the agent reaches a terminal state (goal or hole).
        - Returns (G_t) are computed for each state-action pair in the episode.
        - Q-values are updated using Every-Visit Monte Carlo learning.
        """
        start_state: Tuple[int, int] = (0, 0)
    
        for _ in tqdm(range(self.__n_episodes), desc="Training Progress"):
            current_state: Tuple[int, int] = start_state  # Reset state at beginning of episode
            reward: float = 0  # Ensure reward is reset
            episodes_info_list: List[Dict[str, Union[Tuple[int, int], Action, float, bool]]] = []

            while reward not in [RewardValues.HOLE_STATE_VALUE, RewardValues.GOAL_STATE_VALUE]:
                action: Action = self.__every_visit_monte_carlo.choose_action(current_state)
                next_state, reward, done = self.__env.step(current_state, action)

                # Store transition in episode history
                episodes_info_list.append({
                    'state': current_state,
                    'action': action,
                    'next_state': next_state,
                    'reward': reward,
                    'done': done
                })

                current_state = next_state

            # Compute returns (G_t)
            G: float = 0  
            gamma: float = self.__gamma  
            for t in reversed(range(len(episodes_info_list))):
                G = episodes_info_list[t]['reward'] + gamma * G  # Compute G_t
                episodes_info_list[t]['G'] = G  

            # Learn from episode
            self.__every_visit_monte_carlo.learn(episodes_info_list)
            
            # Decay epsilon
            self.__every_visit_monte_carlo.decay_epsilon()
            
    def test(self) -> None:
        """
        Tests the trained Monte Carlo agent by running it in the environment.
        
        - The agent follows the learned policy.
        - The trajectory is printed step by step.
        """
        current_state: Tuple[int, int] = (0, 0)  # Start from the top-left corner
        reward: float = 0
        done: bool = False

        print("\nTesting the learned policy...\n")

        while not done:
            action: Action = self.__every_visit_monte_carlo.choose_action(current_state)
            next_state, reward, done = self.__env.step(current_state, action)
            print(f"State: {current_state}, Action: {action.name}, Reward: {reward}, Next State: {next_state}, Done: {done}")
            current_state = next_state
            

if __name__ == '__main__':
    # Create GridWorld environment
    env = GridWorldEnvironment(0.5)

    # Initialize Every-Visit Monte Carlo agent
    every_visit_monte_carlo = EveryVisitMonteCarlo(
        state_space_size=env.grid.shape,
        action_state_size=len(Action),
        gamma=0.9,
        alpha=0.1,
        epsilon=0.1,
        epsilon_decay=0.995,
        min_epsilon=0.01,
    )
    
    # Initialize trainer and run training & testing
    trainer = Trainer(every_visit_monte_carlo, env)
    trainer.train()
    trainer.test()
