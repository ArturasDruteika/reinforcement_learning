from enum import Enum
from typing import Tuple, Union

import numpy as np
from tqdm import tqdm
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from learning.mdp_grid_world.grid_world_environment.environment import GridWorldEnvironment
from learning.mdp_grid_world.model_free.temporal_difference_learning.agents.q_learning_agent import QLearningAgent
from learning.mdp_grid_world.model_free.temporal_difference_learning.agents.sarsa_agent import SarsaAgent
from learning.mdp_grid_world.model_free.temporal_difference_learning.agents.agent import Agent


class TrainingMode(Enum):
    """Enumeration for training modes."""
    SARSA = 0
    Q_LEARNING = 1


class Trainer:
    """
    A trainer class for Q-learning and SARSA agents in a GridWorld environment.

    Attributes:
        env (GridWorldEnvironment): The GridWorld environment.
        agent (Union[QLearningAgent, SarsaAgent]): The learning agent.
        num_episodes (int): Number of episodes for training.
        training_mode (TrainingMode): The training mode (SARSA or Q-learning).
    """

    def __init__(self, env: GridWorldEnvironment, agent: Agent, num_episodes: int) -> None:
        """
        Initializes the trainer with an environment, agent, and number of episodes.

        Args:
            env (GridWorldEnvironment): The GridWorld environment.
            agent (Union[QLearningAgent, SarsaAgent]): The learning agent.
            num_episodes (int): Number of episodes for training.
        """
        self.__env: GridWorldEnvironment = env
        self.__agent: Agent = agent
        self.__num_episodes: int = num_episodes
        self.__training_mode: TrainingMode = TrainingMode.SARSA if isinstance(self.__agent, SarsaAgent) else TrainingMode.Q_LEARNING

    @property
    def env(self) -> GridWorldEnvironment:
        """Returns the GridWorld environment."""
        return self.__env

    @property
    def agent(self) -> Agent:
        """Returns the learning agent (Q-learning or SARSA)."""
        return self.__agent

    @property
    def num_episodes(self) -> int:
        """Returns the number of episodes for training."""
        return self.__num_episodes
    
    @property
    def training_mode(self) -> TrainingMode:
        """Returns the training mode (Q-learning or SARSA)."""
        return self.__training_mode
    
    def train(self, epsilon: float = 1e-4) -> None:
        """
        Trains the agent using Q-learning or SARSA in the GridWorld environment.
        Implements early stopping when Q-values stabilize (i.e., updates become smaller than `epsilon`).

        Args:
            epsilon (float, optional): Threshold for Q-value convergence. If the change is below this, stop training.
        """
        print("\n🔹 **Training Agent with Early Stopping**")

        for episode in tqdm(range(self.__num_episodes), desc="Training Progress"):
            prev_q_table = np.copy(self.__agent.q_table)  # Save previous Q-table state
            current_state: Tuple[int, int] = (0, 0)  # Start position
            done: bool = False
            
            if self.__training_mode == TrainingMode.SARSA:
                action: int = self.__agent.choose_action(current_state)

            while not done:
                if self.__training_mode == TrainingMode.Q_LEARNING:
                    action = self.__agent.choose_action(current_state)  # Get current action
                    next_state, reward, done = self.__env.step(current_state, action)
                    self.__agent.update_q_values(current_state, action, reward, next_state, done)
                else:  # SARSA
                    next_state, reward, done = self.__env.step(current_state, action)
                    next_action = self.__agent.choose_action(next_state)
                    self.__agent.update_q_values(current_state, action, reward, next_state, next_action, done)
                    action = next_action  # Update action for next loop

                current_state = next_state  # Move to next state
            
            self.__agent.decay_epsilon()

            max_q_change = np.max(np.abs(self.__agent.q_table - prev_q_table))
            if max_q_change < epsilon:
                print(f"\n✅ Training stopped early at episode {episode} (Q-values converged).")
                break

    def test(self, max_steps: int = 50) -> None:
        """
        Tests the trained agent by running a single episode.

        Args:
            max_steps (int, optional): The maximum number of steps to prevent infinite loops.
        """
        print("\n🔹 **Testing Trained Agent**")

        state: Tuple[int, int] = (0, 0)  # Start position
        total_reward: float = 0
        steps: int = 0
        
        action: int = self.__agent.choose_action(state)

        while True:
            next_state, reward, done = self.__env.step(state, action)
            total_reward += reward
            steps += 1

            print(f"Step {steps}: State {state} → Action {action} → Next State {next_state}, Reward {reward}")

            if done or steps >= max_steps:
                break

            if self.__training_mode == TrainingMode.Q_LEARNING:
                action = self.__agent.choose_action(next_state)
            else:  # SARSA
                next_action = self.__agent.choose_action(next_state)
                action = next_action

            state = next_state

        print(f"\n✅ **Test Completed: Total Reward: {total_reward}, Steps Taken: {steps}**")

        assert steps < max_steps, "❌ Test Failed: Agent did not reach goal efficiently."
        assert total_reward > 0, "❌ Test Failed: Agent did not collect positive rewards."


if __name__ == '__main__':
    env = GridWorldEnvironment(transition_prob=0.9)
    state_size = env.grid.shape
    sarsa_agent = SarsaAgent(
        state_space_size=state_size,
        action_space_size=4,
        learning_rate=0.01,
        discount_ratio=0.9,
    )
    
    q_learning_agent = QLearningAgent(
        state_space_size=state_size,
        action_space_size=4,
        learning_rate=0.01,
        discount_ratio=0.9,
    )
    num_episodes = 10000
    
    trainer_sarsa = Trainer(env, sarsa_agent, num_episodes)
    trainer_sarsa.train()
    trainer_sarsa.test()
    
    trainer_q_learning = Trainer(env, q_learning_agent, num_episodes)
    trainer_q_learning.train()
    trainer_q_learning.test()
