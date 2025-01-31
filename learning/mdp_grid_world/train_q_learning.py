import numpy as np
from tqdm import tqdm
import rootutils
from typing import Tuple

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from learning.mdp_grid_world.environment import GridWorldEnvironment
from learning.mdp_grid_world.q_learning_agent import QLearningAgent
from learning.mdp_grid_world.sarsa_agent import SarsaAgent


class QLearningTrainer:
    """
    A class that encapsulates the Q-learning and SARSA training/testing process.

    Attributes:
        env (GridWorldEnvironment): The GridWorld environment.
        agent (QLearningAgent | SarsaAgent): The learning agent.
        num_episodes (int): Number of episodes for training.
    """

    def __init__(self, env: GridWorldEnvironment, agent: QLearningAgent, num_episodes: int) -> None:
        """
        Initializes the trainer with an environment, agent, and number of episodes.

        Args:
            env (GridWorldEnvironment): The GridWorld environment.
            agent (QLearningAgent | SarsaAgent): The learning agent.
            num_episodes (int): Number of episodes for training.
        """
        self.__env = env
        self.__agent = agent
        self.__num_episodes = num_episodes

    @property
    def env(self) -> GridWorldEnvironment:
        """Returns the GridWorld environment."""
        return self.__env

    @property
    def agent(self) -> QLearningAgent:
        """Returns the learning agent (Q-learning or SARSA)."""
        return self.__agent

    @property
    def num_episodes(self) -> int:
        """Returns the number of episodes for training."""
        return self.__num_episodes

    def train(self, epsilon: float = 1e-4, q_learning=True) -> None:
        """
        Trains the agent using Q-learning or SARSA in the GridWorld environment.
        Implements early stopping when Q-values stabilize (i.e., updates become smaller than `epsilon`).

        Args:
            epsilon (float): Threshold for Q-value convergence. If the change is below this, stop training.
            q_learning (bool): If True, trains using Q-learning; otherwise, trains using SARSA.
        """
        print("\n🔹 **Training Agent with Early Stopping**")

        for episode in tqdm(range(self.__num_episodes), desc="Training Progress"):
            prev_q_table = np.copy(self.__agent.q_table)  # Save previous Q-table state
            current_state = (0, 0)  # Start position
            done = False
            
            if not q_learning:
                action = self.__agent.choose_action(current_state)

            while not done:
                if q_learning:
                    # Q-learning: Select action, take step, update Q-values
                    action = self.__agent.choose_action(current_state)  # Get current action
                    next_state, reward, done = self.__env.step(current_state, action)
                    self.__agent.update_q_values(current_state, action, reward, next_state, done)
                else:
                    # SARSA: Select action, take step, get next action, update Q-values
                    next_state, reward, done = self.__env.step(current_state, action)
                    next_action = self.__agent.choose_action(next_state)  # Get next action
                    self.__agent.update_q_values(current_state, action, reward, next_state, next_action, done)
                    action = next_action  # Update action for next loop

                current_state = next_state  # Move to next state

            # Compute max Q-value change across the table
            max_q_change = np.max(np.abs(self.__agent.q_table - prev_q_table))

            if max_q_change < epsilon:  # Stop if Q-values stabilize
                print(f"\n✅ Training stopped early at episode {episode} (Q-values converged).")
                break

    def test(self, max_steps: int = 50, q_learning: bool = True) -> None:
        """
        Tests the trained Q-learning or SARSA agent by running a single episode.

        Args:
            max_steps (int): The maximum number of steps to prevent infinite loops.
            q_learning (bool): If True, uses Q-learning logic; otherwise, uses SARSA.

        Returns:
            None
        """
        print("\n🔹 **Testing Trained Agent**")

        state = (0, 0)  # Start position
        total_reward = 0
        steps = 0

        # Select the initial action for SARSA
        action = self.__agent.choose_action(state)

        while True:
            # Take action, observe next state and reward
            next_state, reward, done = self.__env.step(state, action)
            total_reward += reward
            steps += 1

            print(f"Step {steps}: State {state} → Action {action} → Next State {next_state}, Reward {reward}")

            if done or steps >= max_steps:
                break

            # Select the next action for SARSA (on-policy learning)
            if q_learning:
                action = self.__agent.choose_action(next_state)  # Q-learning selects greedy action
            else:
                next_action = self.__agent.choose_action(next_state)  # SARSA selects next action
                action = next_action  # Continue with SARSA-style updating

            state = next_state  # Move to the next state

        print(f"\n✅ **Test Completed: Total Reward: {total_reward}, Steps Taken: {steps}**")

        # Assertions to check if the agent learned correctly
        assert steps < max_steps, "❌ Test Failed: Agent did not reach goal efficiently."
        assert total_reward > 0, "❌ Test Failed: Agent did not collect positive rewards."


if __name__ == '__main__':
    env = GridWorldEnvironment()
    state_size = env.grid.shape
    agent = QLearningAgent(
        state_space_size=state_size,
        action_space_size=4,
        learning_rate=0.1,
        discount_ratio=0.9
    )
    num_episodes = 100000

    trainer = QLearningTrainer(env, agent, num_episodes)
    trainer.train(q_learning=True)  # Train the SARSA agent
    trainer.test(q_learning=True)   # Test the SARSA agent
