from tqdm import tqdm
import rootutils
from typing import Tuple

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from learning.mdp_grid_world.environment import GridWorldEnvironment
from learning.mdp_grid_world.q_learning_agent import QLearningAgent
from learning.mdp_grid_world.actions import Action


class QLearningTrainer:
    """
    A class that encapsulates the Q-learning training and testing process.

    Attributes:
        env (GridWorldEnvironment): The GridWorld environment.
        agent (QLearningAgent): The Q-learning agent.
        num_episodes (int): Number of episodes for training.
    """

    def __init__(self, env: GridWorldEnvironment, agent: QLearningAgent, num_episodes: int) -> None:
        """
        Initializes the trainer with an environment, agent, and number of episodes.

        Args:
            env (GridWorldEnvironment): The GridWorld environment.
            agent (QLearningAgent): The Q-learning agent.
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
        """Returns the Q-learning agent."""
        return self.__agent
    
    @property
    def num_episodes(self) -> int:
        """Returns the number of episodes for training."""
        return self.__num_episodes

    def train(self) -> None:
        """
        Trains the Q-learning agent in the GridWorld environment.
        Updates the agent's Q-table over multiple episodes.
        """
        print("\n🔹 **Training Q-learning Agent**")
        for _ in tqdm(range(self.__num_episodes)):
            current_state = (0, 0)  # Start position
            done = False

            while not done:
                action = self.__agent.choose_action(current_state)
                next_state, reward, done = self.__env.step(current_state, action)
                self.__agent.update_q_values(current_state, action, reward, next_state, done)
                current_state = next_state  # Move to the next state

    def test(self, max_steps: int = 50) -> None:
        """
        Tests the trained Q-learning agent by running a single episode.

        Args:
            max_steps (int): The maximum number of steps to prevent infinite loops.
        """
        print("\n🔹 **Testing Trained Agent**")

        state = (0, 0)  # Start position
        total_reward = 0
        steps = 0

        while True:
            action = self.__agent.choose_action(state)  # Use learned policy
            next_state, reward, done = self.__env.step(state, action)
            total_reward += reward
            steps += 1

            print(f"Step {steps}: State {state} → Action {action} → Next State {next_state}, Reward {reward}")

            if done or steps >= max_steps:
                break
            state = next_state  # Move to the next state

        print(f"\n✅ **Test Completed: Total Reward: {total_reward}, Steps Taken: {steps}**")

        # Assertions to check if the agent learned correctly
        assert steps < max_steps, "❌ Test Failed: Agent did not reach goal efficiently."
        assert total_reward > 0, "❌ Test Failed: Agent did not collect positive rewards."


if __name__ == '__main__':
    env = GridWorldEnvironment()
    agent = QLearningAgent(state_space_size=4, action_space_size=4, learning_rate=0.1, discount_ratio=0.9)
    num_episodes = 1000

    trainer = QLearningTrainer(env, agent, num_episodes)
    trainer.train()  # Train the agent
    trainer.test()   # Test the trained agent
