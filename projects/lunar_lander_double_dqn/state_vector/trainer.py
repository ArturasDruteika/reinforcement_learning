from pathlib import Path

import numpy as np
import gymnasium
import torch
import rootutils
from tqdm import trange

rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from projects.lunar_lander_double_dqn.state_vector.agent import LunarLanderDoubleDQNAgent
from metrics_loggers.metric_logger import MetricLogger


class LunarLanderTrainer:
    """Trainer class for training a Double DQN agent on the LunarLander-v3 environment.

    This class manages the training loop, logging, model saving, 
    and visualization of the agent's performance. It uses a Double DQN 
    agent implementation to interact with the environment, update policies, 
    and track metrics across episodes.
    """
    
    def __init__(self, 
                 target_update_freq: int = 100, 
                 render_freq: int = 1000, 
                 model_weights_path: str | Path | None = None,
                 model_weights_dir_path: str | Path | None = None,
                 log_dir: str | Path = 'runs',
                 experiment_name: str = 'lunar_lander_double_dqn') -> None:
        """Initialize the LunarLanderTrainer.

        Args:
            target_update_freq (int): Frequency (in episodes) to update the target network.
            render_freq (int): Frequency (in episodes) to render the environment.
            model_weights_path (Path | None): Path to pretrained model weights if available.
            model_weights_dir_path (Path | None): Directory to save model weights.
            log_dir (Path): Directory to store training logs.
            experiment_name (str): Name of the experiment for logging.
        """
        self.__env = gymnasium.make('LunarLander-v3')
        self.__visual_env = None
        self.__agent = LunarLanderDoubleDQNAgent(
            state_size=self.__env.observation_space.shape[0], 
            action_space_size=self.__env.action_space.n, 
            model_weights_path=Path(model_weights_path) if model_weights_path else None,
            batch_size=64,
            memory_size=100_000,
            learning_rate=1e-4,
            min_epsilon=0.05,
            warmup_steps=10_000,
            epsilon_decay_steps=200_000
        )
        self.__target_update_freq = target_update_freq
        self.__render_freq = render_freq
        self.__model_weights_dir_path: Path | None = Path(model_weights_dir_path) if model_weights_dir_path else None
        self.__episode_losses = []
        self.__episode_rewards = []
        self.__logger: MetricLogger = MetricLogger(log_dir=str(log_dir), experiment_name=experiment_name)
        self.__learning_step = 0
        
    # ==========================
    # Private Methods
    # ==========================

    def __run_episode(self, 
                      max_steps: int = 1000) -> tuple[float, float, int, int]:
        """Run a single training episode.

        Args:
            max_steps (int): Maximum number of steps allowed in an episode.

        Returns:
            tuple[float, float, int, int]: 
                - total_reward (float): Cumulative reward from the episode.
                - total_loss (float): Sum of all losses during the episode.
                - loss_count (int): Number of loss updates performed.
                - step (int): Number of steps taken in the episode.
        """
        state, info = self.__env.reset()
        done = False
        truncated = False
        step = 0
        total_loss = 0.0
        loss_count = 0
        total_reward = 0.0

        while not (done or truncated) and step < max_steps:
            state_tensor = torch.from_numpy(state).float()
            action = self.__agent.choose_action(state_tensor, train_mode=True)
            new_state, reward, done, truncated, info = self.__env.step(action)
            reward -= 0.01  # Penalize for taking too long
            total_reward += reward
            
            terminal = done or truncated

            self.__agent.store_memory(
                state_tensor,
                action,
                reward,
                torch.from_numpy(new_state).float(),
                terminal
            )

            loss = self.__agent.learn(return_loss=True)
            if loss is not None:
                total_loss += loss.item()
                loss_count += 1

            state = new_state
            step += 1
            self.__learning_step += 1

        self.__agent.decay_epsilon(self.__learning_step)
        return total_reward, total_loss, loss_count, step

    def __update_metrics(self, 
                         total_reward: float, 
                         total_loss: float, 
                         loss_count: int, 
                         rolling_avg_episodes_count: int) -> tuple[float, float, float]:
        """Update rolling averages for loss and rewards.

        Args:
            total_reward (float): Reward accumulated in the episode.
            total_loss (float): Total loss accumulated in the episode.
            loss_count (int): Number of loss updates in the episode.
            rolling_avg_episodes_count (int): Number of episodes for rolling average.

        Returns:
            tuple[float, float, float]: 
                - avg_loss (float): Average loss for the episode.
                - rolling_avg_loss (float): Rolling average loss.
                - rolling_avg_reward (float): Rolling average reward.
        """
        if loss_count > 0:
            avg_loss = total_loss / loss_count
            self.__episode_losses.append(avg_loss)
            if len(self.__episode_losses) > rolling_avg_episodes_count:
                self.__episode_losses.pop(0)
            rolling_avg_loss = sum(self.__episode_losses) / len(self.__episode_losses)
        else:
            avg_loss = 0.0
            self.__episode_losses.append(0.0)
            if len(self.__episode_losses) > rolling_avg_episodes_count:
                self.__episode_losses.pop(0)
            rolling_avg_loss = sum(self.__episode_losses) / len(self.__episode_losses)

        self.__episode_rewards.append(total_reward)
        if len(self.__episode_rewards) > rolling_avg_episodes_count:
            self.__episode_rewards.pop(0)
        rolling_avg_reward = sum(self.__episode_rewards) / len(self.__episode_rewards)

        return avg_loss, rolling_avg_loss, rolling_avg_reward

    def __periodic_actions(self, 
                           episode: int, 
                           total_loss: float, 
                           loss_count: int, 
                           save_weights_freq: int) -> None:
        """Perform periodic updates and actions like saving weights or visualization.

        Args:
            episode (int): Current training episode.
            total_loss (float): Total loss accumulated in the episode.
            loss_count (int): Number of loss updates in the episode.
            save_weights_freq (int): Frequency (in episodes) to save model weights.
        """
        if loss_count > 0:
            avg_loss = total_loss / loss_count
            if episode % self.__target_update_freq == 0:
                print(f"Episode {episode} Loss: {avg_loss:.4f}")

        if episode % save_weights_freq == 0 and episode > 0 and self.__model_weights_dir_path is not None:
            self.__agent.save_model(f'{self.__model_weights_dir_path}/lunar_lander_dqn_{episode}.pt')

        if episode % self.__render_freq == 0 and self.__learning_step >= self.__agent.warmup_steps:
            self.visualize_agent(episode)
            
    # ==========================
    # Properties
    # ==========================

    @property
    def env(self):
        """gymnasium.Env: Training environment."""
        return self.__env

    @property
    def visual_env(self):
        """gymnasium.Env | None: Visualization environment."""
        return self.__visual_env

    @property
    def agent(self) -> LunarLanderDoubleDQNAgent:
        """LunarLanderDoubleDQNAgent: The learning agent."""
        return self.__agent

    @property
    def target_update_freq(self) -> int:
        """int: Frequency (episodes) for target network updates."""
        return self.__target_update_freq

    @property
    def model_weights_dir_path(self) -> Path | None:
        """Path | None: Directory path to save model weights."""
        return self.__model_weights_dir_path

    @property
    def log_dir(self) -> Path:
        """Path: Directory where logs are stored."""
        return self.__log_dir

    @property
    def episode_losses(self) -> list[float]:
        """list[float]: History of per-episode average losses."""
        return self.__episode_losses

    @property
    def episode_rewards(self) -> list[float]:
        """list[float]: History of per-episode total rewards."""
        return self.__episode_rewards

    @property
    def logger(self) -> MetricLogger:
        """MetricLogger: Logger used for metrics tracking."""
        return self.__logger

    @property
    def learning_step(self) -> int:
        """int: Number of learning steps completed so far."""
        return self.__learning_step
            
    # ==========================
    # Public Methods
    # ==========================

    def train(self, 
              episodes: int = 100_000, 
              max_steps: int = 1000, 
              weights_dir_path: str | Path = "projects/lunar_lander_double_dqn/state_vector/model_weights/",
              save_weights_freq: int = 1000, 
              rolling_avg_episodes_count: int = 50) -> None:
        """Train the Lunar Lander agent with a progress bar.

        Args:
            episodes (int): Number of training episodes.
            max_steps (int): Maximum number of steps per episode.
            save_weights_freq (int): Frequency (in episodes) to save model weights.
            rolling_avg_episodes_count (int): Number of episodes for rolling averages.
        """
        self.__model_weights_dir_path = Path(weights_dir_path)
        self.__model_weights_dir_path.mkdir(parents=True, exist_ok=True)
        
        with trange(episodes, desc='Training', unit='episode') as t:
            for episode in t:
                # Run a single episode
                total_reward, total_loss, loss_count, steps = self.__run_episode(max_steps)

                # Update metrics
                avg_loss, rolling_avg_loss, rolling_avg_reward = self.__update_metrics(
                    total_reward, 
                    total_loss, 
                    loss_count, 
                    rolling_avg_episodes_count
                )

                # Update progress bar
                t.set_postfix({
                    "Reward": f"{total_reward:.2f}",
                    "Total Loss": f"{total_loss:.4f}",
                    "Avg Loss": f"{avg_loss:.4f}",
                    f"Rolling Avg Reward ({rolling_avg_episodes_count})": f"{rolling_avg_reward:.2f}",
                    f"Rolling Avg Loss ({rolling_avg_episodes_count})": f"{rolling_avg_loss:.4f}"
                })
                
                self.__logger.log("Rolling Average Reward", rolling_avg_reward, episode)
                self.__logger.log("Rolling Average Loss", rolling_avg_loss, episode)

                # Perform periodic actions
                self.__periodic_actions(episode, total_loss, loss_count, save_weights_freq)

        self.__env.close()

    def visualize_agent(self, 
                        episode: int) -> None:
        """Run a test episode without exploration to visualize agent performance.

        Args:
            episode (int): Episode number at which visualization is performed.
        """
        print(f"\nVisualizing performance at episode {episode}")
        if self.__visual_env is None:
            self.__visual_env = gymnasium.make('LunarLander-v3', render_mode="human")

        state, info = self.__visual_env.reset()
        done = False
        max_steps = 1_000
        total_reward = 0

        for i in trange(max_steps, desc="Visualization", unit="step"):
            if done:
                break
            action = self.__agent.choose_action(torch.from_numpy(state).float())
            new_state, reward, done, truncated, info = self.__visual_env.step(action)
            total_reward += reward
            self.__visual_env.render()
            state = new_state

        self.__visual_env.close()
        self.__visual_env = None
        print(f"Visualization Total Reward: {total_reward:.2f}")


if __name__ == '__main__':
    trainer = LunarLanderTrainer(target_update_freq=80, render_freq=500)
    trainer.train(episodes=1_000_000)
