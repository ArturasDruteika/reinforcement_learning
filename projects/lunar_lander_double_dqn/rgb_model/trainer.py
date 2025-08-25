from pathlib import Path
from typing import Optional, Tuple, List

import gymnasium as gym
from PIL import Image
import torch
import rootutils
from tqdm import trange, tqdm

rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from projects.lunar_lander_double_dqn.rgb_model.agent import LunarLanderDoubleDQNAgent
from projects.lunar_lander_dqn.rgb_mode.frame_processor import FramePreprocessor
from projects.lunar_lander_dqn.rgb_mode.frame_stacker import FrameStacker
from metrics_loggers.metric_logger import MetricLogger


class LunarLanderTrainer:
    """Train a DQN agent on LunarLander-v3 using stacked grayscale frames.

    This class manages the training loop, frame preprocessing/stacking,
    logging, periodic evaluation, and model checkpointing.

    Attributes:
        __image_shape (Tuple[int, int]): The shape of preprocessed frames (height, width).
        __stack_size (int): Number of frames to stack for state representation.
        __state_size (Tuple[int, int, int]): Shape of the state input (stack_size, height, width).
        __frame_preprocessor (FramePreprocessor): Preprocessor for resizing and processing frames.
        __frame_stacker (FrameStacker): Stacker for creating state from multiple frames.
        __env (gym.Env): Training environment for Lunar Lander.
        __test_env (gym.Env): Testing environment with human-readable rendering.
        __agent (LunarLanderDQNAgent): DQN agent for learning and decision-making.
        __episode_losses (List[float]): List of total losses per episode for rolling average.
        __episode_rewards (List[float]): List of total rewards per episode for rolling average.
        __logger (MetricLogger): Logger for tracking and saving training metrics.
    """

    def __init__(self, 
                 model_weights_path: Optional[str] = None,
                 log_dir: str = "runs",
                 experiment_name: str = "lunar_lander") -> None:
        """Initialize the trainer with environment, agent, and logging.

        Args:
            model_weights_path (Optional[str]): Path to pre-trained model weights.
            log_dir (str): Directory for saving training logs.
            experiment_name (str): Name of the experiment for logging.
        """
        self.__image_shape: Tuple[int, int] = (96, 96)  # Assuming 96x96 RGB frames
        self.__stack_size: int = 4
        self.__state_size: Tuple[int, int, int] = (self.__stack_size, *self.__image_shape)
        self.__frame_preprocessor: FramePreprocessor = FramePreprocessor(resize_shape=self.__image_shape)
        self.__frame_stacker: FrameStacker = FrameStacker(stack_length=self.__stack_size, image_shape=(1, *self.__image_shape))  # Stack 4 frames
        self.__env: gym.Env = gym.make('LunarLander-v3', render_mode='rgb_array')
        self.__test_env: gym.Env = gym.make('LunarLander-v3', render_mode='human')
        self.__agent: LunarLanderDoubleDQNAgent = LunarLanderDoubleDQNAgent(
            state_size=self.__state_size, 
            action_state_size=self.__env.action_space.n, 
            model_weights_path=model_weights_path,
            learning_rate=5e-5,
            memory_size=20_000,
            sync_target_every=8_000,
            min_epsilon=0.05,
            batch_size=256
        )
        self.__episode_losses: List[float] = []  # List to store total losses for rolling average
        self.__episode_rewards: List[float] = []  # List to store total rewards for rolling average
        self.__logger: MetricLogger = MetricLogger(log_dir=log_dir, experiment_name=experiment_name)
        
    # ==========================
    # Private Methods
    # ==========================

    def __initialize_episode(self) -> None:
        """Reset env and prime the frame stack with the first processed frame."""
        self.__env.reset()
        self.__frame_stacker.clear_stack()
        frame: Image.Image = Image.fromarray(self.__env.render()).convert('L')
        f0: torch.Tensor = self.__frame_preprocessor.preprocess(frame)
        for _ in range(self.__stack_size):
            self.__frame_stacker.push(f0.clone())

    def __process_step(self) -> tuple[float, bool, bool]:
        """Perform one environment step and store the transition.

        Builds s_t from the frame stack, selects an action, steps the env,
        builds s_{t+1}, stores the transition, and advances the stack.

        Returns:
            tuple[float, bool, bool]: (reward, done, truncated)
        """
        state = self.__frame_stacker.get_stacked_frames()   # (4, H, W)

        action = self.__agent.choose_action(state, True)
        _, reward, done, truncated, _ = self.__env.step(action)

        next_frame = self.__env.render()
        next_image = Image.fromarray(next_frame).convert('L')
        preprocessed_next = self.__frame_preprocessor.preprocess(next_image)   # (1, H, W)

        next_state = torch.cat([state[1:], preprocessed_next], dim=0)   # (4, H, W)

        terminal = done or truncated
        self.__agent.store_memory(state, action, reward, next_state, terminal)

        self.__frame_stacker.push(preprocessed_next)

        return reward, done, truncated

    def __learn_from_step(self) -> tuple[float, int]:
        """Run one learner update (if ready) and return accumulated loss info.

        Returns:
            tuple[float, int]: (total_loss, loss_count) for this step.
        """
        total_loss = 0.0
        loss_count = 0

        loss = self.__agent.learn(return_loss=True)
        if loss is not None:
            total_loss += loss
            loss_count += 1
        return total_loss, loss_count

    def __calculate_metrics(self, 
                            total_loss: float, 
                            loss_count: int, 
                            total_reward: float, 
                            rolling_avg_episodes_count: int) -> tuple[float, float, float]:
        """Compute per-episode averages and rolling averages for loss/reward.

        Args:
            total_loss (float): Cumulative loss for the episode.
            loss_count (int): Number of loss updates this episode.
            total_reward (float): Total episode reward.
            rolling_avg_episodes_count (int): Window size for rolling averages.

        Returns:
            tuple[float, float, float]: (avg_loss, rolling_avg_loss, rolling_avg_reward)
        """
        if loss_count > 0:
            avg_loss = total_loss / loss_count
            self.__episode_losses.append(total_loss)
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

    def __update_progress_bar(self,
                              t: tqdm,
                              total_reward: float,
                              total_loss: float,
                              avg_loss: float,
                              rolling_avg_reward: float,
                              rolling_avg_loss: float,
                              rolling_avg_episodes_count: int) -> None:
        """Update tqdm progress bar with current episode metrics.

        Args:
            t (tqdm): Progress bar to update.
            total_reward (float): Total reward in the episode.
            total_loss (float): Total loss in the episode.
            avg_loss (float): Average loss per update in the episode.
            rolling_avg_reward (float): Rolling average reward.
            rolling_avg_loss (float): Rolling average loss.
            rolling_avg_episodes_count (int): Episodes considered for rolling averages.
        """
        t.set_postfix({
            "Reward": f"{total_reward:.2f}",
            "Total Loss": f"{total_loss:.4f}",
            "Avg Loss": f"{avg_loss:.4f}",
            f"Rolling Avg Reward ({rolling_avg_episodes_count})": f"{rolling_avg_reward:.2f}",
            f"Rolling Avg Loss ({rolling_avg_episodes_count})": f"{rolling_avg_loss:.4f}"
        })
        
    # ==========================
    # Properties (Public Getters)
    # ==========================

    @property
    def image_shape(self) -> Tuple[int, int]:
        """Tuple[int, int]: Shape of preprocessed frames (H, W)."""
        return self.__image_shape

    @property
    def stack_size(self) -> int:
        """int: Number of stacked frames used as input."""
        return self.__stack_size

    @property
    def state_size(self) -> Tuple[int, int, int]:
        """Tuple[int, int, int]: Input state shape (stack_size, H, W)."""
        return self.__state_size

    @property
    def frame_preprocessor(self) -> FramePreprocessor:
        """FramePreprocessor: Utility for resizing and preprocessing frames."""
        return self.__frame_preprocessor

    @property
    def frame_stacker(self) -> FrameStacker:
        """FrameStacker: Utility for stacking frames into state representation."""
        return self.__frame_stacker

    @property
    def env(self) -> gym.Env:
        """gym.Env: Training environment (rgb_array render mode)."""
        return self.__env

    @property
    def test_env(self) -> gym.Env:
        """gym.Env: Testing environment (human render mode)."""
        return self.__test_env

    @property
    def agent(self) -> LunarLanderDoubleDQNAgent:
        """LunarLanderDQNAgent: The DQN agent being trained."""
        return self.__agent

    @property
    def episode_losses(self) -> List[float]:
        """List[float]: History of episode losses (for rolling averages)."""
        return self.__episode_losses

    @property
    def episode_rewards(self) -> List[float]:
        """List[float]: History of episode rewards (for rolling averages)."""
        return self.__episode_rewards

    @property
    def logger(self) -> MetricLogger:
        """MetricLogger: Logger for metrics tracking and saving."""
        return self.__logger
    
    # ==========================
    # Public Methods
    # ==========================

    def test_visually(self, 
                      max_steps: int = 1000) -> None:
        """Run a visually rendered episode to inspect the agent's behavior.

        Args:
            max_steps (int): Maximum steps to run the test episode.
        """
        self.__test_env.reset()
        self.__env.reset()
        self.__frame_stacker.clear_stack()
        done = False
        truncated = False
        step = 0
        total_reward = 0

        print("Starting visual test episode...")
        while not (done or truncated) and step < max_steps:
            frame = self.__env.render()
            image = Image.fromarray(frame).convert('L')
            preprocessed_frame = self.__frame_preprocessor.preprocess(image)
            self.__frame_stacker.push(preprocessed_frame)

            if self.__frame_stacker.is_full():
                state = self.__frame_stacker.get_stacked_frames()
                action = self.__agent.choose_action(state)
                observation, reward, done, truncated, info = self.__test_env.step(action)
                self.__env.step(action)  # Keep environments in sync
                total_reward += reward

            step += 1

        print(f"Visual test episode finished. Total Reward: {total_reward}")

    def train(self,
              n_episodes: int,
              max_steps: int = 1000,
              display_interval: int = 50,
              rolling_avg_episodes_count: int = 50,
              weights_dir: str | Path = "projects/lunar_lander_double_dqn/rgb_mode/model_weights",
              weights_prefix: str = "lunar_lander_double_dqn") -> None:
        """Train the agent and optionally run periodic visual tests.

        Args:
            n_episodes (int): Number of training episodes.
            max_steps (int): Maximum steps per episode.
            display_interval (int): How often (in episodes) to visually test the model.
            rolling_avg_episodes_count (int): Window for rolling averages.
            weights_dir (str | Path): Directory for saving model weights.
            weights_prefix (str): Filename prefix for saved weight files.
        """
        weights_dir = Path(weights_dir)
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        total_steps = 0

        with trange(n_episodes, desc='Training', unit='episode') as t:
            for episode in t:
                self.__initialize_episode()
                done = False
                truncated = False
                step = 0
                total_reward = 0.0
                total_loss = 0.0
                loss_count = 0

                while not (done or truncated) and step < max_steps:
                    reward, done, truncated = self.__process_step()
                    reward -= 0.1  # Penalize longer episodes

                    step_loss, step_loss_count = self.__learn_from_step()
                    total_loss += step_loss
                    loss_count += step_loss_count
                    total_reward += reward

                    step += 1
                    total_steps += 1
                    self.__agent.decay_epsilon(total_steps)

                avg_loss, rolling_avg_loss, rolling_avg_reward = self.__calculate_metrics(
                    total_loss,
                    loss_count,
                    total_reward,
                    rolling_avg_episodes_count
                )
                self.__update_progress_bar(
                    t, total_reward, total_loss, avg_loss,
                    rolling_avg_reward, rolling_avg_loss, rolling_avg_episodes_count
                )
                
                self.__logger.log("Rolling Average Reward", rolling_avg_reward, episode)
                self.__logger.log("Rolling Average Loss", rolling_avg_loss, episode)

                # Save at intervals (and on the last episode), only after memory is warm
                should_save = ((episode + 1) % display_interval == 0 or episode == n_episodes - 1)
                if should_save and self.__agent.replay_memory.is_full:
                    # Optional visual test
                    self.test_visually(max_steps=max_steps)

                    weights_path = weights_dir / f"{weights_prefix}_{episode + 1}.pt"
                    self.__agent.save_model(str(weights_path))

        self.__env.close()
        self.__test_env.close()


# Run the trainer
if __name__ == '__main__':
    trainer = LunarLanderTrainer()
    trainer.train(n_episodes=100_000,
                  display_interval=50,
                  rolling_avg_episodes_count=50)
