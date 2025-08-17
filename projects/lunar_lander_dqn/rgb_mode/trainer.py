from pathlib import Path

import gymnasium as gym
from PIL import Image
import torch
import rootutils
from tqdm import trange

rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from projects.lunar_lander_dqn.rgb_mode.agent import LunarLanderDQNAgent
from projects.lunar_lander_dqn.rgb_mode.frame_processor import FramePreprocessor
from projects.lunar_lander_dqn.rgb_mode.frame_stacker import FrameStacker
from projects.lunar_lander_dqn.rgb_mode.metric_logger import MetricLogger


class LunarLanderTrainer:
    def __init__(self, 
                 model_weights_path=None,
                 log_dir: str = "runs",
                 experiment_name: str = "lunar_lander"):
        self.__image_shape = (96, 96)  # Assuming 224x224 RGB frames
        self.__stack_size = 4
        self.__state_size = (self.__stack_size, *self.__image_shape)  # Assuming 96x96 RGB frames
        self.__frame_preprocessor = FramePreprocessor(resize_shape=self.__image_shape)
        self.__frame_stacker = FrameStacker(stack_length=self.__stack_size, image_shape=(1, *self.__image_shape))  # Stack 4 frames
        self.__frame_stacker_next_states = FrameStacker(stack_length=4, image_shape=(1, *self.__image_shape))  # Stack 4 frames
        self.__env = gym.make('LunarLander-v3', render_mode='rgb_array')  # Use v2, v3 doesn't exist yet
        self.__test_env = gym.make('LunarLander-v3', render_mode='human')
        self.__agent = LunarLanderDQNAgent(
            state_size=self.__state_size, 
            action_state_size=self.__env.action_space.n, 
            model_weights_path=model_weights_path,
            learning_rate=5e-5,
            memory_size=20_000,
            sync_target_every=6000,
            min_epsilon=0.05
        )  # Assuming this is the correct param
        self.__episode_losses = []  # List to store total losses for rolling average
        self.__episode_rewards = []  # List to store total rewards for rolling average
        self.__logger = MetricLogger(log_dir=log_dir, experiment_name=experiment_name)

    def __initialize_episode(self):
        """Initializes a new training episode by resetting the environment and frame stackers."""
        observation, info = self.__env.reset()
        self.__frame_stacker.clear_stack()
        self.__frame_stacker_next_states.clear_stack()
        return observation

    def __process_step(self):
        """Processes a single step: renders, preprocesses, chooses an action, and stores memory."""
        frame = self.__env.render()  # RGB array
        image = Image.fromarray(frame).convert('L')
        preprocessed_frame = self.__frame_preprocessor.preprocess(image)
        self.__frame_stacker.push(preprocessed_frame)

        state = self.__frame_stacker.get_stacked_frames()
        action = self.__agent.choose_action(state)  # Training mode (with exploration)
        _, reward, done, truncated, _ = self.__env.step(action)

        next_frame = self.__env.render()  # RGB array
        next_image = Image.fromarray(next_frame).convert('L')
        preprocessed_next_frame = self.__frame_preprocessor.preprocess(next_image)
        self.__frame_stacker_next_states.push(preprocessed_next_frame)

        next_state = self.__frame_stacker_next_states.get_stacked_frames()
        self.__agent.store_memory(state, action, reward, next_state, done)
        return reward, done, truncated

    def __learn_from_step(self):
        """Handles learning: updates the agent and returns loss."""
        total_loss = 0.0
        loss_count = 0

        loss = self.__agent.learn(return_loss=True)
        if loss is not None:
            total_loss += loss
            loss_count += 1
        return total_loss, loss_count

    def __calculate_metrics(self, total_loss, loss_count, total_reward, rolling_avg_episodes_count):
        """Calculates loss and reward metrics, including rolling averages."""
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

    def __update_progress_bar(self, t, total_reward, total_loss, avg_loss, rolling_avg_reward, rolling_avg_loss, rolling_avg_episodes_count):
        """Updates the tqdm progress bar with current episode metrics."""
        t.set_postfix({
            "Reward": f"{total_reward:.2f}",
            "Total Loss": f"{total_loss:.4f}",
            "Avg Loss": f"{avg_loss:.4f}",
            f"Rolling Avg Reward ({rolling_avg_episodes_count})": f"{rolling_avg_reward:.2f}",
            f"Rolling Avg Loss ({rolling_avg_episodes_count})": f"{rolling_avg_loss:.4f}"
        })

    def test_visually(self, max_steps=1000):
        """
        Runs the current agent in a visually rendered environment for one episode to inspect performance.

        :param max_steps: Maximum steps to run the test episode (default: 1000).
        """
        observation, info = self.__test_env.reset()
        self.__frame_stacker.clear_stack()  # Reset frame stack for testing
        done = False
        truncated = False
        step = 0
        total_reward = 0

        print("Starting visual test episode...")
        while not (done or truncated) and step < max_steps:
            frame = self.__env.render()  # Render in human mode (visible window)
            image = Image.fromarray(frame).convert('L')
            preprocessed_frame = self.__frame_preprocessor.preprocess(image)
            self.__frame_stacker.push(preprocessed_frame)

            if self.__frame_stacker.is_full():
                state = self.__frame_stacker.get_stacked_frames()
                action = self.__agent.choose_action(state)  # Assume no exploration in test
                observation, reward, done, truncated, info = self.__test_env.step(action)
                total_reward += reward

            step += 1

        print(f"Visual test episode finished. Total Reward: {total_reward}")


    def train(
        self,
        n_episodes: int,
        max_steps: int = 1000,
        display_interval: int = 50,
        rolling_avg_episodes_count: int = 50,
        weights_dir: str | Path = "projects/lunar_lander_dqn/rgb_mode/model_weights",
        weights_prefix: str = "lunar_lander_dqn",
    ) -> None:
        """
        Trains the Lunar Lander agent and visually tests it every `display_interval` episodes.

        Args:
            n_episodes: Number of training episodes.
            max_steps: Maximum steps per episode (default: 1000).
            display_interval: How often (in episodes) to visually test the model (default: 50).
            rolling_avg_episodes_count: Number of episodes for rolling averages (default: 50).
            weights_dir: Directory where model weights will be saved (created if missing).
            weights_prefix: Filename prefix for saved weight files.
        """
        weights_dir = Path(weights_dir)
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        total_steps = 0

        with trange(n_episodes, desc="Training", unit="episode") as t:
            for episode in t:
                observation = self.__initialize_episode()
                done = False
                truncated = False
                step = 0
                total_reward = 0.0
                total_loss = 0.0
                loss_count = 0

                for _ in range(self.__stack_size):
                    self.__frame_stacker.push(torch.zeros((1, *self.__image_shape)))
                    self.__frame_stacker_next_states.push(torch.zeros((1, *self.__image_shape)))

                while not (done or truncated) and step < max_steps:
                    reward, done, truncated= self.__process_step()
                    reward -= step * 0.1  # Penalize longer episodes

                    step_loss, step_loss_count = self.__learn_from_step()
                    total_loss += step_loss
                    loss_count += step_loss_count
                    total_reward += reward

                    step += 1
                    total_steps += 1
                    self.__agent.decay_epsilon(total_steps)

                # self.__agent.decay_epsilon()
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
    trainer.train(n_episodes=10_000,
                  display_interval=50,
                  rolling_avg_episodes_count=50)  # Test every 50 episodes
