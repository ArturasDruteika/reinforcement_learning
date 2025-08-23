import numpy as np
import gymnasium
import torch
import rootutils
from tqdm import trange

rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from projects.lunar_lander_double_dqn.state_vector.agent import LunarLanderDoubleDQNAgent
from metrics_loggers.metric_logger import MetricLogger


class LunarLanderTrainer:
    
    def __init__(
        self, 
        target_update_freq=100, 
        render_freq=1000, 
        model_weights_path=None,
        log_dir: str = "runs",
        experiment_name: str = "lunar_lander_double_dqn"
        ):
        """Initialize the LunarLanderTrainer."""
        self.__env = gymnasium.make('LunarLander-v3')
        self.__visual_env = None
        self.__agent = LunarLanderDoubleDQNAgent(
            state_size=self.__env.observation_space.shape[0], 
            action_space_size=self.__env.action_space.n, 
            model_weights_path=model_weights_path,
            batch_size=64,
            memory_size=100_000,
            learning_rate=1e-4,
            min_epsilon=0.05,
            warmup_steps=10_000,
            epsilon_decay_steps=200_000
        )
        self.__target_update_freq = target_update_freq
        self.__render_freq = render_freq
        self.__episode_losses = []
        self.__episode_rewards = []
        self.__logger: MetricLogger = MetricLogger(log_dir=log_dir, experiment_name=experiment_name)
        self.__learning_step = 0

    def __run_episode(self, max_steps=1000):
        """Run a single training episode and return metrics."""
        state, info = self.__env.reset()
        done = False
        truncated = False
        step = 0
        total_loss = 0.0
        loss_count = 0
        total_reward = 0.0

        while not (done or truncated) and step < max_steps:
            state_tensor = torch.from_numpy(state).float()
            action = self.__agent.choose_action(state_tensor)
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

    def __update_metrics(self, total_reward, total_loss, loss_count, rolling_avg_episodes_count):
        """Update rolling averages and return metrics for display."""
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

    def __periodic_actions(self, episode, total_loss, loss_count, save_weights_freq):
        """Perform periodic updates and actions like saving weights or visualization."""
        if loss_count > 0:
            avg_loss = total_loss / loss_count
            if episode % self.__target_update_freq == 0:
                print(f"Episode {episode} Loss: {avg_loss:.4f}")

        if episode % save_weights_freq == 0 and episode > 0:
            self.__agent.save_model(f'projects/lunar_lander_dqn/state_vector/model_weights/lunar_lander_dqn_{episode}.pt')

        if episode % self.__render_freq == 0 and self.__learning_step >= self.__agent.warmup_steps:
            self.visualize_agent(episode)

    def train(self, episodes=100_000, max_steps=1000, save_weights_freq=1000, rolling_avg_episodes_count=50):
        """Train the Lunar Lander agent with a progress bar."""
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

    def visualize_agent(self, episode):
        """Runs a test episode without exploration to visualize agent performance."""
        print(f"\nVisualizing performance at episode {episode}")
        if self.__visual_env is None:
            self.__visual_env = gymnasium.make('LunarLander-v3', render_mode="human")

        state, info = self.__visual_env.reset()
        done = False
        original_epsilon = self.__agent.epsilon
        self.__agent.epsilon = 0
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

        self.__agent.epsilon = original_epsilon
        self.__visual_env.close()
        self.__visual_env = None
        print(f"Visualization Total Reward: {total_reward:.2f}")


if __name__ == '__main__':
    trainer = LunarLanderTrainer(target_update_freq=80, render_freq=500)
    trainer.train(episodes=1_000_000)