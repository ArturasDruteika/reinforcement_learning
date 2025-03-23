import gymnasium
import torch
import rootutils
from tqdm import trange

rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from projects.lunar_lander_dqn.state_vector.agent import LunarLanderDQNAgent


class LunarLanderTrainer:
    
    def __init__(self, target_update_freq=100, render_freq=1000, model_weights_path=None):
        """Initialize the LunarLanderTrainer."""
        self.__env = gymnasium.make('LunarLander-v3')
        self.__visual_env = None
        self.__agent = LunarLanderDQNAgent(
            state_size=self.__env.observation_space.shape[0], 
            action_space_size=self.__env.action_space.n, 
            model_weights_path=model_weights_path
        )
        self.__target_update_freq = target_update_freq
        self.__render_freq = render_freq
        self.__episode_losses = []
        self.__episode_rewards = []

    def __run_episode(self, max_steps=1000):
        """Run a single training episode and return metrics."""
        observation, info = self.__env.reset()
        done = False
        truncated = False
        step = 0
        total_loss = 0.0
        loss_count = 0
        total_reward = 0.0

        while not (done or truncated) and step < max_steps:
            observation_tensor = torch.from_numpy(observation).float()
            action = self.__agent.choose_action(observation_tensor)
            new_observation, reward, done, truncated, info = self.__env.step(action)
            reward -= step * 0.01  # Penalize for taking too long
            total_reward += reward

            self.__agent.store_memory(
                torch.from_numpy(observation).float(),
                action,
                reward,
                torch.from_numpy(new_observation).float(),
                done,
            )

            loss = self.__agent.learn(return_loss=True)
            if loss is not None:
                total_loss += loss.item()
                loss_count += 1

            observation = new_observation
            step += 1

        self.__agent.decay_epsilon()
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

        if episode % self.__render_freq == 0 and self.__agent.replay_memory.is_full:
            self.visualize_agent(episode)

    def train(self, episodes=100_000, max_steps=1000, save_weights_freq=1000, rolling_avg_episodes_count=50):
        """Train the Lunar Lander agent with a progress bar."""
        with trange(episodes, desc='Training', unit='episode') as t:
            for episode in t:
                # Run a single episode
                total_reward, total_loss, loss_count, steps = self.__run_episode(max_steps)

                # Update metrics
                avg_loss, rolling_avg_loss, rolling_avg_reward = self.__update_metrics(
                    total_reward, total_loss, loss_count, rolling_avg_episodes_count
                )

                # Update progress bar
                t.set_postfix({
                    "Reward": f"{total_reward:.2f}",
                    "Total Loss": f"{total_loss:.4f}",
                    "Avg Loss": f"{avg_loss:.4f}",
                    f"Rolling Avg Reward ({rolling_avg_episodes_count})": f"{rolling_avg_reward:.2f}",
                    f"Rolling Avg Loss ({rolling_avg_episodes_count})": f"{rolling_avg_loss:.4f}"
                })

                # Perform periodic actions
                self.__periodic_actions(episode, total_loss, loss_count, save_weights_freq)

        self.__env.close()

    def visualize_agent(self, episode):
        """Runs a test episode without exploration to visualize agent performance."""
        print(f"\nVisualizing performance at episode {episode}")
        if self.__visual_env is None:
            self.__visual_env = gymnasium.make('LunarLander-v3', render_mode="human")

        observation, info = self.__visual_env.reset()
        done = False
        original_epsilon = self.__agent.epsilon
        self.__agent.epsilon = 0
        max_steps = 1_000
        total_reward = 0

        for i in trange(max_steps, desc="Visualization", unit="step"):
            if done:
                break
            action = self.__agent.choose_action(torch.from_numpy(observation).float())
            new_observation, reward, done, truncated, info = self.__visual_env.step(action)
            total_reward += reward
            self.__visual_env.render()
            observation = new_observation

        self.__agent.epsilon = original_epsilon
        self.__visual_env.close()
        self.__visual_env = None
        print(f"Visualization Total Reward: {total_reward:.2f}")


if __name__ == '__main__':
    trainer = LunarLanderTrainer(target_update_freq=100, render_freq=100)
    trainer.train(episodes=10_000)