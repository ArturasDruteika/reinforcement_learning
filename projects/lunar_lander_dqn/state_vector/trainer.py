import gymnasium
import torch
import rootutils
from tqdm import trange  # Changed from tqdm to trange

rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from projects.lunar_lander_dqn.state_vector.agent import LunarLanderDQNAgent


class LunarLanderTrainer:
    
    def __init__(self, target_update_freq=100, render_freq=1000):
        """Initialize the LunarLanderTrainer.

        Args:
            target_update_freq (int): Frequency of target network updates.
            render_freq (int): Frequency of rendering episodes for visualization.
        """
        self.__env = gymnasium.make('LunarLander-v3')  # Regular training environment
        self.__visual_env = None  # Separate environment for visualization
        self.__agent = LunarLanderDQNAgent(self.__env.observation_space.shape[0], self.__env.action_space.n)
        self.__target_update_freq = target_update_freq  # Frequency of target network update
        self.__render_freq = render_freq  # Frequency of rendering episodes
        self.__episode_losses = []  # List to store total losses for rolling average
        self.__episode_rewards = []  # List to store total rewards for rolling average

    def train(
        self, 
        episodes=100_000, 
        max_steps=1000, 
        save_weights_freq=1000, 
        rolling_avg_episodes_count=50
    ):
        """Train the Lunar Lander agent with a progress bar.

        Args:
            episodes (int): Number of training episodes.
            save_weights_freq (int): Frequency of saving model weights.
            rolling_avg_episodes_count (int): Number of episodes for rolling averages.
        """
        with trange(episodes, desc='Training', unit='episode') as t:
            for episode in t:
                # Reset the environment and get the initial observation
                observation, info = self.__env.reset()
                done = False
                step = 0
                truncated = False
                total_loss_per_epsiode = 0.0  # Track total loss for the episode
                loss_count = 0    # Count number of learning steps with loss
                total_reward_per_epsiode = 0.0  # Track total reward for the episode

                while not (done or truncated):
                    # Select action using epsilon-greedy policy
                    observation_tensor = torch.from_numpy(observation).float()
                    action = self.__agent.choose_action(observation_tensor)

                    # Take a step in the environment
                    new_observation, reward, done, truncated, info = self.__env.step(action)
                    total_reward_per_epsiode += reward  # Accumulate reward

                    # Store experience in the replay memory
                    self.__agent.store_memory(
                        torch.from_numpy(observation).float(), 
                        action, 
                        reward, 
                        torch.from_numpy(new_observation).float(), 
                        done
                    )

                    # Learn and accumulate loss
                    loss = self.__agent.learn(return_loss=True)
                    if loss is not None:  # Only count if learning occurred
                        total_loss_per_epsiode += loss.item()  # Use .item() to get scalar value
                        loss_count += 1

                    # Update observation
                    observation = new_observation
                    
                    step += 1
                    
                self.__agent.decay_epsilon()

                # Calculate metrics
                if loss_count > 0:
                    avg_loss = total_loss_per_epsiode / loss_count
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

                self.__episode_rewards.append(total_reward_per_epsiode)
                if len(self.__episode_rewards) > rolling_avg_episodes_count:
                    self.__episode_rewards.pop(0)
                rolling_avg_reward = sum(self.__episode_rewards) / len(self.__episode_rewards)

                # Update progress bar
                t.set_postfix({
                    "Reward": f"{total_reward_per_epsiode:.2f}",
                    "Total Loss": f"{total_loss_per_epsiode:.4f}",
                    "Avg Loss": f"{avg_loss:.4f}",
                    f"Rolling Avg Reward ({rolling_avg_episodes_count})": f"{rolling_avg_reward:.2f}",
                    f"Rolling Avg Loss ({rolling_avg_episodes_count})": f"{rolling_avg_loss:.4f}"
                })

                # Periodic updates and actions
                if episode % self.__target_update_freq == 0 and loss is not None:
                    print(f"Episode {episode} Loss: {loss.item():.4f}")
                
                # Save model weights
                if episode % save_weights_freq == 0 and episode > 0:
                    self.__agent.save_model(f'projects/lunar_lander_dqn/state_vector/model_weights/lunar_lander_dqn_{episode}.pt')
                
                # Visualize progress
                if episode % self.__render_freq == 0 and self.__agent.replay_memory.is_full:
                    self.visualize_agent(episode)
        
        self.__env.close()

    def visualize_agent(self, episode):
        """
        Runs a test episode without exploration (epsilon = 0) to visualize agent performance.
        Uses a separate environment to avoid interfering with training.

        Args:
            episode (int): Current episode number for logging.
        """
        print(f"\nVisualizing performance at episode {episode}")

        # Create a new environment just for rendering
        if self.__visual_env is None:
            self.__visual_env = gymnasium.make('LunarLander-v3', render_mode="human")  # Human mode for visualization

        observation, info = self.__visual_env.reset()
        done = False

        # Disable exploration for visualization
        original_epsilon = self.__agent.epsilon
        self.__agent.epsilon = 0
        
        max_steps = 1_000
        total_reward_per_epsiode = 0

        for i in trange(max_steps, desc="Visualization", unit="step"):
            if done:
                break
            
            action = self.__agent.choose_action(torch.from_numpy(observation).float())  # Greedy action
            new_observation, reward, done, truncated, info = self.__visual_env.step(action)
            total_reward_per_epsiode += reward

            self.__visual_env.render()  # Render the environment

            observation = new_observation
            
        # Restore original epsilon after visualization
        self.__agent.epsilon = original_epsilon

        # Properly close the visualization environment
        self.__visual_env.close()
        self.__visual_env = None  # Reset the environment so it's recreated when needed
        print(f"Visualization Total Reward: {total_reward_per_epsiode:.2f}")


if __name__ == '__main__':
    trainer = LunarLanderTrainer(target_update_freq=100, render_freq=100)
    trainer.train(episodes=10_000)