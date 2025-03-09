import gymnasium
import torch
import rootutils
from tqdm import tqdm

rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from projects.lunar_lander_dqn.human_mode.agent import LunarLanderDQNAgent


class LunarLanderTrainer:
    
    def __init__(self, target_update_freq=100, render_freq=1000):
        self.__env = gymnasium.make('LunarLander-v3')  # Regular training environment
        self.__visual_env = None  # Separate environment for visualization
        self.__agent = LunarLanderDQNAgent(self.__env.observation_space.shape[0], self.__env.action_space.n)
        self.__target_update_freq = target_update_freq  # Frequency of target network update
        self.__render_freq = render_freq  # Frequency of rendering episodes

    def train(self, episodes=100_000, save_weights_freq = 1000):
        for episode in tqdm(range(episodes)):
        
            # Reset the environment and get the initial observation
            observation, info = self.__env.reset()
            done = False
            loss = None

            while not done:
                # Select action using epsilon-greedy policy
                observation_tensor = torch.from_numpy(observation).float().unsqueeze(0)
                action = self.__agent.choose_action(observation_tensor)

                # Take a step in the environment
                new_observation, reward, done, truncated, info = self.__env.step(action)

                # Store experience in the replay memory
                self.__agent.store_memory(
                    torch.from_numpy(observation).float(), 
                    action, 
                    reward, 
                    torch.from_numpy(new_observation).float(), 
                    done
                )

                loss = self.__agent.learn(return_loss=True)

                # Update observation
                observation = new_observation

                # Stop episode if it's done or truncated
                if done or truncated:
                    break

            if episode % self.__target_update_freq == 0:
                print(loss)
                
            # Every `render_freq` episodes, visualize progress
            if episode % save_weights_freq == 0 and episode > 0:
                self.__agent.save_model(f"projects/lunar_lander_dqn/human_mode/model_weights/lunar_lander_dqn_{episode}.pt")
                
            self.__agent.decay_epsilon()
                
            # Every `render_freq` episodes, visualize progress
            if episode % self.__render_freq == 0 and episode > 0:
                self.visualize_agent(episode)
                
    def visualize_agent(self, episode):
        """
        Runs a test episode without exploration (epsilon = 0) to visualize agent performance.
        Uses a separate environment to avoid interfering with training.
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
        total_reward = 0

        for i in tqdm(range(max_steps)):
            if done:
                break
            
            action = self.__agent.choose_action(torch.from_numpy(observation).float())  # Greedy action
            new_observation, reward, done, truncated, info = self.__visual_env.step(action)
            total_reward += reward

            self.__visual_env.render()  # Render the environment

            observation = new_observation
            
        # Restore original epsilon after visualization
        self.__agent.epsilon = original_epsilon

        # Properly close the visualization environment
        self.__visual_env.close()
        self.__visual_env = None  # Reset the environment so it's recreated when needed
        print(total_reward)


if __name__ == '__main__':
    trainer = LunarLanderTrainer(target_update_freq=100, render_freq=1000)
    trainer.train(episodes=10_000)
