import gymnasium
import torch
import rootutils
from tqdm import tqdm
import time  # To slow down rendering for better visualization

rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from projects.lunar_lander_dqn.human_mode.agent import LunarLanderDQNAgent


class LunarLanderTrainer:
    
    def __init__(self, target_update_freq=100, render_freq=1000):
        self.__env = gymnasium.make('LunarLander-v3')  # Regular training environment
        self.__visual_env = None  # Separate environment for visualization
        self.__agent = LunarLanderDQNAgent(self.__env.observation_space.shape[0], self.__env.action_space.n)
        self.__target_update_freq = target_update_freq  # Frequency of target network update
        self.__render_freq = render_freq  # Frequency of rendering episodes

    def train(self, episodes=100_000):
        for episode in tqdm(range(episodes)):
        
            # Reset the environment and get the initial observation
            observation, info = self.__env.reset()
            done = False

            while not done:
                # Select action using epsilon-greedy policy
                action = self.__agent.choose_action(torch.from_numpy(observation).float())

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

                # Train the agent
                loss = self.__agent.training_step()

                # Update observation
                observation = new_observation

                # Stop episode if it's done or truncated
                if done or truncated:
                    break

            # Every `target_update_freq` episodes, update the target network
            if episode % self.__target_update_freq == 0:
                print(f"\nUpdating target network at episode {episode}")
                self.__agent.update_target_model()
                print(loss)

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

        for i in tqdm(range(max_steps)):
            if done:
                break
            
            action = self.__agent.choose_action(torch.from_numpy(observation).float())  # Greedy action
            new_observation, reward, done, truncated, info = self.__visual_env.step(action)

            self.__visual_env.render()  # Render the environment
            time.sleep(0.01)  # Slow down rendering for better visualization

            observation = new_observation
            
        # Restore original epsilon after visualization
        self.__agent.epsilon = original_epsilon

        # Properly close the visualization environment
        self.__visual_env.close()
        self.__visual_env = None  # Reset the environment so it's recreated when needed


if __name__ == '__main__':
    trainer = LunarLanderTrainer(target_update_freq=100, render_freq=1000)
    trainer.train(episodes=100_000)
