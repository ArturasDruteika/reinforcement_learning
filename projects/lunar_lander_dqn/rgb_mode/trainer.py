import gymnasium as gym
from PIL import Image
import rootutils

rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from projects.lunar_lander_dqn.rgb_mode.agent import LunarLanderDQNAgent
from projects.lunar_lander_dqn.rgb_mode.frame_processor import FramePreprocessor
from projects.lunar_lander_dqn.rgb_mode.frame_stacker import FrameStacker


class LunarLanderTrainer:
    def __init__(self):
        self.__frame_preprocessor = FramePreprocessor()
        self.__frame_stacker = FrameStacker(stack_length=4)  # Stack 4 frames
        self.__frame_stacker_next_states = FrameStacker(stack_length=4)  # Stack 4 frames
        self.__agent = LunarLanderDQNAgent(action_state_size=4)  # Assuming this is the correct param
        self.__env = gym.make("LunarLander-v3", render_mode="rgb_array")  # Use v2, v3 doesn't exist yet
        # Separate env for visual testing
        self.__test_env = gym.make("LunarLander-v3", render_mode="human")
    
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
            # Optional: Add a small delay to make it watchable (e.g., time.sleep(0.02))
            # import time; time.sleep(0.02)
        
        print(f"Visual test episode finished. Total Reward: {total_reward}")
    
    def train(self, n_episodes, max_steps=1000, display_interval=50):
        """
        Trains the Lunar Lander agent and visually tests it every display_interval episodes.
        
        :param n_episodes: Number of training episodes.
        :param max_steps: Maximum steps per episode (default: 1000).
        :param display_interval: How often (in episodes) to visually test the model (default: 50).
        """
        for episode in range(n_episodes):
            observation, info = self.__env.reset()
            self.__frame_stacker.clear_stack()
            self.__frame_stacker_next_states.clear_stack()
            done = False
            truncated = False
            step = 0
            total_reward = 0
            loss = None
            
            while not (done or truncated) and step < max_steps:
                frame = self.__env.render()  # RGB array
                image = Image.fromarray(frame).convert('L')
                preprocessed_frame = self.__frame_preprocessor.preprocess(image)
                self.__frame_stacker.push(preprocessed_frame)
                
                if self.__frame_stacker.is_full():
                    state = self.__frame_stacker.get_stacked_frames()
                    action = self.__agent.choose_action(state)  # Training mode (with exploration)
                    new_observation, reward, done, truncated, info = self.__env.step(action)
                    
                    next_frame = self.__env.render()  # RGB array
                    next_image = Image.fromarray(next_frame).convert('L')
                    preprocessed_next_frame = self.__frame_preprocessor.preprocess(next_image)
                    self.__frame_stacker_next_states.push(preprocessed_next_frame)
                    
                    if self.__frame_stacker_next_states.is_full():
                        next_state = self.__frame_stacker_next_states.get_stacked_frames()
                        self.__agent.store_memory(state, action, reward, next_state, done)
                        loss = self.__agent.learn(return_loss=True)
                        total_reward += reward
                        observation = new_observation
                        if loss is not None:
                            print(f"Episode {episode}, Step {step}, Loss: {loss:.4f}")
                step += 1
            
            print(f"Episode {episode} finished. Total Reward: {total_reward}")
            
            # Visual test every display_interval episodes (including after the last episode)
            if (episode + 1) % display_interval == 0 or episode == n_episodes - 1:
                self.test_visually(max_steps=max_steps)
        
        self.__env.close()
        self.__test_env.close()

# Run the trainer
if __name__ == "__main__":
    trainer = LunarLanderTrainer()
    trainer.train(n_episodes=100_000, display_interval=10)  # Test every 50 episodes