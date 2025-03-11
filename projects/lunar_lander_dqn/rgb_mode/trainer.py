import time

import gymnasium as gym
from PIL import Image
import rootutils

rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from projects.lunar_lander_dqn.agent import LunarLanderDQNAgent
from projects.lunar_lander_dqn.frame_processor import FramePreprocessor
from projects.lunar_lander_dqn.frame_stacker import FrameStacker


class LunarLanderTrainer:
    
    def __init__(self):
        self.__frame_preprocessor = FramePreprocessor()
        self.__frame_stacker = FrameStacker()
        self.__agent = LunarLanderDQNAgent()
        self.__env = gym.make("LunarLander-v3", render_mode="rgb_array")
        
    def train(self):
        # Reset the environment and get the initial observation
        observation, info = env.reset()
        
        done = False
        while not done:
            # Choose a random action from the action space
            action = env.action_space.sample()
            
            # Take a step in the environment
            observation, reward, done, truncated, info = env.step(action)

            # Capture the current frame as a NumPy array
            frame = env.render()
            
            image = Image.fromarray(frame)

            self.__frame_stacker.push(self.__frame_preprocessor.preprocess(image))
            if self.__frame_stacker.is_full():
                agent.store_memory(self.__frame_stacker.get_stacked_frames(), action, reward, done)
            
            # Exit if the episode is finished or truncated
            if done or truncated:
                break


if __name__ == "__main__":
    frame_preprocessor = FramePreprocessor()
    frame_stacker = FrameStacker()
    agent = LunarLanderDQNAgent()

    # Create the environment
    env = gym.make("LunarLander-v3", render_mode="rgb_array")

    # Reset the environment to get the initial observation
    obs, info = env.reset()
    print(obs)

    # Render a frame and get it as an array
    frame = env.render()  # This returns an RGB array

    # Convert the frame to a PIL Image
    image = Image.fromarray(frame)
    preprocessed_frame = frame_preprocessor.preprocess(image)
    
    print(preprocessed_frame.shape)
    frame_stacker.push(preprocessed_frame)
    frame_stacker.push(preprocessed_frame)
    frame_stacker.push(preprocessed_frame)
    print(frame_stacker.get_stacked_frames().shape)

    # Show the image
    # image.show()

    # Close the environment
    # env.close()