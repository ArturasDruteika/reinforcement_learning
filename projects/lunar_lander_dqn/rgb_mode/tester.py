from typing import Tuple

import gymnasium as gym
from PIL import Image
import torch
from torch import Tensor
import rootutils

rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from projects.lunar_lander_dqn.rgb_mode.agent import LunarLanderDQNAgent
from projects.lunar_lander_dqn.rgb_mode.frame_processor import FramePreprocessor
from projects.lunar_lander_dqn.rgb_mode.frame_stacker import FrameStacker


class LunarLanderDQNTester:
    """
    Tester for evaluating a pre-trained DQN agent on the LunarLander-v3 environment using RGB frames.

    This class runs an evaluation episode using:
      - An image-only environment for state input (rgb_array mode)
      - A visible environment for rendering (human mode)
      - Preprocessing and stacking of frames before passing them to the agent
    """

    def __init__(self, model_weights_path: str) -> None:
        """
        Initializes the LunarLanderDQNTester.

        Args:
            model_weights_path: Path to the saved DQN model weights (.pt file).
        """
        self.__image_shape: Tuple[int, int] = (96, 96)
        self.__stack_size: int = 4
        self.__state_size: Tuple[int, int, int] = (self.__stack_size, *self.__image_shape)

        self.__frame_preprocessor = FramePreprocessor(resize_shape=self.__image_shape)
        self.__frame_stacker = FrameStacker(stack_length=self.__stack_size,
                                            image_shape=(1, *self.__image_shape))

        # Environment for producing RGB image frames for state input
        self.__image_env = gym.make('LunarLander-v3', render_mode='rgb_array')
        # Environment for live rendering in a visible window
        self.__test_env = gym.make('LunarLander-v3', render_mode='human')

        self.__agent = LunarLanderDQNAgent(
            state_size=self.__state_size,
            action_state_size=self.__image_env.action_space.n,
            model_weights_path=model_weights_path
        )
        # Disable exploration during testing
        self.__agent.epsilon = 0.0

    def run(self, max_steps: int = 1000) -> None:
        """
        Runs a visual test episode using the pre-trained agent.

        Args:
            max_steps: Maximum number of steps before terminating the test.
        """
        observation, info = self.__test_env.reset()
        self.__image_env.reset()  # Synchronize both environments
        self.__frame_stacker.clear_stack()

        step = 0
        done = False
        truncated = False
        total_reward = 0.0

        # Prime the frame stack with black frames
        for _ in range(self.__stack_size):
            self.__frame_stacker.push(torch.zeros((1, *self.__image_shape)))

        print("Starting test episode...")
        while not (done or truncated) and step < max_steps:
            # Retrieve the current RGB frame from the image environment
            frame = self.__image_env.render()
            if frame is None:
                raise RuntimeError("Render returned None. Ensure render_mode='rgb_array' is set correctly.")

            # Convert to grayscale and preprocess
            image = Image.fromarray(frame).convert('L')
            preprocessed: Tensor = self.__frame_preprocessor.preprocess(image)
            self.__frame_stacker.push(preprocessed)

            if self.__frame_stacker.is_full():
                state = self.__frame_stacker.get_stacked_frames()
                action = self.__agent.choose_action(state)

                observation, reward, done, truncated, info = self.__test_env.step(action)
                self.__image_env.step(action)  # Keep environments in sync
                total_reward += reward

            step += 1

        print(f"Test episode finished. Total reward: {total_reward:.2f}")
        self.__test_env.close()
        self.__image_env.close()


if __name__ == '__main__':
    tester = LunarLanderDQNTester(model_weights_path='projects/lunar_lander_dqn/rgb_mode/model_weights/lunar_lander_dqn_300.pt')
    tester.run()
