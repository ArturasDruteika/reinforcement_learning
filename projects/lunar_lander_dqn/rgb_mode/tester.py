import gymnasium as gym
from PIL import Image
import torch
import rootutils

rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from projects.lunar_lander_dqn.rgb_mode.agent import LunarLanderDQNAgent
from projects.lunar_lander_dqn.rgb_mode.frame_processor import FramePreprocessor
from projects.lunar_lander_dqn.rgb_mode.frame_stacker import FrameStacker


class LunarLanderDQNTester:
    def __init__(self, model_weights_path: str):
        self.__image_shape = (96, 96)
        self.__stack_size = 4
        self.__state_size = (self.__stack_size, *self.__image_shape)

        self.__frame_preprocessor = FramePreprocessor(resize_shape=self.__image_shape)
        self.__frame_stacker = FrameStacker(stack_length=self.__stack_size, image_shape=(1, *self.__image_shape))

        # Environment for getting image frames for preprocessing
        self.__image_env = gym.make('LunarLander-v3', render_mode='rgb_array')
        # Environment for rendering visible window
        self.__test_env = gym.make('LunarLander-v3', render_mode='human')

        self.__agent = LunarLanderDQNAgent(
            state_size=self.__state_size,
            action_state_size=self.__image_env.action_space.n,
            model_weights_path=model_weights_path
        )
        self.__agent.epsilon = 0.0  # Disable exploration manually

    def run(self, max_steps: int = 1000):
        """Runs a visual test episode with a pre-trained model."""
        observation, info = self.__test_env.reset()
        self.__image_env.reset()  # Make sure sync starts at same time
        self.__frame_stacker.clear_stack()

        step = 0
        done = False
        truncated = False
        total_reward = 0.0

        # Prime the stack with black frames
        for _ in range(self.__stack_size):
            self.__frame_stacker.push(torch.zeros((1, *self.__image_shape)))

        print("Starting test episode...")
        while not (done or truncated) and step < max_steps:
            # Get RGB frame from the image-rendering env
            frame = self.__image_env.render()
            if frame is None:
                raise RuntimeError("Render returned None. Is render_mode='rgb_array' set properly?")

            # Preprocess the grayscale frame
            image = Image.fromarray(frame).convert('L')
            preprocessed = self.__frame_preprocessor.preprocess(image)
            self.__frame_stacker.push(preprocessed)

            if self.__frame_stacker.is_full():
                state = self.__frame_stacker.get_stacked_frames()
                action = self.__agent.choose_action(state)

                observation, reward, done, truncated, info = self.__test_env.step(action)
                self.__image_env.step(action)  # Keep image env in sync
                total_reward += reward

            step += 1

        print(f"Test episode finished. Total reward: {total_reward}")
        self.__test_env.close()
        self.__image_env.close()


# Example usage
if __name__ == '__main__':
    tester = LunarLanderDQNTester(model_weights_path='projects/lunar_lander_dqn/rgb_mode/model_weights/lunar_lander_dqn_300.pt')
    tester.run()
