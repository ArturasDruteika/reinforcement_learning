from typing import Deque, Tuple
from collections import deque

import torch


DEFAULT_STACK_LENGTH = 4
DEFAULT_IMAGE_SHAPE = (1, 224, 224)  # Default grayscale shape: (channels, height, width)


class FrameStacker:
    """
    Stacks the last N grayscale frames along the channel dimension into a single tensor.
    Maintains a rolling buffer where the oldest frame is removed when the stack is full.
    No padding is applied; output shape reflects the current number of frames.
    """

    def __init__(self, stack_length: int = DEFAULT_STACK_LENGTH, image_shape: Tuple[int, int, int] = DEFAULT_IMAGE_SHAPE):
        """
        Initializes the frame stacker for grayscale frames.
        
        :param stack_length: Number of frames to stack. Default is 4.
        :param image_shape: Tuple of (channels, height, width) for each frame. Default is (1, 224, 224).
        """
        if not (len(image_shape) == 3 and image_shape[0] == 1):  # Ensure grayscale (1 channel)
            raise ValueError("image_shape must be a tuple of (1, height, width) for grayscale frames")
        self.__stack_length: int = stack_length
        self.__image_shape: Tuple[int, int, int] = image_shape
        self.__frames: Deque[torch.Tensor] = deque(maxlen=stack_length)

    @property
    def stack_length(self) -> int:
        """
        Returns the current stack length.
        
        :return: The number of frames to be stacked.
        """
        return self.__stack_length

    @property
    def image_shape(self) -> Tuple[int, int, int]:
        """
        Returns the current image shape.
        
        :return: The shape (channels, height, width) of each frame.
        """
        return self.__image_shape

    def set_stack_length(self, new_stack_length: int) -> None:
        """
        Dynamically updates the number of stacked frames.

        :param new_stack_length: The new number of frames to keep in the stack.
        """
        self.__stack_length = new_stack_length
        self.__frames = deque(list(self.__frames)[-new_stack_length:], maxlen=new_stack_length)

    def set_image_shape(self, new_image_shape: Tuple[int, int, int]) -> None:
        """
        Dynamically updates the image shape, validating it remains grayscale.

        :param new_image_shape: The new shape (channels, height, width) for each frame.
        """
        if not (len(new_image_shape) == 3 and new_image_shape[0] == 1):
            raise ValueError("new_image_shape must be a tuple of (1, height, width) for grayscale frames")
        self.__image_shape = new_image_shape
        # Clear stack to avoid shape mismatch with existing frames
        self.clear_stack()

    def clear_stack(self) -> None:
        """
        Clears the current stack.
        """
        self.__frames.clear()

    def push(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Adds a new grayscale frame to the stack while ensuring that at most stack_length frames are kept.
        When the stack is full, the oldest frame is removed, and the new frame is appended.
        Validates the frame shape matches the expected image_shape.

        :param frame: The new grayscale frame.
        :raises ValueError: If the frame shape does not match image_shape.
        """
        if frame.shape != self.__image_shape:
            raise ValueError(f"Frame shape {frame.shape} does not match expected shape {self.__image_shape}")
        self.__frames.append(frame)  # deque automatically removes oldest frame if full

    def get_stacked_frames(self) -> torch.Tensor:
        """
        Returns the stacked frames as a single tensor with frames stacked along the channel dimension.

        :return: Tensor of shape (current_stack_size, height, width) based on image_shape.
        """
        if not self.__frames:
            return torch.zeros((0,) + self.__image_shape[1:])  # Shape: (0, height, width)
        return torch.cat(list(self.__frames), dim=0)  # Shape: (current_size, height, width)

    def is_full(self) -> bool:
        """
        Checks if the stack is full.
        
        :return: True if the stack is full, False otherwise.
        """
        return len(self.__frames) == self.__stack_length


if __name__ == '__main__':
    def generate_fake_frame(height: int = 224, width: int = 224) -> torch.Tensor:
        """
        Generates a random grayscale frame tensor simulating an image.
        
        :param height: Height of the frame.
        :param width: Width of the frame.
        :return: A tensor with shape (1, height, width) with random pixel values.
        """
        return torch.rand((1, height, width))  # Grayscale: single channel

    # Test with default settings (224 x 224)
    stacker = FrameStacker(stack_length=4)
    print("Pushing frame 1 (default 224x224)")
    stacker.push(generate_fake_frame())
    print("Current stacked shape:", stacker.get_stacked_frames().shape)  # (1, 224, 224)

    print("Pushing frame 2")
    stacker.push(generate_fake_frame())
    print("Current stacked shape:", stacker.get_stacked_frames().shape)  # (2, 224, 224)

    print("Pushing frame 3")
    stacker.push(generate_fake_frame())
    print("Current stacked shape:", stacker.get_stacked_frames().shape)  # (3, 224, 224)

    print("Pushing frame 4")
    stacker.push(generate_fake_frame())
    print("Current stacked shape:", stacker.get_stacked_frames().shape)  # (4, 224, 224)

    print("Pushing frame 5 (should remove oldest and keep 4)")
    stacker.push(generate_fake_frame())
    print("Current stacked shape:", stacker.get_stacked_frames().shape)  # (4, 224, 224)

    # Test with custom image shape (84 x 84)
    stacker_custom = FrameStacker(stack_length=4, image_shape=(1, 84, 84))
    print("\nPushing frame 1 (custom 84x84)")
    stacker_custom.push(generate_fake_frame(84, 84))
    print("Current stacked shape:", stacker_custom.get_stacked_frames().shape)  # (1, 84, 84)

    print("Pushing frame 2")
    stacker_custom.push(generate_fake_frame(84, 84))
    print("Current stacked shape:", stacker_custom.get_stacked_frames().shape)  # (2, 84, 84)

    print("Pushing frame 3")
    stacker_custom.push(generate_fake_frame(84, 84))
    print("Current stacked shape:", stacker_custom.get_stacked_frames().shape)  # (3, 84, 84)

    print("Pushing frame 4")
    stacker_custom.push(generate_fake_frame(84, 84))
    print("Current stacked shape:", stacker_custom.get_stacked_frames().shape)  # (4, 84, 84)

    print("Pushing frame 5 (should remove oldest and keep 4)")
    stacker_custom.push(generate_fake_frame(84, 84))
    print("Current stacked shape:", stacker_custom.get_stacked_frames().shape)  # (4, 84, 84)