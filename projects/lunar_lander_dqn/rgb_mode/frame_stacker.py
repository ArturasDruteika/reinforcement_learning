from typing import Deque
from collections import deque

import torch


class FrameStacker:
    """
    Stacks the last N frames and maintains a rolling buffer where only the last two frames are retained when full.
    """

    def __init__(self, stack_length: int = 3):
        """
        Initializes the frame stacker.
        
        :param stack_length: Number of frames to stack. Default is 3.
        """
        self.__stack_length: int = stack_length
        self.__frames: Deque[torch.Tensor] = deque(maxlen=stack_length)

    @property
    def stack_length(self) -> int:
        """
        Returns the current stack length.
        
        :return: The number of frames to be stacked.
        """
        return self.__stack_length

    def set_stack_length(self, new_stack_length: int) -> None:
        """
        Dynamically updates the number of stacked frames.

        :param new_stack_length: The new number of frames to keep in the stack.
        """
        self.__stack_length = new_stack_length
        self.__frames = deque(list(self.__frames)[-new_stack_length:], maxlen=new_stack_length)

    def clear_stack(self) -> None:
        """
        Clears the current stack
        """
        self.__frames.clear()

    def push(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Adds a new frame to the stack while ensuring that at most stack_length frames are kept.
        If the stack exceeds the limit, the last two frames are retained, and the new frame is appended.

        :param frame: The new frame (Tensor shape: (3, 224, 224)).
        :return: Stacked frames as a Tensor of shape (min(current size, stack_length), 3, 224, 224).
        """
        if len(self.__frames) >= self.__stack_length:  # If at capacity, keep only last 2
            self.__frames = deque(list(self.__frames)[-2:], maxlen=self.__stack_length)

        self.__frames.append(frame)
        return self.get_stacked_frames()

    def get_stacked_frames(self) -> torch.Tensor:
        """
        Returns the stacked frames as a batch tensor.

        :return: Tensor of shape (current_stack_size, 3, 224, 224).
        """
        return torch.stack(list(self.__frames), dim=0)


if __name__ == '__main__':
    def generate_fake_frame() -> torch.Tensor:
        """
        Generates a random frame tensor simulating an image.
        
        :return: A tensor with shape (3, 224, 224) with random pixel values.
        """
        return torch.rand((3, 224, 224))

    # Initialize FrameStacker with stack_length = 3
    stacker = FrameStacker(stack_length=3)

    print("Pushing frame 1")
    stacker.push(generate_fake_frame())
    print("Current stacked shape:", stacker.get_stacked_frames().shape)  # (1, 3, 224, 224)

    print("Pushing frame 2")
    stacker.push(generate_fake_frame())
    print("Current stacked shape:", stacker.get_stacked_frames().shape)  # (2, 3, 224, 224)

    print("Pushing frame 3")
    stacker.push(generate_fake_frame())
    print("Current stacked shape:", stacker.get_stacked_frames().shape)  # (3, 3, 224, 224)

    print("Pushing frame 4 (should keep last 2 and add new one)")
    stacker.push(generate_fake_frame())
    print("Current stacked shape:", stacker.get_stacked_frames().shape)  # (3, 3, 224, 224)
    
    stacker.clear_stack()

    print("Pushing frame 5 into a new stack")
    stacker.push(generate_fake_frame())
    print("Current stacked shape:", stacker.get_stacked_frames().shape)  # (3, 3, 224, 224)
