import torch
from collections import deque
import numpy as np


class FrameStacker:
    """Stacks the last N frames and returns them as a batch of stacked images."""

    def __init__(self, stack_length=3):
        """
        Initializes the frame stacker.
        :param stack_length: Number of frames to stack.
        """
        self.__stack_length = stack_length
        self.__frames = deque(maxlen=stack_length)
        
    @property
    def stack_length(self):
        """Returns the number of stacked frames."""
        return self.__stack_length

    def set_stack_length(self, new_stack_length):
        """
        Dynamically update the number of stacked frames.
        :param new_stack_length: New length of the stack.
        """
        self.__stack_length = new_stack_length
        self.__frames = deque(list(self.__frames)[-new_stack_length:], maxlen=new_stack_length)

    def reset(self, initial_frame):
        """
        Resets the frame stack with the first frame repeated.
        :param initial_frame: The first frame (Tensor shape: (3, 224, 224)).
        """
        self.__frames.clear()
        for _ in range(self.__stack_length):
            self.__frames.append(initial_frame)

    def push(self, frame):
        """
        Adds a new frame to the stack and returns the stacked frames.
        :param frame: The new frame (Tensor shape: (3, 224, 224)).
        :return: Stacked frames as a Tensor of shape (stack_length, 3, 224, 224).
        """
        self.__frames.append(frame)
        return self.get_stacked_frames()

    def get_stacked_frames(self):
        """
        Returns the stacked frames as a batch tensor.
        :return: Tensor of shape (stack_length, 3, 224, 224).
        """
        return torch.stack(list(self.__frames), dim=0)
    
    
def preprocess_frame(frame: np.ndarray):
    """Manually converts NumPy array to a normalized PyTorch tensor."""
    if not isinstance(frame, np.ndarray):
        raise TypeError("Frame must be a NumPy array.")

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Frame must have shape (H, W, 3) for RGB images.")

    # Convert (H, W, C) → (C, H, W)
    frame = np.transpose(frame, (2, 0, 1))

    # Convert to tensor
    frame = torch.tensor(frame, dtype=torch.float32)

    # Normalize using ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    frame = (frame / 255.0 - mean) / std  # Scale 0-255 → Normalize

    return frame


if __name__ == '__main__':
    # Initialize stacker with stack_length = 3
    stacker = FrameStacker(stack_length=3)
    
    # Generate a random RGB frame (100x100 NumPy array)
    random_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    processed_frame = preprocess_frame(random_image)

    # Reset stacker with the initial frame
    stacker.reset(processed_frame)

    # Push new frames and get stacked output
    for _ in range(5):  # More than stack_length to test overwriting
        new_frame = preprocess_frame(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        stacked_frames = stacker.push(new_frame)
        print(f"Stacked frames shape: {stacked_frames.shape}")  # Expected: (3, 3, 224, 224)

    # Dynamically change stack length
    stacker.set_stack_length(5)
    print(f"New stack length: {stacker.stack_length}")
