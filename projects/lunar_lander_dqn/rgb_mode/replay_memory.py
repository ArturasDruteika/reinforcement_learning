import random
import numpy as np
import torch
from typing import Any, Tuple, Optional


class ReplayMemory:
    """
    A replay memory buffer for reinforcement learning, storing experiences in a NumPy array.

    This class implements a circular buffer where new experiences replace the oldest ones
    when the memory reaches its maximum capacity.

    Attributes:
        capacity (int): The maximum number of experiences that can be stored.
        shuffle (bool): Whether to shuffle the experiences during sampling.
    """

    def __init__(self, capacity: int, shuffle: bool) -> None:
        """
        Initializes the ReplayMemory buffer with a NumPy array.

        Args:
            capacity (int): The maximum number of experiences to store.
            shuffle (bool): Whether to shuffle the experiences during sampling.
        """
        self.__capacity: int = capacity
        self.__shuffle: bool = shuffle
        self.__memory: np.ndarray = np.empty((capacity,), dtype=object)
        self.__position: int = 0
        self.__size: int = 0  # Tracks current number of stored experiences
        self.__is_full: bool = False

    def __len__(self) -> int:
        """
        Returns the current number of stored experiences.

        Returns:
            int: The number of stored experiences in the memory.
        """
        return self.__size

    @property
    def capacity(self) -> int:
        """Returns the maximum capacity of the replay memory."""
        return self.__capacity

    @property
    def shuffle(self) -> bool:
        """Returns whether shuffling is enabled for sampling."""
        return self.__shuffle

    @shuffle.setter
    def shuffle(self, value: bool) -> None:
        """
        Updates the shuffle setting.

        Args:
            value (bool): The new shuffle setting.
        """
        self.__shuffle = value
        
    @property
    def size(self) -> int:
        """Returns the current number of stored experiences."""
        return self.__size

    @property
    def memory(self) -> np.ndarray:
        """Returns the stored experiences in the replay memory."""
        return self.__memory[:self.__size]  # Return only filled experiences

    @property
    def position(self) -> int:
        """Returns the current position in the circular buffer."""
        return self.__position

    @property
    def is_full(self) -> bool:
        """Returns whether the replay memory is full."""
        return self.__is_full

    def add(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        """
        Adds an experience to the replay memory.

        Args:
            state (Any): The current state (e.g., stacked image tensor of shape (stack_size, height, width)).
            action (Any): The action taken (e.g., integer).
            reward (float): The reward received.
            next_state (Any): The next state (e.g., stacked image tensor).
            done (bool): Whether the episode has ended.
        """
        experience = (state, action, reward, next_state, done)

        self.__memory[self.__position] = experience
        self.__position = (self.__position + 1) % self.__capacity  # Overwrite when full
        self.__size = min(self.__size + 1, self.__capacity)  # Keep track of filled size
        self.__is_full = (self.__size == self.__capacity)  # Update is_full status

    def sample(self, batch_size: int, torch_tensor: bool = False):
        """
        Samples a batch of experiences from the memory.

        Args:
            batch_size (int): The number of experiences to sample.
            torch_tensor (bool): If True, returns experiences as PyTorch tensors.

        Returns:
            If torch_tensor=False:
                np.ndarray: A batch of sampled experiences.
            If torch_tensor=True:
                Tuple of PyTorch tensors: (states, actions, rewards, next_states, dones)
        """
        if self.__size == 0:
            return np.array([], dtype=object)  # Return empty array if no data available

        sample_size = min(self.__size, batch_size)  # Ensure valid sample size

        if self.__shuffle:
            indices = np.random.choice(self.__size, sample_size, replace=False)
        else:
            indices = np.arange(sample_size)

        batch = self.__memory[indices]  # Retrieve sampled batch

        # If torch_tensor=False, return as a NumPy array (original behavior)
        if not torch_tensor:
            return batch

        # Unpack batch into separate lists: (state, action, reward, next_state, done)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to PyTorch tensors (states and next_states are already tensors, just stack them)
        states = torch.stack(states)  # Shape: (batch_size, stack_size, height, width)
        actions = torch.tensor(actions, dtype=torch.long)  # Discrete actions → long tensor
        rewards = torch.tensor(rewards, dtype=torch.float32)  # Rewards → float tensor
        next_states = torch.stack(next_states)  # Shape: (batch_size, stack_size, height, width)
        dones = torch.tensor(dones, dtype=torch.bool)  # Boolean mask for episode end

        return states, actions, rewards, next_states, dones


if __name__ == '__main__':
    print("=== Testing ReplayMemory with Dummy Image Data ===\n")

    # Helper function to generate dummy stacked image frames
    def generate_dummy_image(stack_size: int = 4, height: int = 224, width: int = 224) -> torch.Tensor:
        """Generates a random stacked grayscale image tensor."""
        return torch.rand((stack_size, height, width))  # Shape: (stack_size, height, width)

    # Test initial length and is_full
    rm = ReplayMemory(capacity=5, shuffle=True)
    print("Test 1: Initial length and is_full")
    print("Expected length: 0, Actual length:", len(rm))
    print("Expected is_full: False, Actual is_full:", rm.is_full)

    # Test add and length increase with image data
    print("\nTest 2: Adding experiences with images and verifying length/is_full")
    state1 = generate_dummy_image()
    next_state1 = generate_dummy_image()
    rm.add(state1, 0, 1.0, next_state1, False)
    print("After 1st add, length:", len(rm), "is_full:", rm.is_full)
    state2 = generate_dummy_image()
    next_state2 = generate_dummy_image()
    rm.add(state2, 1, 2.0, next_state2, True)
    print("After 2nd add, length:", len(rm), "is_full:", rm.is_full)

    # Test circular buffer behavior and is_full with image data
    print("\nTest 3: Circular buffer overwrite and is_full with images")
    rm2 = ReplayMemory(capacity=3, shuffle=False)
    state3 = generate_dummy_image()
    next_state3 = generate_dummy_image()
    rm2.add(state3, 0, 0.1, next_state3, False)
    state4 = generate_dummy_image()
    next_state4 = generate_dummy_image()
    rm2.add(state4, 1, 0.2, next_state4, True)
    print("Memory before full (size=2):", rm2.memory, "is_full:", rm2.is_full)
    state5 = generate_dummy_image()
    next_state5 = generate_dummy_image()
    rm2.add(state5, 2, 0.3, next_state5, False)
    print("Memory when full (size=3):", rm2.memory, "is_full:", rm2.is_full)
    state6 = generate_dummy_image()
    next_state6 = generate_dummy_image()
    rm2.add(state6, 3, 0.4, next_state6, True)
    print("Memory after overwrite (should still be full):", rm2.memory, "is_full:", rm2.is_full)
    print("Current insertion position (expected 1):", rm2.position)

    # Test sample on empty memory
    print("\nTest 4: Sampling from empty memory")
    rm3 = ReplayMemory(capacity=10, shuffle=True)
    print("Sampled batch (expected empty array):", rm3.sample(2))
    print("is_full on empty memory:", rm3.is_full)

    # Test sample in sequential mode (shuffle=False) with image data
    print("\nTest 5: Sequential sampling (shuffle=False) with images")
    rm4 = ReplayMemory(capacity=10, shuffle=False)
    for i in range(5):
        state = generate_dummy_image()
        next_state = generate_dummy_image()
        rm4.add(state, i, float(i), next_state, i % 2 == 0)
    sample_seq = rm4.sample(2)
    print("Memory contents:", rm4.memory)
    print("Sequential sample (first 2 experiences):", sample_seq[0].shape if isinstance(sample_seq, tuple) else sample_seq.shape)
    print("is_full after 5 adds (capacity=10):", rm4.is_full)

    # Test sample in shuffle mode (shuffle=True) with image data
    print("\nTest 6: Shuffled sampling (shuffle=True) with images")
    rm5 = ReplayMemory(capacity=10, shuffle=True)
    for i in range(5):
        state = generate_dummy_image()
        next_state = generate_dummy_image()
        rm5.add(state, i, float(i), next_state, i % 2 == 0)
    sample_shuffled = rm5.sample(2, torch_tensor=True)
    print("Memory contents:", rm5.memory)
    print("Shuffled sample shapes - States:", sample_shuffled[0].shape, 
          "Actions:", sample_shuffled[1].shape, 
          "Rewards:", sample_shuffled[2].shape, 
          "Next States:", sample_shuffled[3].shape, 
          "Dones:", sample_shuffled[4].shape)
    print("is_full after 5 adds (capacity=10):", rm5.is_full)

    # Test getters and setters for shuffle
    print("\nTest 7: Getters and setters for shuffle")
    print("Initial shuffle (rm5):", rm5.shuffle)
    rm5.shuffle = False
    print("Updated shuffle (rm5):", rm5.shuffle)

    # Test memory, position, and is_full properties
    print("\nTest 8: Accessing memory, position, and is_full properties")
    print("Memory property (rm5):", rm5.memory)
    print("Position property (rm5):", rm5.position)
    print("is_full property (rm5):", rm5.is_full)

    print("\n=== All tests completed ===")