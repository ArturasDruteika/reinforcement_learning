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
        batch_size (int): The number of experiences to sample in a batch.
        shuffle (bool): Whether to shuffle the experiences during sampling.
    """

    def __init__(self, capacity: int, batch_size: int, shuffle: bool) -> None:
        """
        Initializes the ReplayMemory buffer with a NumPy array.

        Args:
            capacity (int): The maximum number of experiences to store.
            batch_size (int): The number of experiences to sample.
            shuffle (bool): Whether to shuffle the experiences during sampling.
        """
        self.__capacity: int = capacity
        self.__batch_size: int = batch_size
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
    def batch_size(self) -> int:
        """Returns the batch size used for sampling experiences."""
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        """
        Updates the batch size.

        Args:
            value (int): The new batch size.
        """
        self.__batch_size = value

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
            state (Any): The current state.
            action (Any): The action taken.
            reward (float): The reward received.
            next_state (Any): The next state.
            done (bool): Whether the episode has ended.
        """
        experience = (state, action, reward, next_state, done)

        self.__memory[self.__position] = experience
        self.__position = (self.__position + 1) % self.__capacity  # Overwrite when full
        self.__size = min(self.__size + 1, self.__capacity)  # Keep track of filled size
        self.__is_full = (self.__size == self.__capacity)  # Update is_full status

    def sample(self, torch_tensor=False):
        """
        Samples a batch of experiences from the memory.

        Args:
            torch_tensor (bool): If True, returns experiences as PyTorch tensors.

        Returns:
            If torch_tensor=False:
                np.ndarray: A batch of sampled experiences.
            If torch_tensor=True:
                Tuple of PyTorch tensors: (states, actions, rewards, next_states, dones)
        """
        if self.__size == 0:
            return np.array([], dtype=object)  # Return empty array if no data available

        sample_size = min(self.__size, self.__batch_size)  # Ensure valid sample size

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

        # Convert to PyTorch tensors
        states = torch.stack(states)  # Assuming states are already torch tensors
        actions = torch.tensor(actions, dtype=torch.long)  # Discrete actions → long tensor
        rewards = torch.tensor(rewards, dtype=torch.float32)  # Rewards → float tensor
        next_states = torch.stack(next_states)  # Assuming next_states are already torch tensors
        dones = torch.tensor(dones, dtype=torch.bool)  # Boolean mask for episode end

        return states, actions, rewards, next_states, dones


if __name__ == '__main__':
    print("=== Testing ReplayMemory with NumPy Array ===\n")

    # Test initial length and is_full
    rm = ReplayMemory(capacity=5, batch_size=2, shuffle=True)
    print("Test 1: Initial length and is_full")
    print("Expected length: 0, Actual length:", len(rm))
    print("Expected is_full: False, Actual is_full:", rm.is_full)

    # Test add and length increase
    print("\nTest 2: Adding experiences and verifying length/is_full")
    rm.add("state1", "action1", 1.0, "state2", False)
    print("After 1st add, length:", len(rm), "is_full:", rm.is_full)
    rm.add("state2", "action2", 2.0, "state3", True)
    print("After 2nd add, length:", len(rm), "is_full:", rm.is_full)

    # Test circular buffer behavior and is_full
    print("\nTest 3: Circular buffer overwrite and is_full")
    rm2 = ReplayMemory(capacity=3, batch_size=2, shuffle=False)
    rm2.add(10, 'a', 0.1, 20, False)
    rm2.add(11, 'b', 0.2, 21, True)
    print("Memory before full (size=2):", rm2.memory, "is_full:", rm2.is_full)
    rm2.add(12, 'c', 0.3, 22, False)
    print("Memory when full (size=3):", rm2.memory, "is_full:", rm2.is_full)
    # Add one more to trigger overwrite
    rm2.add(13, 'd', 0.4, 23, True)
    print("Memory after overwrite (should still be full):", rm2.memory, "is_full:", rm2.is_full)
    print("Current insertion position (expected 1):", rm2.position)

    # Test sample on empty memory
    print("\nTest 4: Sampling from empty memory")
    rm3 = ReplayMemory(capacity=10, batch_size=3, shuffle=True)
    print("Sampled batch (expected empty array):", rm3.sample())
    print("is_full on empty memory:", rm3.is_full)

    # Test sample in sequential mode (shuffle=False)
    print("\nTest 5: Sequential sampling (shuffle=False)")
    rm4 = ReplayMemory(capacity=10, batch_size=3, shuffle=False)
    for i in range(5):
        rm4.add(i, f"action{i}", float(i), i + 1, i % 2 == 0)
    sample_seq = rm4.sample()
    print("Memory contents:", rm4.memory)
    print("Sequential sample (first 3 experiences):", sample_seq)
    print("is_full after 5 adds (capacity=10):", rm4.is_full)

    # Test sample in shuffle mode (shuffle=True)
    print("\nTest 6: Shuffled sampling (shuffle=True)")
    rm5 = ReplayMemory(capacity=10, batch_size=3, shuffle=True)
    for i in range(5):
        rm5.add(i, f"action{i}", float(i), i + 1, i % 2 == 0)
    sample_shuffled = rm5.sample()
    print("Memory contents:", rm5.memory)
    print("Shuffled sample (3 random experiences):", sample_shuffled)
    print("is_full after 5 adds (capacity=10):", rm5.is_full)

    # Test getters and setters for batch_size
    print("\nTest 7: Getters and setters for batch_size")
    print("Initial batch_size (rm5):", rm5.batch_size)
    rm5.batch_size = 4
    print("Updated batch_size (rm5):", rm5.batch_size)

    # Test getters and setters for shuffle
    print("\nTest 8: Getters and setters for shuffle")
    print("Initial shuffle (rm5):", rm5.shuffle)
    rm5.shuffle = False
    print("Updated shuffle (rm5):", rm5.shuffle)

    # Test memory, position, and is_full properties
    print("\nTest 9: Accessing memory, position, and is_full properties")
    print("Memory property (rm5):", rm5.memory)
    print("Position property (rm5):", rm5.position)
    print("is_full property (rm5):", rm5.is_full)

    print("\n=== All tests completed ===")