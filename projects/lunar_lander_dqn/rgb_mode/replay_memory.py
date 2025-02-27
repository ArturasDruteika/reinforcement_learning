import random
from typing import List, Tuple, Any, Optional


class ReplayMemory:
    """
    A replay memory buffer for storing and sampling experiences in reinforcement learning.

    This class implements a circular buffer where new experiences replace the oldest ones
    when the memory reaches its maximum capacity.

    Attributes:
        capacity (int): The maximum number of experiences the memory can store.
        batch_size (int): The number of experiences to sample in a batch.
        shuffle (bool): Whether to randomly shuffle the experiences during sampling.
    """

    def __init__(self, capacity: int, batch_size: int, shuffle: bool) -> None:
        """
        Initializes the ReplayMemory buffer.

        Args:
            capacity (int): The maximum number of experiences to store.
            batch_size (int): The number of experiences to sample.
            shuffle (bool): Whether to shuffle the experiences during sampling.
        """
        self.__capacity: int = capacity
        self.__batch_size: int = batch_size
        self.__shuffle: bool = shuffle
        self.__memory: List[Optional[Tuple[Any, Any, float, Any, bool]]] = []
        self.__position: int = 0

    def __len__(self) -> int:
        """
        Returns the current number of stored experiences.

        Returns:
            int: The number of experiences in the memory.
        """
        return len(self.__memory)

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
    def memory(self) -> List[Optional[Tuple[Any, Any, float, Any, bool]]]:
        """Returns the stored experiences in the replay memory."""
        return self.__memory

    @property
    def position(self) -> int:
        """Returns the current position in the circular buffer."""
        return self.__position

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

        if len(self.__memory) < self.__capacity:
            self.__memory.append(None)  # Expand memory size if needed

        self.__memory[self.__position] = experience
        self.__position = (self.__position + 1) % self.__capacity  # Overwrite when full

    def sample(self) -> List[Tuple[Any, Any, float, Any, bool]]:
        """
        Samples a batch of experiences from the memory.

        Returns:
            List[Tuple[Any, Any, float, Any, bool]]: A batch of sampled experiences.
        """
        if len(self.__memory) == 0:
            return []  # Return empty list if no data available

        sample_size = min(len(self.__memory), self.__batch_size)  # Avoid out-of-range error

        if self.__shuffle:
            return random.sample(self.__memory, sample_size)
        else:
            return self.__memory[:sample_size]  # Take sequentially if shuffle=False
        

if __name__ == '__main__':
    print("=== Testing ReplayMemory ===\n")
    
    # Test initial length
    rm = ReplayMemory(capacity=5, batch_size=2, shuffle=True)
    print("Test 1: Initial length")
    print("Expected length: 0, Actual length:", len(rm))
    
    # Test add and length increase
    print("\nTest 2: Adding experiences and verifying length")
    rm.add("state1", "action1", 1.0, "state2", False)
    print("After 1st add, length:", len(rm))
    rm.add("state2", "action2", 2.0, "state3", True)
    print("After 2nd add, length:", len(rm))
    
    # Test circular buffer behavior (overwriting)
    print("\nTest 3: Circular buffer overwrite")
    rm2 = ReplayMemory(capacity=3, batch_size=2, shuffle=False)
    rm2.add(10, 'a', 0.1, 20, False)
    rm2.add(11, 'b', 0.2, 21, True)
    rm2.add(12, 'c', 0.3, 22, False)
    print("Memory before overwrite (should have 3 experiences):", rm2.memory)
    # Add one more to trigger overwrite
    rm2.add(13, 'd', 0.4, 23, True)
    print("Memory after overwrite (oldest overwritten):", rm2.memory)
    print("Current insertion position (expected 1):", rm2.position)
    
    # Test sample on empty memory
    print("\nTest 4: Sampling from empty memory")
    rm3 = ReplayMemory(capacity=10, batch_size=3, shuffle=True)
    print("Sampled batch (expected empty list):", rm3.sample())
    
    # Test sample in sequential mode (shuffle=False)
    print("\nTest 5: Sequential sampling (shuffle=False)")
    rm4 = ReplayMemory(capacity=10, batch_size=3, shuffle=False)
    for i in range(5):
        rm4.add(i, f"action{i}", float(i), i + 1, i % 2 == 0)
    sample_seq = rm4.sample()
    print("Memory contents:", rm4.memory)
    print("Sequential sample (first 3 experiences):", sample_seq)
    
    # Test sample in shuffle mode (shuffle=True)
    print("\nTest 6: Shuffled sampling (shuffle=True)")
    rm5 = ReplayMemory(capacity=10, batch_size=3, shuffle=True)
    for i in range(5):
        rm5.add(i, f"action{i}", float(i), i + 1, i % 2 == 0)
    sample_shuffled = rm5.sample()
    print("Memory contents:", rm5.memory)
    print("Shuffled sample (3 random experiences):", sample_shuffled)
    
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
    
    # Test memory and position properties
    print("\nTest 9: Accessing memory and position properties")
    print("Memory property (rm5):", rm5.memory)
    print("Position property (rm5):", rm5.position)
    
    print("\n=== All tests completed ===")