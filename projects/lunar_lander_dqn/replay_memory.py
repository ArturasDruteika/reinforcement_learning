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
    replay_memory = ReplayMemory(capacity=1000, batch_size=32, shuffle=True)

    # Add some experience
    replay_memory.add(
        state=[1,2,3,4], 
        action=1, 
        reward=1.0,
        next_state=[2,3,4,5], 
        done=False
    )

    # Sample experiences
    batch = replay_memory.sample()
    print(f"Sampled batch size: {len(batch)}")