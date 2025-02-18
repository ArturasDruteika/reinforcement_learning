import random


class ReplayMemory:
    
    def __init__(self, capacity, batch_size, shuffle):
        self.__capacity = capacity
        self.__batch_size = batch_size
        self.__shuffle = shuffle
        self.__memory = []
        self.__position = 0
        
    def __len__(self):
        """Returns the current number of stored experiences."""
        return len(self.__memory)
        
    @property
    def capacity(self):
        return self.__capacity
    
    @property
    def batch_size(self):
        return self.__batch_size
    
    @property
    def shuffle(self):
        return self.__shuffle
    
    @property
    def memory(self):
        return self.__memory
    
    @property
    def position(self):
        return self.__position
    
    @batch_size.setter
    def batch_size(self, value):
        self.__batch_size = value
        
    @shuffle.setter
    def shuffle(self, value):
        self.__shuffle = value
        
    def add(self, state, action, reward, next_state, done):
        if len(self.__memory) < self.__capacity:
            self.__memory.append(None)
        self.__memory[self.__position] = (state, action, reward, next_state, done)
        self.__position = (self.__position + 1) % self.__capacity
        
    def sample(self):
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