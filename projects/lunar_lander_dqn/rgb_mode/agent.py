import torch
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from projects.lunar_lander_dqn.net import LunarLanderConvNet
from projects.lunar_lander_dqn.replay_memory import ReplayMemory


class LunarLanderDQNAgent:
    
    def __init__(
        self, 
        learning_rate = 1e-5, 
        gamma = 0.99, 
        epsilon = 1.0, 
        epsilon_decay = 0.999, 
        min_epsilon = 1e-5, 
        memory_size = 1024,
        shuffle = False,
        batch_size = 32
    ):
        self.__learning_rate = learning_rate
        self.__gamma = gamma
        self.__epsilon = epsilon
        self.__epsilon_decay = epsilon_decay
        self.__min_epsilon = min_epsilon
        self.__memory_size = memory_size
        self.__shuffle = shuffle
        self.__batch_size = batch_size

        self.__model = LunarLanderConvNet()
        self.__target_model = LunarLanderConvNet()
        self.__replay_memory = ReplayMemory(self.__memory_size, self.__batch_size, self.__shuffle)
        
    @property
    def learning_rate(self):
        return self.__learning_rate
    
    @property
    def gamma(self):
        return self.__gamma
    
    @property
    def epsilon(self):
        return self.__epsilon
    
    @property
    def epsilon_decay(self):
        return self.__epsilon_decay
    
    @property
    def min_epsilon(self):
        return self.__min_epsilon
    
    @property
    def memory_size(self):
        return self.__memory_size
    
    @property
    def shuffle(self):
        return self.__shuffle
    
    @property
    def batch_size(self):
        return self.__batch_size
    
    @property
    def model(self):
        return self.__model
    
    @property
    def target_model(self):
        return self.__target_model
    
    @property
    def replay_memory(self):
        return self.__replay_memory
        
    def decay_epsilon(self) -> None:
        """
        Decreases the epsilon value for exploration.
        """
        self._epsilon = max(self._epsilon * self._epsilon_decay, self._min_epsilon)
        
    def choose_action(self, state: torch.tensor) -> int:
        with torch.no_grad():
            if torch.rand(1).item() < self.epsilon:
                return torch.randint(0, 4, (1,)).item()
            else:
                return torch.argmax(self.model(state)).item()
            
    def store_memory(
        self, 
        state: torch.tensor, 
        action: int, 
        reward: float, 
        next_state: torch.tensor, 
        done: bool
    ) -> None:
        self.__replay_memory.add(state, action, reward, next_state, done)
        
    def update_target_model(self) -> None:
        self.__target_model.load_state_dict(self.__model.state_dict())
        
    def save_model(self, filepath: str) -> None:
        self.__model.save_model_data(filepath)
        
    def load_model(self, filepath: str) -> None:
        self.__model.load_model_data(filepath)
        
    def save_target_model(self, filepath: str) -> None:
        self.__target_model.save_model_data(filepath)
        
    def load_target_model(self, filepath: str) -> None:
        self.__target_model.load_model_data(filepath)