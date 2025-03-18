import torch
from torch import nn
from torch import optim
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from projects.lunar_lander_dqn.rgb_mode.net import LunarLanderCNN
from projects.lunar_lander_dqn.rgb_mode.replay_memory import ReplayMemory


class LunarLanderDQNAgent:
    
    def __init__(
        self,
        action_state_size,
        learning_rate = 1e-4, 
        gamma = 0.99, 
        epsilon = 1.0, 
        epsilon_decay = 0.999, 
        min_epsilon = 1e-4, 
        memory_size = 10_000,
        shuffle = True,
        batch_size = 64,
        sync_target_every: int = 10_000,
        device = None
    ):
        self.__action_state_size = action_state_size
        self.__learning_rate = learning_rate
        self.__gamma = gamma
        self.__epsilon = epsilon
        self.__epsilon_decay = epsilon_decay
        self.__min_epsilon = min_epsilon
        self.__memory_size = memory_size
        self.__shuffle = shuffle
        self.__batch_size = batch_size
        self.__sync_target_every: int = sync_target_every
        self.__learning_step = 0
        if device is not None:
            self.__device = device
        else:
            self.__device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.__model = LunarLanderCNN(
            input_channels=4, 
            input_height=96, 
            input_width=96
        ).to(self.__device)
        self.__target_model = LunarLanderCNN(
            input_channels=4, 
            input_height=96, 
            input_width=96
        ).to(self.__device)
        self.__target_model.load_state_dict(self.__model.state_dict())
        self.__target_model.eval()
        
        self.__optimizer = optim.Adam(self.__model.parameters(), lr = self.__learning_rate)
        self.__criterion = nn.SmoothL1Loss()
        
        self.__replay_memory = ReplayMemory(self.__memory_size, self.__shuffle)
        
    @property
    def action_state_size(self):
        return self.__action_state_size
        
    @property
    def learning_rate(self):
        return self.__learning_rate
    
    @property
    def gamma(self):
        return self.__gamma
    
    @property
    def epsilon(self):
        return self.__epsilon
    
    @epsilon.setter
    def epsilon(self, value):
        self.__epsilon = value
    
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
    def sync_target_every(self):
        return self.__sync_target_every
    
    @property
    def learning_step(self):
        return self.__learning_step
    
    @property
    def model(self):
        return self.__model
    
    @property
    def target_model(self):
        return self.__target_model
    
    @property
    def optimizer(self):
        return self.__optimizer
    
    @property
    def criterion(self):
        return self.__criterion
    
    @property
    def replay_memory(self):
        return self.__replay_memory
        
    def decay_epsilon(self) -> None:
        """
        Decreases the epsilon value for exploration.
        """
        self.__epsilon = max(self.__epsilon * self.__epsilon_decay, self.__min_epsilon)
        
    def choose_action(self, state: torch.tensor) -> int:
        self.__model.eval()
        
        with torch.inference_mode():
            if torch.rand(1).item() < self.epsilon:
                return torch.randint(0, 4, (1,)).item()
            else:
                return torch.argmax(self.model(state.unsqueeze(0).to(self.__device))).item()
            
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
        print("Target model updated.")
        
    def save_model(self, filepath: str) -> None:
        self.__model.save_model_data(filepath)
        
    def load_model(self, filepath: str) -> None:
        self.__model.load_model_data(filepath)
        
    def save_target_model(self, filepath: str) -> None:
        self.__target_model.save_model_data(filepath)
        
    def load_target_model(self, filepath: str) -> None:
        self.__target_model.load_model_data(filepath)
        
    def learn(self, return_loss = False):
        if not self.__replay_memory.is_full:
            return None
        
        self.__model.train()
        
        states, actions, rewards, next_states, dones = self.__replay_memory.sample(
            batch_size=self.__batch_size,
            torch_tensor=True
        )
        
        states = states.to(self.__device)
        actions = actions.to(self.__device)
        rewards = rewards.to(self.__device)
        next_states = next_states.to(self.__device)
        dones = dones.to(self.__device)
        
        q_values = self.__model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        with torch.no_grad():
            next_q_values = self.__target_model(next_states).max(dim=1).values
            expected_q_values = rewards + self.__gamma * next_q_values * (1 - dones.float())
            
        loss = self.__criterion(q_values, expected_q_values)
        
        self.__optimizer.zero_grad()
        loss.backward()
        self.__optimizer.step()
        
        self.__learning_step += 1
        
        if self.__learning_step % self.__sync_target_every == 0:
            self.update_target_model()
        
        if return_loss:
            return loss