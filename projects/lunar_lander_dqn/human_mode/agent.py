import torch
from torch import nn, optim
import rootutils
from lightning import LightningModule

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from projects.lunar_lander_dqn.human_mode.net import LunarLanderMLP
from projects.lunar_lander_dqn.human_mode.replay_memory import ReplayMemory


class LunarLanderDQNAgent(LightningModule):  # Now inherits from LightningModule
    
    def __init__(
        self, 
        state_size,
        action_space_size,
        learning_rate=1e-5, 
        gamma=0.99, 
        epsilon=1.0, 
        epsilon_decay=0.999, 
        min_epsilon=1e-6, 
        memory_size=1024,
        shuffle=False,
        batch_size=32,
        sync_target_every=10,  # How often to update the target model
    ):
        super().__init__()
        self.__state_size = state_size
        self.__action_space_size = action_space_size
        self.__learning_rate = learning_rate
        self.__gamma = gamma
        self.__epsilon = epsilon
        self.__epsilon_decay = epsilon_decay
        self.__min_epsilon = min_epsilon
        self.__memory_size = memory_size
        self.__shuffle = shuffle
        self.__batch_size = batch_size
        self.__sync_target_every = sync_target_every

        # DQN Networks
        self.__model = LunarLanderMLP(self.__state_size, self.__action_space_size)
        self.__target_model = LunarLanderMLP(self.__state_size, self.__action_space_size)
        self.__target_model.load_state_dict(self.__model.state_dict())  # Sync initially
        
        self.__optimizer = optim.AdamW(self.__model.parameters(), lr=self.__learning_rate)
        self.__criterion = nn.MSELoss()  # Correct loss function for Q-learning
        self.__replay_memory = ReplayMemory(self.__memory_size, self.__batch_size, self.__shuffle)

        self.save_hyperparameters()
        
    @property
    def state_size(self):
        return self.__state_size
    
    @property
    def action_space_size(self):
        return self.__action_space_size
    
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
    def sync_target_every(self):
        return self.__sync_target_every

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

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        self.__epsilon = value
        
    def decay_epsilon(self) -> None:
        """Decreases the epsilon value for exploration."""
        self.__epsilon = max(self.__epsilon * self.__epsilon_decay, self.__min_epsilon)
        
    def choose_action(self, state: torch.tensor) -> int:
        """Selects an action using an epsilon-greedy policy."""
        with torch.no_grad():
            if torch.rand(1).item() < self.__epsilon:
                return torch.randint(0, self.__action_space_size, (1,)).item()
            else:
                return torch.argmax(self.__model(state)).item()
            
    def store_memory(
        self, 
        state: torch.tensor, 
        action: int, 
        reward: float, 
        next_state: torch.tensor, 
        done: bool
    ) -> None:
        """Stores the experience in replay memory."""
        self.__replay_memory.add(state, action, reward, next_state, done)

    def update_target_model(self) -> None:
        """Updates the target network with the main model's weights."""
        self.__target_model.load_state_dict(self.__model.state_dict())

    def save_model(self, filepath: str) -> None:
        """Saves the model to a file."""
        torch.save(self.__model.state_dict(), filepath)
        
    def load_model(self, filepath: str) -> None:
        """Loads the model from a file."""
        self.__model.load_state_dict(torch.load(filepath))
        
    def save_target_model(self, filepath: str) -> None:
        """Saves the target model to a file."""
        torch.save(self.__target_model.state_dict(), filepath)
        
    def load_target_model(self, filepath: str) -> None:
        """Loads the target model from a file."""
        self.__target_model.load_state_dict(torch.load(filepath))

    def training_step(self):
        """
        Manually performs one training step (called once per `train()` call).
        """
        # Ensure enough experience is collected before training
        if self.__replay_memory.size < self.__batch_size:
            return None

        # Sample from replay memory
        states, actions, rewards, next_states, dones = self.__replay_memory.sample(torch_tensor=True)

        # Compute Q-values for current states
        q_values = self.__model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.__target_model(next_states).max(1).values
            expected_q_values = rewards + self.__gamma * next_q_values * (1 - dones.float())

        # Compute loss
        loss = self.__criterion(q_values, expected_q_values)
        self.log("train_loss", loss, prog_bar=True)

        # Perform optimization step
        self.__optimizer.zero_grad()
        loss.backward()
        self.__optimizer.step()

        return loss

    def configure_optimizers(self):
        """Configures the optimizer for Lightning."""
        return self.__optimizer

    def on_train_epoch_end(self):
        """Syncs target model weights every `sync_target_every` epochs."""
        if self.current_epoch % self.__sync_target_every == 0:
            self.update_target_model()
