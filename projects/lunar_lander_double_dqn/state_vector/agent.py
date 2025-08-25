from typing import Optional

import torch
from torch import nn, optim
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from projects.lunar_lander_dqn.state_vector.net import LunarLanderMLP
from projects.lunar_lander_dqn.state_vector.replay_memory import ReplayMemory


class LunarLanderDoubleDQNAgent:
    """A Double Deep Q-Network (DQN) agent for solving the Lunar Lander environment.

    This agent maintains both a main Q-network and a target Q-network to 
    stabilize training. It uses an experience replay buffer to sample 
    batches of past transitions for learning. The agent supports epsilon-greedy 
    exploration, Double DQN updates, and saving/loading model weights.
    """

    def __init__(self,
                 state_size: int,
                 action_space_size: int,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay_steps: float = 1e6,
                 max_epsilon: float = 1.0,
                 min_epsilon: float = 1e-2,
                 memory_size: int = 1_000_000,
                 warmup_steps: int = 10_000,
                 shuffle: bool = True,
                 batch_size: int = 128,
                 sync_target_every: int = 10000,
                 model_weights_path: Optional[str] = None) -> None:
        """
        Initializes the Lunar Lander Double DQN agent.

        Args:
            state_size (int): Size of the state space (e.g., 8 for Lunar Lander).
            action_space_size (int): Number of possible actions (e.g., 4 for Lunar Lander).
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
            gamma (float, optional): Discount factor for future rewards. Defaults to 0.99.
            epsilon (float, optional): Initial exploration probability. Defaults to 1.0.
            epsilon_decay_steps (float, optional): Steps over which epsilon decays. Defaults to 1e6.
            max_epsilon (float, optional): Maximum epsilon value. Defaults to 1.0.
            min_epsilon (float, optional): Minimum epsilon value. Defaults to 1e-2.
            memory_size (int, optional): Capacity of the replay memory. Defaults to 1,000,000.
            warmup_steps (int, optional): Steps before training begins. Defaults to 10,000.
            shuffle (bool, optional): Whether to shuffle replay samples. Defaults to True.
            batch_size (int, optional): Batch size for training. Defaults to 128.
            sync_target_every (int, optional): Steps between target updates. Defaults to 10,000.
            model_weights_path (Optional[str], optional): Path to load pretrained weights. Defaults to None.
        """
        super().__init__()
        self.__state_size = state_size
        self.__action_space_size = action_space_size
        self.__learning_rate = learning_rate
        self.__gamma = gamma
        self.__epsilon = epsilon
        self.__epsilon_decay_steps: int = epsilon_decay_steps
        self.__max_epsilon = max_epsilon
        self.__min_epsilon = min_epsilon
        self.__memory_size = memory_size
        self.__warmup_steps = warmup_steps
        self.__shuffle = shuffle
        self.__batch_size = batch_size
        self.__sync_target_every = sync_target_every
        self.__model_weights_path = model_weights_path
        self.__learning_step = 0

        # DQN Networks
        self.__model = LunarLanderMLP(self.__state_size, self.__action_space_size)
        if self.__model_weights_path is not None:
            self.__model.load_model_data(self.__model_weights_path)
        self.__target_model = LunarLanderMLP(self.__state_size, self.__action_space_size)
        self.__target_model.load_state_dict(self.__model.state_dict())  # Sync initially
        self.__target_model.eval()
        
        self.__optimizer = optim.Adam(self.__model.parameters(), lr=self.__learning_rate)
        self.__criterion = nn.SmoothL1Loss()
        self.__replay_memory = ReplayMemory(self.__memory_size, self.__shuffle)
        
    # ==========================
    # Private Methods
    # ==========================
        
    def __calculate_expected_q_values(self, 
                                      next_states: torch.Tensor, 
                                      rewards: torch.Tensor, 
                                      dones: torch.Tensor) -> torch.Tensor:
        """
        Calculates the expected Q-values for the next states using Double DQN.

        Args:
            next_states (torch.Tensor): Batch of next states.
            rewards (torch.Tensor): Rewards received for each transition.
            dones (torch.Tensor): Boolean flags for episode termination.

        Returns:
            torch.Tensor: Expected Q-values for the next states.
        """
        with torch.no_grad():
            # Get the actions that maximize Q-values from the main model
            q_values_next = self.__model(next_states)
            next_actions = q_values_next.argmax(dim=1)

            # Use the target model to get the Q-values for these actions
            next_q_values = self.__target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            expected_q_values = rewards + self.__gamma * next_q_values * (1 - dones.float())
        
        return expected_q_values

    def __compute_q_values_and_targets(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples from replay memory and computes Q-values and target Q-values.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 
                - Q-values predicted by the main model.  
                - Expected Q-values computed with Double DQN.
        """
        states, actions, rewards, next_states, dones = self.__replay_memory.sample(batch_size=self.__batch_size, torch_tensor=True)
        q_values = self.__model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        expected_q_values = self.__calculate_expected_q_values(next_states, rewards, dones)

        return q_values, expected_q_values
    
    def __optimize(self, loss: torch.Tensor) -> None:
        """
        Performs a gradient descent step on the main model.

        Args:
            loss (torch.Tensor): Computed loss for the current batch.
        """
        self.__optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.__model.parameters(), max_norm=1.0)  # Clip gradients
        self.__optimizer.step()
        
    # ==========================
    # Properties Getters
    # ==========================
    
    @property
    def state_size(self) -> int:
        """Size of the state space."""
        return self.__state_size
    
    @property
    def action_space_size(self) -> int:
        """Number of possible actions."""
        return self.__action_space_size
    
    @property
    def learning_rate(self) -> float:
        """Learning rate for the optimizer."""
        return self.__learning_rate
    
    @property
    def gamma(self) -> float:
        """Discount factor for future rewards."""
        return self.__gamma
    
    @property
    def epsilon(self) -> float:
        """Current exploration probability (epsilon-greedy)."""
        return self.__epsilon
    
    @property
    def epsilon_decay_steps(self) -> float:
        """Number of steps over which epsilon decays to min epsilon."""
        return self.__epsilon_decay_steps
    
    @property
    def min_epsilon(self) -> float:
        """Minimum epsilon value allowed."""
        return self.__min_epsilon
    
    @property
    def memory_size(self) -> int:
        """Capacity of the replay memory buffer."""
        return self.__memory_size
    
    @property
    def warmup_steps(self) -> int:
        """Number of steps before training begins."""
        return self.__warmup_steps
    
    @property
    def shuffle(self) -> bool:
        """Whether replay memory samples are shuffled."""
        return self.__shuffle
    
    @property
    def batch_size(self) -> int:
        """Batch size used during training updates."""
        return self.__batch_size
    
    @property
    def sync_target_every(self) -> int:
        """Steps between syncing the target network with the main network."""
        return self.__sync_target_every
    
    @property
    def model_weights_path(self) -> str:
        """Path to the model weights (if loaded)."""
        return self.__model_weights_path
    
    @property
    def learning_step(self) -> int:
        """Number of training steps performed so far."""
        return self.__learning_step

    @property
    def model(self) -> LunarLanderMLP:
        """Main Q-network model."""
        return self.__model

    @property
    def target_model(self) -> LunarLanderMLP:
        """Target Q-network model."""
        return self.__target_model

    @property
    def optimizer(self) -> optim.Adam:
        """Optimizer for the main Q-network."""
        return self.__optimizer

    @property
    def criterion(self) -> nn.SmoothL1Loss:
        """Loss function used for training."""
        return self.__criterion

    @property
    def replay_memory(self) -> ReplayMemory:
        """Experience replay buffer."""
        return self.__replay_memory
    
    # ==========================
    # Properties Setters
    # ==========================
    
    @model_weights_path.setter
    def model_weights_path(self, path: str) -> None:
        """Sets the path to the model weights."""
        self.__model_weights_path = path
    
    # ==========================
    # Public Methods
    # ==========================
        
    def decay_epsilon(self, step: int) -> None:
        """
        Decays epsilon linearly towards min_epsilon over epsilon_decay_steps.

        Args:
            step (int): Current training step.
        """
        self.__epsilon = max(
            self.__min_epsilon,
            self.__max_epsilon - (self.__max_epsilon - self.__min_epsilon) * (step / self.__epsilon_decay_steps)
        )
        
    def choose_action(self, 
                      state: torch.Tensor, 
                      train_mode: bool = False) -> int:
        """
        Selects an action using epsilon-greedy policy.

        Args:
            state (torch.Tensor): Current environment state.

        Returns:
            int: Chosen action index.
        """
        self.__model.eval()
        
        with torch.inference_mode():
            if train_mode and torch.rand(1).item() < self.__epsilon:
                return torch.randint(0, self.__action_space_size, (1,)).item()
            q_values = self.__model(state.unsqueeze(0))
            return torch.argmax(q_values).item()
            
    def store_memory(self,
                     state: torch.Tensor,
                     action: int,
                     reward: float,
                     next_state: torch.Tensor,
                     done: bool) -> None:
        """
        Stores a transition in the replay buffer.

        Args:
            state (torch.Tensor): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (torch.Tensor): Next state observed.
            done (bool): Whether the episode terminated.
        """
        self.__replay_memory.add(state, action, reward, next_state, done)

    def update_target_model(self) -> None:
        """Copies weights from the main model to the target model."""
        self.__target_model.load_state_dict(self.__model.state_dict())

    def save_model(self, filepath: str) -> None:
        """
        Saves the main model weights.

        Args:
            filepath (str): File path to save the model.
        """
        torch.save(self.__model.state_dict(), filepath)
        
    def load_model(self, filepath: str) -> None:
        """
        Loads weights into the main model.

        Args:
            filepath (str): Path to the saved model weights.
        """
        self.__model.load_state_dict(torch.load(filepath))
        
    def save_target_model(self, filepath: str) -> None:
        """
        Saves the target model weights.

        Args:
            filepath (str): File path to save the target model.
        """
        torch.save(self.__target_model.state_dict(), filepath)
        
    def load_target_model(self, filepath: str) -> None:
        """
        Loads weights into the target model.

        Args:
            filepath (str): Path to the saved target model weights.
        """
        self.__target_model.load_state_dict(torch.load(filepath))

    def learn(self, return_loss: bool = False) -> torch.Tensor | None:
        """
        Performs one training step using sampled experiences.

        Args:
            return_loss (bool, optional): Whether to return the computed loss. Defaults to False.

        Returns:
            torch.Tensor | None: Loss value if return_loss is True, otherwise None.
        """
        if not len(self.__replay_memory) >= self.__warmup_steps:
            return
        
        self.__model.train()
        
        q_values, expected_q_values = self.__compute_q_values_and_targets()
        loss = self.__criterion(q_values, expected_q_values)
        self.__optimize(loss)
        
        self.__learning_step += 1
        
        if self.__learning_step % self.__sync_target_every == 0:
            self.update_target_model()
        
        if return_loss:
            return loss
