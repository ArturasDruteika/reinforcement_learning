import torch
from torch import nn, optim
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from projects.lunar_lander_dqn.human_mode.net import LunarLanderMLP
from projects.lunar_lander_dqn.human_mode.replay_memory import ReplayMemory


class LunarLanderDQNAgent:
    """A Deep Q-Network (DQN) agent for solving the Lunar Lander environment."""

    def __init__(
        self,
        state_size: int,
        action_space_size: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.999,
        min_epsilon: float = 1e-6,
        memory_size: int = 100_000,
        shuffle: bool = False,
        batch_size: int = 128,
        sync_target_every: int = 10000,
    ) -> None:
        """
        Initializes the Lunar Lander DQN agent with networks, memory, and hyperparameters.

        Args:
            state_size (int): Size of the state space (e.g., 8 for Lunar Lander).
            action_space_size (int): Number of possible actions (e.g., 4 for Lunar Lander).
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
            gamma (float, optional): Discount factor for future rewards. Defaults to 0.99.
            epsilon (float, optional): Initial exploration probability. Defaults to 1.0.
            epsilon_decay (float, optional): Decay rate for epsilon. Defaults to 0.999.
            min_epsilon (float, optional): Minimum epsilon value. Defaults to 1e-6.
            memory_size (int, optional): Capacity of the replay memory. Defaults to 100,000.
            shuffle (bool, optional): Whether to shuffle replay memory samples. Defaults to False.
            batch_size (int, optional): Number of transitions sampled per training step. Defaults to 128.
            sync_target_every (int, optional): Steps between target network updates. Defaults to 10.
        """
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
        self.__learning_step = 0

        # DQN Networks
        self.__model = LunarLanderMLP(self.__state_size, self.__action_space_size)
        self.__target_model = LunarLanderMLP(self.__state_size, self.__action_space_size)
        self.__target_model.load_state_dict(self.__model.state_dict())  # Sync initially
        self.__target_model.eval()
        
        self.__optimizer = optim.AdamW(self.__model.parameters(), lr=self.__learning_rate)
        self.__criterion = nn.SmoothL1Loss()  # Correct loss function for Q-learning
        self.__replay_memory = ReplayMemory(self.__memory_size, self.__batch_size, self.__shuffle)

    def __compute_q_values_and_targets(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples from replay memory and computes Q-values and target Q-values for a DQN training step.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Predicted Q-values and target Q-values for the sampled batch.
        """
        # Sample from replay memory
        states, actions, rewards, next_states, dones = self.__replay_memory.sample(batch_size=self.__batch_size, torch_tensor=True)

        # Compute Q-values for current states
        q_values = self.__model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.__target_model(next_states).max(1).values
            expected_q_values = rewards + self.__gamma * next_q_values * (1 - dones.float())

        return q_values, expected_q_values
    
    def __optimize(self, loss: torch.Tensor) -> None:
        """
        Performs an optimization step on the main model using the computed loss.

        Args:
            loss (torch.Tensor): Loss value to backpropagate.
        """
        self.__optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.__model.parameters(), max_norm=1.0)  # Clip gradients
        self.__optimizer.step()
    
    @property
    def state_size(self) -> int:
        """int: Size of the state space."""
        return self.__state_size
    
    @property
    def action_space_size(self) -> int:
        """int: Number of possible actions."""
        return self.__action_space_size
    
    @property
    def learning_rate(self) -> float:
        """float: Learning rate for the optimizer."""
        return self.__learning_rate
    
    @property
    def gamma(self) -> float:
        """float: Discount factor for future rewards."""
        return self.__gamma
    
    @property
    def epsilon(self) -> float:
        """float: Current exploration probability."""
        return self.__epsilon
    
    @property
    def epsilon_decay(self) -> float:
        """float: Decay rate for epsilon."""
        return self.__epsilon_decay
    
    @property
    def min_epsilon(self) -> float:
        """float: Minimum epsilon value."""
        return self.__min_epsilon
    
    @property
    def memory_size(self) -> int:
        """int: Capacity of the replay memory."""
        return self.__memory_size
    
    @property
    def shuffle(self) -> bool:
        """bool: Whether replay memory samples are shuffled."""
        return self.__shuffle
    
    @property
    def batch_size(self) -> int:
        """int: Number of transitions sampled per training step."""
        return self.__batch_size
    
    @property
    def sync_target_every(self) -> int:
        """int: Steps between target network updates."""
        return self.__sync_target_every
    
    @property
    def learning_step(self) -> int:
        """int: Number of training steps performed."""
        return self.__learning_step

    @property
    def model(self) -> LunarLanderMLP:
        """LunarLanderMLP: Main Q-network model."""
        return self.__model

    @property
    def target_model(self) -> LunarLanderMLP:
        """LunarLanderMLP: Target Q-network model."""
        return self.__target_model

    @property
    def optimizer(self) -> optim.AdamW:
        """optim.AdamW: Optimizer for the main model."""
        return self.__optimizer

    @property
    def criterion(self) -> nn.SmoothL1Loss:
        """nn.SmoothL1Loss: Loss function for Q-learning."""
        return self.__criterion

    @property
    def replay_memory(self) -> ReplayMemory:
        """ReplayMemory: Experience replay buffer."""
        return self.__replay_memory

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        """
        Sets the exploration probability.

        Args:
            value (float): New epsilon value.
        """
        self.__epsilon = value
        
    def decay_epsilon(self) -> None:
        """Decreases the epsilon value for exploration, respecting the minimum threshold."""
        self.__epsilon = max(self.__epsilon * self.__epsilon_decay, self.__min_epsilon)
        
    def choose_action(self, state: torch.Tensor) -> int:
        """
        Selects an action using an epsilon-greedy policy based on the current state.

        Args:
            state (torch.Tensor): Current state of the environment.

        Returns:
            int: Selected action index.
        """
        with torch.no_grad():
            if torch.rand(1).item() < self.__epsilon:
                return torch.randint(0, self.__action_space_size, (1,)).item()
            else:
                return torch.argmax(self.__model(state)).item()
            
    def store_memory(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool
    ) -> None:
        """
        Stores a transition in the replay memory.

        Args:
            state (torch.Tensor): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (torch.Tensor): Next state observed.
            done (bool): Whether the episode terminated.
        """
        self.__replay_memory.add(state, action, reward, next_state, done)

    def update_target_model(self) -> None:
        """Updates the target network by copying weights from the main model."""
        self.__target_model.load_state_dict(self.__model.state_dict())

    def save_model(self, filepath: str) -> None:
        """
        Saves the main model’s weights to a file.

        Args:
            filepath (str): Path to save the model weights.
        """
        torch.save(self.__model.state_dict(), filepath)
        
    def load_model(self, filepath: str) -> None:
        """
        Loads the main model’s weights from a file.

        Args:
            filepath (str): Path to the saved model weights.
        """
        self.__model.load_state_dict(torch.load(filepath))
        
    def save_target_model(self, filepath: str) -> None:
        """
        Saves the target model’s weights to a file.

        Args:
            filepath (str): Path to save the target model weights.
        """
        torch.save(self.__target_model.state_dict(), filepath)
        
    def load_target_model(self, filepath: str) -> None:
        """
        Loads the target model’s weights from a file.

        Args:
            filepath (str): Path to the saved target model weights.
        """
        self.__target_model.load_state_dict(torch.load(filepath))

    def learn(self, return_loss: bool = False) -> torch.Tensor | None:
        """
        Performs one training step using sampled experiences from replay memory.

        Args:
            return_loss (bool, optional): Whether to return the loss value. Defaults to False.

        Returns:
            torch.Tensor | None: Loss value if return_loss is True, otherwise None.
        """
        # Ensure memory size is full
        if not self.__replay_memory.is_full:
            return

        q_values, expected_q_values = self.__compute_q_values_and_targets()
        # Compute loss
        loss = self.__criterion(q_values, expected_q_values)
        # Perform optimization step
        self.__optimize(loss)
        
        self.__learning_step += 1
        
        if self.__learning_step % self.__sync_target_every == 0:
            self.update_target_model()
        
        if return_loss:
            return loss