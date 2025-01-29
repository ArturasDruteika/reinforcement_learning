from typing import Tuple

import torch
from torch import nn


class SPNet(nn.Module):
    """
    Stochastic Policy Network (SPNet) for continuous action spaces.
    
    This neural network learns a policy that outputs a mean action and 
    a standard deviation (sigma) for a Gaussian distribution from which 
    actions are sampled.

    Attributes:
        fc1 (nn.Linear): Fully connected hidden layer.
        fc_mu (nn.Linear): Linear layer to output mean action values.
        fc_log_sigma (nn.Linear): Linear layer to output log standard deviation.
        relu (nn.ReLU): Activation function.
    """

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        """
        Initializes the SPNet model.
        
        Args:
            input_dim (int): Dimensionality of the input state.
            action_dim (int): Dimensionality of the action space.
            hidden_dim (int, optional): Number of neurons in the hidden layer. Defaults to 128.
        """
        super(SPNet, self).__init__()
        self.__input_dim = input_dim
        self.__action_dim = action_dim
        self.__hidden_dim = hidden_dim
        
        self.fc1 = nn.Linear(self.__input_dim, self.__hidden_dim)
        self.fc_mu = nn.Linear(self.__hidden_dim, self.__action_dim)  # Mean output
        self.fc_log_sigma = nn.Linear(self.__hidden_dim, self.__action_dim)  # Log std deviation
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor representing the state.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Mean (mu) tensor for the Gaussian distribution.
                - Standard deviation (sigma) tensor for the Gaussian distribution.
        """
        x = self.relu(self.fc1(x))  # Hidden layer with activation
        mu = self.fc_mu(x)  # Mean of the action distribution
        
        # Log standard deviation to ensure sigma is always positive
        log_sigma = self.fc_log_sigma(x)
        sigma = torch.exp(log_sigma)

        return mu, sigma
    
    
if __name__ == '__main__':
    input_dim = 4
    action_dim = 5
    hidden_dim = 128
    sp_net = SPNet(input_dim, action_dim, hidden_dim)
    print(sp_net)
    
    state = torch.randn(1, input_dim)
    mu, sigma = sp_net(state)
    print('Mean:', mu)
    print('Sigma:', sigma)