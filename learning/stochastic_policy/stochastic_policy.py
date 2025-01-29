from typing import Tuple

import rootutils
import torch
from torch import distributions as dist

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from learning.stochastic_policy.sp_net import SPNet


class StochasticPolicy:
    """
    Stochastic Policy class for continuous action spaces.
    
    This class uses a neural network (SPNet) to output a mean action and 
    standard deviation, defining a Gaussian distribution from which actions 
    are sampled.
    
    Attributes:
        __state_dim (int): The dimensionality of the state space.
        __action_dim (int): The dimensionality of the action space.
        __sp_net (SPNet): Neural network for predicting action distributions.
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        """
        Initializes the Stochastic Policy.

        Args:
            state_dim (int): The dimensionality of the state space.
            action_dim (int): The dimensionality of the action space.
        """
        self.__state_dim = state_dim
        self.__action_dim = action_dim
        self.__sp_net = SPNet(state_dim, action_dim)

    def sample_action(self, state: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """
        Samples an action from the learned policy distribution.
        
        Args:
            state (torch.Tensor): The input state tensor.
        
        Returns:
            Tuple[float, torch.Tensor]: 
                - The sampled action as a scalar (or vector if action_dim > 1).
                - The log probability of the sampled action.
        """
        mu, sigma = self.__sp_net(state)  # Get mean and std deviation
        normal_dist = dist.Normal(mu, sigma)  # Define Gaussian distribution
        
        action = normal_dist.sample()  # Sample an action
        log_prob = normal_dist.log_prob(action)  # Log probability for training
        
        return action.item(), log_prob
    

if __name__ == '__main__':
    state_dim = 4
    action_dim = 1
    policy = StochasticPolicy(state_dim, action_dim)
    
    state = torch.randn(1, state_dim)
    action, log_prob = policy.sample_action(state)
    print(f'Sampled action: {action:.4f}, Log probability: {log_prob}')