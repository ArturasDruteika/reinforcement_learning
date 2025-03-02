import torch
from torch import nn, Tensor


class LunarLanderMLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) for Lunar Lander with:
    - Leaky ReLU (instead of standard ReLU)
    - Layer Normalization (better than BatchNorm in DQN)
    - Dropout (to prevent overfitting)

    This network processes an 8-dimensional state and outputs Q-values for 4 actions.
    """

    def __init__(self, input_size: int, output_size: int, dropout_rate=0.2) -> None:
        """
        Initializes the MLP model with:
            - Fully connected layers
            - Layer Normalization
            - Leaky ReLU activations
            - Dropout
        
        Args:
            input_size (int): Dimensionality of the input state (8 for Lunar Lander).
            output_size (int): Dimensionality of the output action space (4 discrete actions).
            dropout_rate (float): Probability of dropout (default 0.2).
        """
        super().__init__()

        # Define MLP layers using helper function
        self.fc1 = self.__create_mlp_layer(input_size, 64, dropout_rate)
        self.fc2 = self.__create_mlp_layer(64, 128, dropout_rate)
        self.fc3 = self.__create_mlp_layer(128, 64, dropout_rate)
        self.fc4 = nn.Linear(64, output_size)  # Output layer (raw Q-values)

    def __create_mlp_layer(self, input_size: int, output_size: int, dropout_rate: float) -> nn.Sequential:
        """
        Creates a fully connected MLP layer with:
        - Linear layer
        - Layer Normalization
        - Leaky ReLU activation
        - Dropout

        Args:
            input_size (int): Number of input neurons.
            output_size (int): Number of output neurons.
            dropout_rate (float): Dropout probability.

        Returns:
            nn.Sequential: A sequential block of layers.
        """
        return nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LayerNorm(output_size),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the forward pass of the MLP.

        Args:
            x (Tensor): Input tensor of shape (batch_size, 8).

        Returns:
            Tensor: Output tensor of shape (batch_size, 4).
        """
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.fc4(x)  # Output raw Q-values (NO softmax!)

    def save_model_data(self, filepath: str) -> None:
        """Saves the model parameters to a file."""
        torch.save(self.state_dict(), filepath)

    def load_model_data(self, filepath: str) -> None:
        """Loads the model parameters from a file."""
        self.load_state_dict(torch.load(filepath))


# Testing the model with a dummy input
if __name__ == '__main__':
    model = LunarLanderMLP(8, 4)
    sample_input = torch.randn(1, 8)
    output = model(sample_input)
    print("Model Output:", output)
