import torch
from torch import nn, Tensor


class LunarLanderMLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) for Lunar Lander.

    This network processes an 8-dimensional state and outputs Q-values for 4 actions.
    Note: The current implementation uses a basic MLP structure without additional features
    like LeakyReLU, Layer Normalization, or Dropout, despite the initial description.
    """

    def __init__(self, input_size: int, output_size: int, hidden_size: int = 64) -> None:
        """
        Initialize the MLP model with fully connected layers.

        Args:
            input_size (int): Dimensionality of the input state (e.g., 8 for Lunar Lander).
            output_size (int): Dimensionality of the output action space (e.g., 4 discrete actions).
            hidden_size (int, optional): Number of units in the hidden layers. Defaults to 64.
        """
        super().__init__()

        # Define MLP layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)  # Output layer (raw Q-values)

    def forward(self, x: Tensor) -> Tensor:
        """
        Define the forward pass of the MLP.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            Tensor: Output tensor of shape (batch_size, output_size) containing raw Q-values.
        """
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc_out(x)  # Output raw Q-values (no softmax applied)
        return x

    def save_model_data(self, filepath: str) -> None:
        """
        Save the model parameters to a file.

        Args:
            filepath (str): Path to the file where the model parameters will be saved.
        """
        torch.save(self.state_dict(), filepath)

    def load_model_data(self, filepath: str) -> None:
        """
        Load the model parameters from a file.

        Args:
            filepath (str): Path to the file containing the model parameters.
        """
        self.load_state_dict(torch.load(filepath))


# Testing the model with a dummy input
if __name__ == '__main__':
    model = LunarLanderMLP(8, 4)
    sample_input = torch.randn(1, 8)
    output = model(sample_input)
    print("Model Output:", output)
    