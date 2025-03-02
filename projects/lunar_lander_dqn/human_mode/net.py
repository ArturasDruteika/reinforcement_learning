import torch
from torch import nn
from torch import Tensor


class LunarLanderMLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) designed for processing an 8-dimensional input from the Lunar Lander environment.

    The network consists of:
    - Three fully connected layers with ReLU activation.
    - Outputs 4 neurons, corresponding to the possible actions in the discrete Lunar Lander environment.

    Expected input shape: (batch_size, 8)
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        """
        Initializes the MLP model with fully connected layers and activation functions.
        
        Args:
            input_size (int): Dimensionality of the input state.
            output_size (int): Dimensionality of the output action space.
        """
        super(LunarLanderMLP, self).__init__()

        # Fully Connected Layers
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)  # 4 output neurons (discrete actions in Lunar Lander)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the forward pass of the MLP.

        Args:
            x (Tensor): Input tensor of shape (batch_size, 8).

        Returns:
            Tensor: Output tensor of shape (batch_size, 4).
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # Output layer
        return x
    
    def save_model_data(self, filepath: str) -> None:
        """
        Saves the model parameters to a file.

        Args:
            filepath (str): The name of the file to save the model parameters.
        """
        torch.save(self.state_dict(), filepath)
        
    def load_model_data(self, filepath: str) -> None:
        """
        Loads the model parameters from a file with safety restrictions.

        Args:
            filepath (str): The path of the file to load the model parameters.
        """
        self.load_state_dict(torch.load(filepath))


# Testing the model with additional tests for all functions
if __name__ == '__main__':
    # Instantiate and print the model architecture
    model = LunarLanderMLP(8, 4)
    print("Initial model:")
    print(model)

    # Test forward function with dummy input: batch_size=1, input_size=8
    sample_input = torch.randn(1, 8)
    output = model(sample_input)
    print("\nInitial output from forward pass:")
    print(output)
    print(f"Output shape: {output.shape}")  # Expected: [1, 4]

    # Test save_model_data function by saving the model's state_dict
    model_filepath = "lunar_lander_mlp.pth"
    model.save_model_data(model_filepath)
    print(f"\nModel parameters saved to '{model_filepath}'.")

    # Modify the model's parameters to simulate training or accidental changes
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param))
    modified_output = model(sample_input)
    print("\nOutput after modifying model weights:")
    print(modified_output)

    # Test load_model_data function by reloading the saved parameters
    model.load_model_data(model_filepath)
    loaded_output = model(sample_input)
    print("\nOutput after reloading saved model parameters:")
    print(loaded_output)
    print(f"Loaded output shape: {loaded_output.shape}")

    # Compare the initial output and the output after reloading to ensure consistency
    difference = torch.abs(output - loaded_output).sum().item()
    print(f"\nTotal difference between initial and reloaded outputs: {difference:.6f}")
