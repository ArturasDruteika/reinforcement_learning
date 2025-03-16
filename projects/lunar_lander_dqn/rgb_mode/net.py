import torch
from torch import nn
from torch import Tensor


class LunarLanderCNN(nn.Module):
    """
    A Convolutional Neural Network (CNN) designed for processing images from the Lunar Lander environment.
    Supports input shapes where height and width are divisible by 32.
    """

    def __init__(self, input_channels: int = 4, input_height: int = 224, input_width: int = 224, output_size: int = 4) -> None:
        """
        Initializes the CNN model with convolutional, pooling, and fully connected layers.

        Args:
            input_channels (int): Number of input channels (default: 4 for stacked frames).
            input_height (int): Height of input image, must be divisible by 32 (default: 224).
            input_width (int): Width of input image, must be divisible by 32 (default: 224).
            output_size (int): Number of output neurons (default: 4 for Lunar Lander actions).

        Raises:
            ValueError: If input_height or input_width is not divisible by 32.
        """
        super(LunarLanderCNN, self).__init__()

        self.__input_channels: int = input_channels
        self.__input_height: int = input_height
        self.__input_width: int = input_width

        # Validate input shape divisibility
        if input_height % 32 != 0 or input_width % 32 != 0:
            raise ValueError(f"Input height ({input_height}) and width ({input_width}) must be divisible by 32.")

        # First Convolutional Layer
        self.conv1: nn.Conv2d = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2: nn.Conv2d = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # MaxPooling Layer (2x2)
        self.pool: nn.MaxPool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc1: nn.Linear = nn.Linear(self.__calculate_flat_size(), 128)
        self.fc2: nn.Linear = nn.Linear(128, output_size)

        # Activation function
        self.relu: nn.ReLU = nn.ReLU()

    def __calculate_flat_size(self) -> int:
        """
        Calculates the flattened size of the feature map after convolutional and pooling layers.

        The size is determined as follows:
        - After conv1 + pool: (H/2, W/2)
        - After conv2 + pool: (H/4, W/4)
        - Flattened size: 64 * (H/4) * (W/4)

        Returns:
            int: The size of the flattened feature map.
        """
        return 64 * (self.__input_height // 4) * (self.__input_width // 4)

    @property
    def input_channels(self) -> int:
        """
        Gets the number of input channels.

        Returns:
            int: The number of input channels.
        """
        return self.__input_channels

    @property
    def input_height(self) -> int:
        """
        Gets the height of the input image.

        Returns:
            int: The input height.
        """
        return self.__input_height

    @property
    def input_width(self) -> int:
        """
        Gets the width of the input image.

        Returns:
            int: The input width.
        """
        return self.__input_width

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the forward pass of the CNN.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_channels, input_height, input_width).

        Returns:
            Tensor: Output tensor of shape (batch_size, output_size) containing Q-values for each action.
        """
        x = self.relu(self.conv1(x))
        x = self.pool(x)  # Reduce size by 2

        x = self.relu(self.conv2(x))
        x = self.pool(x)  # Reduce size by 2 again

        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Output layer

        return x

    def save_model_data(self, filepath: str) -> None:
        """
        Saves the model parameters to a file.

        Args:
            filepath (str): The path to the file where model parameters will be saved.
        """
        torch.save(self.state_dict(), filepath)

    def load_model_data(self, filepath: str) -> None:
        """
        Loads the model parameters from a file with safety restrictions.

        Args:
            filepath (str): The path to the file from which to load the model parameters.

        Raises:
            FileNotFoundError: If the specified filepath does not exist.
            RuntimeError: If the loaded state dict does not match the model architecture.
        """
        self.load_state_dict(torch.load(filepath, weights_only=True))


# Testing the model with additional tests
if __name__ == "__main__":
    # Test with default size (224x224, divisible by 32)
    model: LunarLanderCNN = LunarLanderCNN(input_channels=4, input_height=224, input_width=224)
    print("Model with input shape (4, 224, 224):")
    print(model)

    # Test forward pass
    sample_input: Tensor = torch.randn(1, 4, 224, 224)
    output: Tensor = model(sample_input)
    print("\nOutput from forward pass:")
    print(output)
    print(f"Output shape: {output.shape}")  # Expected: [1, 4]

    # Test with another size divisible by 32 (e.g., 96x96)
    model_96: LunarLanderCNN = LunarLanderCNN(input_channels=4, input_height=96, input_width=96)
    print("\nModel with input shape (4, 96, 96):")
    print(model_96)

    sample_input_96: Tensor = torch.randn(1, 4, 96, 96)
    output_96: Tensor = model_96(sample_input_96)
    print("\nOutput from forward pass (96x96):")
    print(output_96)
    print(f"Output shape: {output_96.shape}")  # Expected: [1, 4]

    # Test invalid size (not divisible by 32)
    try:
        model_invalid: LunarLanderCNN = LunarLanderCNN(input_channels=4, input_height=100, input_width=100)
    except ValueError as e:
        print("\nExpected error for invalid size (100x100):", str(e))

    # Test save and load
    model_filepath: str = "lunar_lander_conv_net.pth"
    model.save_model_data(model_filepath)
    print(f"\nModel parameters saved to '{model_filepath}'.")

    model.load_model_data(model_filepath)
    loaded_output: Tensor = model(sample_input)
    print("\nOutput after reloading saved model:")
    print(loaded_output)
    print(f"Total difference: {torch.abs(output - loaded_output).sum().item():.6f}")