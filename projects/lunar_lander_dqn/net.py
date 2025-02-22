import torch
from torch import nn
from torch import Tensor


class LunarLanderConvNet(nn.Module):
    """
    A Convolutional Neural Network (CNN) designed for processing images from the Lunar Lander environment.

    The network consists of:
    - Two convolutional layers with ReLU activation.
    - Max-pooling layers to reduce spatial dimensions.
    - Two fully connected layers for classification.
    - Outputs 4 neurons, corresponding to the possible actions in the discrete Lunar Lander environment.

    Expected input shape: (batch_size, 3, 224, 224)
    """

    def __init__(self) -> None:
        """
        Initializes the CNN model with convolutional, pooling, and fully connected layers.
        """
        super(LunarLanderConvNet, self).__init__()

        # First Convolutional Layer: 3 input channels (RGB), 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # MaxPooling Layer (2x2) to downsample feature maps
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  # Based on reduced feature map size
        self.fc2 = nn.Linear(128, 4)  # 4 output neurons (discrete actions in Lunar Lander)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the forward pass of the CNN.

        Args:
            x (Tensor): Input tensor of shape (batch_size, 3, 224, 224).

        Returns:
            Tensor: Output tensor of shape (batch_size, 4).
        """
        x = self.relu(self.conv1(x))
        x = self.pool(x)  # Reduce size to (112, 112)

        x = self.relu(self.conv2(x))
        x = self.pool(x)  # Reduce size to (56, 56)

        x = x.view(x.size(0), -1)  # Flatten before fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Output layer

        return x
    
    def save_model_data(self, filepath: str) -> None:
        """
        Saves the model parameters to a file.

        Args:
            filename (str): The name of the file to save the model parameters.
        """
        torch.save(self.state_dict(), filepath)
        
    def load_model_data(self, filepath: str) -> None:
        """
        Loads the model parameters from a file with safety restrictions.

        Args:
            filepath (str): The path of the file to load the model parameters.
        """
        self.load_state_dict(torch.load(filepath, weights_only=True))


# Testing the model with additional tests for all functions
if __name__ == '__main__':
    # Instantiate and print the model architecture
    model = LunarLanderConvNet()
    print("Initial model:")
    print(model)

    # Test forward function with dummy input: batch_size=1, channels=3, height=224, width=224
    sample_input = torch.randn(1, 3, 224, 224)
    output = model(sample_input)
    print("\nInitial output from forward pass:")
    print(output)
    print(f"Output shape: {output.shape}")  # Expected: [1, 4]

    # Test save_model_data function by saving the model's state_dict
    model_filepath = "lunar_lander_conv_net.pth"
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
