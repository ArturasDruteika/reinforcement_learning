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
        self.conv_1 = self.__create_conv_layer(input_channels, 16, 3, 1, 1)
        self.conv_2 = self.__create_conv_layer(16, 32, 3, 1, 1)
        self.conv_3 = self.__create_conv_layer(32, 64, 3, 1, 1)

        # Fully Connected Layers (updated for global average pooling)
        self.mlp = self.__create_mlp_layer(self.__calculate_flat_size(), 64, 256)
        self.fc_out: nn.Linear = nn.Linear(64, output_size)
        
    def __create_conv_layer(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int) -> nn.Conv2d:
        """
        Create a single convolutional layer with a LeakyReLU activation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride for the convolution operation.
            padding (int): Padding to apply around the input.

        Returns:
            nn.Conv2d: Convolutional layer with LeakyReLU activation.
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def __create_mlp_layer(self, in_features: int, out_features: int, hidden_dim: int) -> nn.Linear:
        """
        Create a single fully connected layer with a LeakyReLU activation.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.

        Returns:
            nn.Linear: Fully connected layer with LeakyReLU activation.
        """
        return nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_features),
            nn.LeakyReLU()
        )
        
    def __calculate_flat_size(self):
        return 64 * (self.__input_height // 8) * (self.__input_width // 8)

    @property
    def input_channels(self) -> int:
        return self.__input_channels

    @property
    def input_height(self) -> int:
        return self.__input_height

    @property
    def input_width(self) -> int:
        return self.__input_width

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the forward pass of the CNN with global average pooling.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_channels, input_height, input_width).

        Returns:
            Tensor: Output tensor of shape (batch_size, output_size) containing Q-values for each action.
        """
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        
        x = x.view(x.size(0), -1)

        x = self.mlp(x)
        x = self.fc_out(x)

        return x

    def save_model_data(self, filepath: str) -> None:
        torch.save(self.state_dict(), filepath)

    def load_model_data(self, filepath: str) -> None:
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

    # Calculate and print total parameters
    total_params = sum(p.numel() for p in model_96.parameters())
    print(f"\nTotal parameters: {total_params}")