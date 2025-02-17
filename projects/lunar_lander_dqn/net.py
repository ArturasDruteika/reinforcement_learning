import torch
from torch import nn


class LunarLanderConvNet(nn.Module):
    def __init__(self):
        super(LunarLanderConvNet, self).__init__()

        # First Convolutional Layer: (3 input channels, 32 output channels, 3x3 kernel)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Pooling layer (2x2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  # Based on reduced feature map size
        self.fc2 = nn.Linear(128, 4)  # 4 output neurons

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)  # Reduce size to 112x112

        x = self.relu(self.conv2(x))
        x = self.pool(x)  # Reduce size to 56x56

        x = x.view(x.size(0), -1)  # Flatten before FC layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Output layer

        return x


# Testing the model with a dummy input
if __name__ == '__main__':
    model = LunarLanderConvNet()
    print(model)

    # Dummy input (batch_size=1, channels=3, height=224, width=224)
    sample_input = torch.randn(1, 3, 224, 224)
    output = model(sample_input)
    print(output)
    print(f"Output shape: {output.shape}")  # Expected: [1, 4]
