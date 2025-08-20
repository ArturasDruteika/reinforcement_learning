from typing import Tuple

import numpy as np
import torchvision.transforms as transforms
from PIL import Image


class FramePreprocessor:
    """Preprocesses frames by resizing, converting to tensors, and applying ImageNet normalization.

    This class creates a transformation pipeline for preprocessing image frames, typically for
    input to convolutional neural networks (CNNs) like ResNet. The pipeline resizes frames,
    converts them to PyTorch tensors, and applies normalization using ImageNet statistics.
    """

    def __init__(self, resize_shape: Tuple[int, int] = (224, 224)) -> None:
        """Initializes the FramePreprocessor with a transformation pipeline.

        Args:
            resize_shape (Tuple[int, int], optional): The target size for resizing frames (height, width).
                Defaults to (224, 224) to match common CNN input sizes.

        Attributes:
            __transform (transforms.Compose): The composed transformation pipeline for preprocessing.
        """
        self.__transform = transforms.Compose([
            transforms.Resize(resize_shape),  # Resize to fit ResNet or similar CNNs
            transforms.ToTensor(),  # Convert to PyTorch Tensor (float32)
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def preprocess(self, frame):
        """Applies transformations: resize, convert to tensor, and normalize."""
        return self.__transform(frame)  # `frame` should be a PIL image or NumPy array


if __name__ == '__main__':
    preprocessor = FramePreprocessor()

    # Example: Load an image (assuming `image` is a NumPy array)
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)  # Fake RGB image
    image = Image.fromarray(image)  # Convert NumPy to PIL Image

    processed_frame = preprocessor.preprocess(image)  # Apply transformations

    print(processed_frame.shape)  # Should be (3, 224, 224)