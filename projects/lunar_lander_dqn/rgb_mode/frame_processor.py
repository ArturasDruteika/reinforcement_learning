import numpy as np
import torchvision.transforms as transforms
from PIL import Image


class FramePreprocessor:
    """Preprocesses frames by resizing, converting to tensors, and applying ImageNet normalization."""

    def __init__(self):
        self.__transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to fit ResNet or similar CNNs
            transforms.ToTensor(),  # Convert to PyTorch Tensor (float32)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
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