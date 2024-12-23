import numpy as np
from PIL import Image
import torch

class ImagePreprocessor:
    def __init__(
        self, 
        size: tuple[int, int] = (640, 640),
    ):
        self.size = size

    def __call__(
        self,
        img: Image,
    ) -> torch.tensor:
        # Convert to grayscale and resize
        image = img.convert('L').resize(self.size)
        # convert to float and normalize values
        image = np.array(image, dtype=np.float32)
        image = torch.tensor(image) / 255
        # add a channel dimension. Output shape: (1, 640, 640)
        image = image.unsqueeze(0)
        return image


