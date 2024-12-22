import numpy as np
from PIL import Image
import torch

class ImagePreprocessor:
    def __call__(
        self,
        img: Image,
        size: tuple[int, int] = (64, 64),
    ) -> torch.tensor:
        # Convert to grayscale and resize
        image = img.convert('L').resize(size)
        # convert to float and normalize values
        image = np.array(image, dtype=np.float32)
        image = torch.tensor(image) / 255
        # add a batch dimension. Output shape: (1, 64, 64)
        image = image.unsqueeze(0)
        return image


