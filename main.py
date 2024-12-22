from io import BytesIO
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

BASE_URL = "hf://datasets/Francesco/trail-camera/"
DATA_DIR = "data"
DATASET_LOCATIONS = {
    'train': 'data/train-00000-of-00001-931b9615f2251ad8.parquet',
    'validation': 'data/validation-00000-of-00001-ac26c1956c34fa02.parquet',
    'test': 'data/test-00000-of-00001-11d0ac39410a634d.parquet',
}
IMAGE_WIDTH = 640

class ImagePreprocessor:
    def __call__(self, img: Image) -> torch.tensor:
        image = img.convert('L').resize((IMAGE_WIDTH,IMAGE_WIDTH))
        image = np.array(image, dtype=np.float32)
        image = torch.tensor(image) / 255
        image = image.unsqueeze(0)
        return image


class TrailCameraDataset(Dataset):
    """ 
    url: https://huggingface.co/datasets/Francesco/trail-camera
    """
    def __init__(
        self,
        parquet: str | Path | None = None,
        preprocessor: ImagePreprocessor = ImagePreprocessor(),
    ):
        if parquet:
            self.data = pl.scan_parquet(parquet).collect(streaming=True)
            self.length = self.data.height
        assert self.data is not None
        self.preprocessor = preprocessor


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        row = self.data.row(index)
        image = self._image_from_row(row)
        label = self._label_from_row(row)

        image = self.preprocessor(image)

        return  image, label

    def _get_label(self, index: int) -> torch.tensor:
        row = self.data.row(index)
        return self._label_from_row(row)

    def _label_from_row(self, row) -> torch.tensor:
        labels = row[4]['category']
        contains_animal = int(any([label > 0 for label in labels]))
        label = contains_animal
        return label

    def _image_from_row(self, row) -> Image:
        image_bytes = row[1]['bytes']
        image = Image.open(BytesIO(image_bytes))
        return image


class TrailCaML(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            # First conv layer: 32 filters, 3x3 kernel
            # Input: 1x640x640, Output: 32x638x638
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: 32x319x319
            
            # Second conv layer: 64 filters, 3x3 kernel
            # Output: 64x317x317
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: 64x158x158
            
            # Third conv layer: 128 filters, 3x3 kernel
            # Output: 128x156x156
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: 128x78x78
            
            # Flatten the output for the final classification layers
            nn.Flatten(),
            nn.Linear(128 * 78 * 78, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.conv_stack(x)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 5 == 0:
            loss, current = loss.item(), batch * dataloader.batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



def contains_animal(image, model):
    img = ImagePreprocessor()(image)
    
    # Add batch dimension - transforms shape to (1, 1, height, width)
    # The first 1 is the batch size (one image)
    # The second 1 is the number of channels (grayscale)
    img = img.unsqueeze(0)
    
    model.eval()
    
    with torch.no_grad():
        # Get model predictions
        # pred will have shape (1, 2) - one batch, two classes (no animal, animal)
        pred = model(img)
        
        # Apply softmax to get probabilities
        probabilities = nn.Softmax(dim=1)(pred)
        
        # Get the predicted class (0 for no animal, 1 for animal)
        is_animal = probabilities.argmax(1).item() == 1
        confidence = probabilities[0][is_animal][0][is_animal].item()
        
        return is_animal, confidence


def main(batch_size=32, learning_rate=1e-4, epochs=5):
    data_paths = _load_or_fetch_dataset()

    train_data = TrailCameraDataset(data_paths['train']) 
    analyze_class_distribution(train_data)
    train_dataloader = DataLoader(train_data, batch_size=batch_size)

    test_data = TrailCameraDataset(data_paths['test'])
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    validation_data = TrailCameraDataset(data_paths['validation'])
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size)

    model = TrailCaML()
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)

    print(f"Validating\n-------------------------------")
    test_loop(validation_dataloader, model, loss_fn)

    print("Done!")
    return model


def _load_or_fetch_dataset() -> dict[str, Path]:
    if not Path(DATA_DIR).exists():
        Path(DATA_DIR).mkdir()

    parquet_paths = {}
    for key, value in DATASET_LOCATIONS.items():
        f = Path(value)
        if not f.exists():
            df = pl.read_parquet(BASE_URL + value)
            df.write_parquet(f)
        parquet_paths[key] = f
    return parquet_paths


def analyze_class_distribution(dataset):
    animal_count = 0
    total_count = len(dataset)

    for i in range(total_count):
        label = dataset._get_label(i)
        if label == 1:
            animal_count += 1

    no_animal_count = total_count - animal_count
    
    print(f"Total images: {total_count}")
    print(f"Images with animals: {animal_count} ({(animal_count/total_count)*100:.1f}%)")
    print(f"Images without animals: {no_animal_count} ({(no_animal_count/total_count)*100:.1f}%)")
    
    # Calculate the baseline accuracy (if we just guessed the majority class)
    majority_accuracy = max(animal_count, no_animal_count) / total_count
    print(f"\nBaseline accuracy (always guessing majority): {majority_accuracy*100:.1f}%")
