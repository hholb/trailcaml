from pathlib import Path

import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from preprocessors import ImagePreprocessor
from datasets import TrailCameraDataset


BASE_URL = "hf://datasets/Francesco/trail-camera/"
DATA_DIR = "data"
DATASET_LOCATIONS = {
    'train': 'data/train-00000-of-00001-931b9615f2251ad8.parquet',
    'validation': 'data/validation-00000-of-00001-ac26c1956c34fa02.parquet',
    'test': 'data/test-00000-of-00001-11d0ac39410a634d.parquet',
}

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


def _fetch_dataset() -> dict[str, Path]:
    if not Path(DATA_DIR).exists():
        Path(DATA_DIR).mkdir()

    parquet_paths = {}
    for key, value in DATASET_LOCATIONS.items():
        f = Path(value)
        if not f.exists():
            # download the parquet from hugging face if we don't have it locally
            df = pl.read_parquet(BASE_URL + value)
            df.write_parquet(f)
        parquet_paths[key] = f

    return parquet_paths


def main(batch_size=32, learning_rate=1e-4, epochs=5):
    data_paths = _fetch_dataset()

    train_data = TrailCameraDataset(data_paths['train']) 
    train_dataloader = DataLoader(train_data, batch_size=batch_size)

    test_data = TrailCameraDataset(data_paths['test'])
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    validation_data = TrailCameraDataset(data_paths['validation'])
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size)

    model = TrailCaML()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)

    print(f"Validating\n-------------------------------")
    test_loop(validation_dataloader, model, loss_fn)

    print("Done!")
    return model


