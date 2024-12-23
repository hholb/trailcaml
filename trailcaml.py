from collections import namedtuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from preprocessors import ImagePreprocessor


ImageSize = namedtuple("ImageSize", ["x", "y"])

DEFAULT_CONV_LAYERS = nn.Sequential(
    # First conv block: 1x640x640 -> 32x638x638 -> 32x319x319
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    # Second conv block: 32x319x319 -> 64x317x317 -> 64x158x158
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    # Third conv block: 64x158x158 -> 128x156x156 -> 128x78x78
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),
)

class TrailCaML(nn.Module):
    def __init__(self, img_size: ImageSize = ImageSize(640, 640), conv_layers: nn.Module = DEFAULT_CONV_LAYERS):
        super().__init__()
        self.conv_layers = conv_layers
        self.flatten = nn.Flatten()

        # Calculate flattened image size
        with torch.no_grad():
            dummy_input = torch.zeros(
                1, 1, img_size.x, img_size.y
            )
            # (channels, height, width)
            dummy_output = self.flatten(self.conv_layers(dummy_input))
            # (channels, output_size)
            flattened_size = dummy_output.shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 256), 
            nn.ReLU(),
            nn.Linear(256, 4),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # print the progress every few batches
        if batch % 5 == 0:
            loss, current = loss.item(), batch * dataloader.batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    
    # Initialize metrics
    exact_matches = 0  # All classes predicted correctly
    per_class_correct = torch.zeros(4, device=device)  # Correct predictions per class
    per_class_total = torch.zeros(4, device=device)    # Total instances per class

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            
            # Get binary predictions
            probabilities = torch.sigmoid(pred)
            predictions = (probabilities > 0.5).float()
            
            # Count exact matches (all classes correct)
            exact_matches += (predictions == y).all(dim=1).sum().item()
            
            # Count per-class correct predictions
            per_class_correct += ((predictions == y) & (y == 1)).sum(dim=0)
            per_class_total += (y == 1).sum(dim=0)

    # Calculate metrics
    test_loss /= num_batches
    exact_match_accuracy = exact_matches / size
    
    # Calculate per-class accuracy
    class_names = ["no animal", "deer", "pig", "coyote"]
    per_class_accuracy = per_class_correct / per_class_total.clamp(min=1)  # Avoid division by zero
    
    # Print results
    print(f"Test Error:")
    print(f"Exact Match Accuracy: {(100*exact_match_accuracy):>0.1f}%")
    print(f"Average Loss: {test_loss:>8f}")
    print("\nPer-class Accuracy:")
    for name, acc in zip(class_names, per_class_accuracy):
        print(f"{name}: {(100*acc):>0.1f}%")


def train_model(
    model: TrailCaML,
    train: Dataset,
    test: Dataset,
    batch_size: int = 32,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    optimizer=torch.optim.SGD,
    loss_fn=nn.BCEWithLogitsLoss(),
) -> TrailCaML:
    dataloader_args = {
        "batch_size": batch_size,
        "pin_memory": torch.cuda.is_available(),
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        dataloader_args["pin_memory_device"] = device

    train_dataloader = DataLoader(
        train,
        shuffle=True,
        **dataloader_args,
    )
    test_dataloader = DataLoader(
        test,
        **dataloader_args,
    )

    if device == "cuda":
        print("Moving model to CUDA device...")
        model.to(device)

    optimizer = optimizer(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        test_loop(test_dataloader, model, loss_fn, device)

    print("Done!")
    return model


def contains_animal(image, model):
    img = ImagePreprocessor()(image)
    # add the batch dimension
    img = img.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        # Get model predictions
        pred = model(img)

        # Apply softmax to get probabilities
        probabilities = nn.Softmax(dim=1)(pred)
        return probabilities.argmax(1).item() != 0
