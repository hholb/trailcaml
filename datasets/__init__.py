from io import BytesIO
from pathlib import Path

from PIL import Image
import polars as pl
import torch
from torch.utils.data import Dataset

from preprocessors import ImagePreprocessor

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
