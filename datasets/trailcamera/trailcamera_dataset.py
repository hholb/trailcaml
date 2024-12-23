from io import BytesIO
from pathlib import Path

from PIL import Image
import polars as pl
import torch
from torch.utils.data import Dataset

from preprocessors import ImagePreprocessor


# see: https://huggingface.co/datasets/Francesco/trail-camera
BASE_URL = "hf://datasets/Francesco/trail-camera/"
DATASET_LOCATIONS = {
    'train': 'data/train-00000-of-00001-931b9615f2251ad8.parquet',
    'validation': 'data/validation-00000-of-00001-ac26c1956c34fa02.parquet',
    'test': 'data/test-00000-of-00001-11d0ac39410a634d.parquet',
}

ANIMAL_CLASSES = {
    "none": 0,
    "deer": 1,
    "pig": 2,
    "coyote": 3,
}

class TrailCameraDataset(Dataset):
    """ 
    url: https://huggingface.co/datasets/Francesco/trail-camera
    """
    def __init__(
        self,
        parquet: str | Path,
        preprocessor: ImagePreprocessor = ImagePreprocessor(),
    ):
        self.data = pl.scan_parquet(parquet)
        self.length = self.data.count().collect()['image'][0]
        self.preprocessor = preprocessor

    def __len__(self):
        return self.length

    def _get_row(self, index):
        return self.data.slice(index, 1).collect().row(0)

    def __getitem__(self, index):
        row = self._get_row(index)
        label = self._label_from_row(row)
        image = self.preprocessor(self._image_from_row(row))
        return  image, label

    def _label_from_row(self, row: pl.DataFrame) -> list[int]:
        labels = row[4]['category']
        # label structure = [no animal, deer, pig, coyote]
        labels = [
            1.0 if not any(labels) else 0.0,
            1.0 if ANIMAL_CLASSES['deer'] in labels else 0.0,
            1.0 if ANIMAL_CLASSES['pig'] in labels else 0.0,
            1.0 if ANIMAL_CLASSES['coyote'] in labels else 0.0,
        ]
        return torch.tensor(labels)

    def _image_from_row(self, row: pl.DataFrame) -> Image:
        image_bytes = row[1]['bytes']
        image = Image.open(BytesIO(image_bytes))
        return image


def fetch_if_missing(data_dir: str | Path = Path("data")) -> dict[str, Path]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        data_dir.mkdir()

    parquet_paths = {}
    for key, value in DATASET_LOCATIONS.items():
        f = Path(value)
        if not f.exists():
            # download the parquet from hugging face if we don't have it locally
            print("Fetching data from hugging face...")
            df = pl.read_parquet(BASE_URL + value)
            df.write_parquet(f)
        parquet_paths[key] = f

    return parquet_paths
