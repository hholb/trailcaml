import json
from pathlib import Path

import PIL
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.tv_tensors import Image


class TrailCameraDataset():
    def __init__(
        self,
        data_dir: Path = Path("data/trailcam-dataset/processed"),
        size = (64, 64),
    ):
        self.data_dir = data_dir
        self.size = size
        annotations_filename = "annotations.json"
        for split in ["train", "valid", "test"]:
            annotations_file = data_dir / f"{split}" / f"{annotations_filename}"
            with open(annotations_file, "r") as f:
                annotations = json.load(f)
                setattr(self, f"{split}_annotations", annotations)


    @property
    def train(self) -> Dataset:
        return _TrailCameraDataset("train", self.data_dir, self.train_annotations, self.size)

    @property
    def valid(self) -> Dataset:
        return _TrailCameraDataset("valid", self.data_dir, self.valid_annotations, self.size)

    @property
    def test(self) -> Dataset:
        return _TrailCameraDataset("test", self.data_dir, self.test_annotations, self.size)

    @property
    def splits(self) -> tuple[Dataset]:
        return self.train, self.valid, self.test

    def dataloader_splits(self, batch_size=36, num_workers: int | None = None):
        return [
            DataLoader(
                split,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                shuffle=(i == 0)
            )
            for i, split in enumerate(self.splits)
        ]


class _TrailCameraDataset(Dataset):
    def __init__(
        self,
        split: str,
        data_dir: Path,
        annotations: dict,
        size: tuple[int, int],
    ):
        super().__init__()
        self.split = split
        self.data_dir = data_dir
        self.annotations = annotations

        self.base_transforms = v2.Compose([
            v2.Resize(size, antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

        self.augmentations = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(10),
            v2.RandomAutocontrast(p=0.2),
        ]) if split == "train" else None


    def __len__(self):
        return len(self.annotations['images'])

    def __getitem__(self, index):
        img_meta = self.annotations['images'][index]
        image = self._load_image(img_meta)
        label = torch.tensor(self._get_label_for_img(img_meta['id']))

        image = self.base_transforms(image)
        if self.augmentations is not None:
            image = self.augmentations(image)

        return  image, label

    def _load_image(self, img_meta: dict) -> Image:
        image_file = self.data_dir / f"{self.split}" / f"{img_meta['file_name']}"
        pil_img = PIL.Image.open(Path(image_file))
        return Image(pil_img)

    def _get_label_for_img(self, id: int) -> list[float]:
        antns = [
            an for an in self.annotations['annotations'] if an['image_id'] == id
        ]
        animals = set(a['category_id'] for a in antns)
        categories = [
            c['id'] for c in self.annotations['categories']
        ]

        labels = [
            1.0 if c in animals else 0.0 for c in categories
        ]

        # say there is no animal if none of the other labels are present
        if not any(labels[1:]):
            labels[0] = 1.0

        return labels

