import shutil
from datetime import datetime
from pathlib import Path
import json
from typing import Optional, List, Dict
import hashlib

import PIL
from PIL import Image
from rich.console import Console
from rich.progress import track

console = Console()

class TrailCamDataManager:
    """Manages the ingestion and organization of new trail camera images."""
    
    def __init__(
        self,
        base_dir: Path,
        splits: Dict[str, float] = {"train": 0.7, "valid": 0.15, "test": 0.15}
    ):
        """
        Args:
            base_dir: Root directory for the dataset
            splits: Dictionary defining train/valid/test split ratios
        """
        self.base_dir = Path(base_dir)
        self.splits = splits
        self.categories = [ 
            {"id": 0, "name": "No Animal"},
            {"id": 1, "name": "Coyote"},
            {"id": 2, "name": "Deer"},
            {"id": 3, "name": "Hog"},
            {"id": 4, "name": "Rabbit"},
            {"id": 5, "name": "Raccoon"},
            {"id": 6, "name": "Squirrel"},
        ]
        
        # Create necessary directories
        self.raw_dir = self.base_dir / "raw_images"
        self.processed_dir = self.base_dir / "processed"
        for split in splits:
            (self.processed_dir / split).mkdir(parents=True, exist_ok=True)
            
        # Initialize or load annotations
        self.annotations_file = self.base_dir / "annotations.json"
        self.load_annotations()
        self.annotations['categories'] = self.categories
        
    def load_annotations(self):
        """Load existing annotations or create new ones."""
        if self.annotations_file.exists():
            with open(self.annotations_file, "r") as f:
                self.annotations = json.load(f)
        else:
            self.annotations = {
                "info": {
                    "description": "Trail Camera Dataset",
                    "date_created": datetime.now().isoformat()
                },
                "images": [],
                "annotations": []
            }
    
    def generate_image_id(self, image_path: Path) -> str:
        """Generate a unique ID for an image based on its content."""
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def ingest_images(self, source_dir: Path, batch_name: Optional[str] = None):
        """
        Ingest new images from a source directory.
        
        Args:
            source_dir: Directory containing new images
            batch_name: Optional name for this batch of images
        """
        source_dir = Path(source_dir)
        if not batch_name:
            batch_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        batch_dir = self.raw_dir / batch_name
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy and validate images
        image_files = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.JPG"))
        for img_path in track(image_files, description="Ingesting images"):
            try:
                # Validate image
                with Image.open(img_path) as img:
                    img.verify()
                
                # Copy to raw directory
                new_path = batch_dir / img_path.name
                shutil.copy2(img_path, new_path)
                
            except (PIL.UnidentifiedImageError, OSError) as e:
                console.print(f"[red]Error processing {img_path}: {e}")
                continue
        
        console.print(f"[green]Successfully ingested {len(image_files)} images to {batch_dir}")
        return batch_dir
    
    def create_labeling_batch(self, batch_size: int = 50) -> List[Path]:
        """
        Create a batch of unlabeled images for labeling.
        
        Args:
            batch_size: Number of images to include in the batch
            
        Returns:
            List of paths to images needing labels
        """
        # Find all images without annotations
        labeled_ids = {ann["image_id"] for ann in self.annotations["annotations"]}
        unlabeled = []
        
        for img_dir in self.raw_dir.iterdir():
            if img_dir.is_dir():
                for img_path in img_dir.glob("*.JPG"): 
                    img_id = self.generate_image_id(img_path)
                    if img_id not in labeled_ids:
                        unlabeled.append(img_path)
        return unlabeled[:batch_size]
    
    def add_labels(self, image_labels: Dict[Path, List[str]]):
        """
        Add new labels to the dataset.
        
        Args:
            image_labels: Dictionary mapping image paths to lists of category names
        """
        category_map = {cat["name"]: cat["id"] for cat in self.annotations["categories"]}
        
        for img_path, labels in track(image_labels.items(), description="Adding labels"):
            img_id = self.generate_image_id(img_path)
            
            # Add image to annotations if not present
            if not any(img["image_id"] == img_id for img in self.annotations["images"]):
                with Image.open(img_path) as img:
                    width, height = img.size
                    
                self.annotations["images"].append({
                    "image_id": img_id,
                    "file_name": img_path.name,
                    "width": width,
                    "height": height,
                    "date_captured": datetime.fromtimestamp(
                        img_path.stat().st_mtime
                    ).isoformat()
                })
            
            # Add label annotations
            for label in labels:
                if label in category_map:
                    self.annotations["annotations"].append({
                        "image_id": img_id,
                        "category_id": category_map[label]
                    })
                else:
                    console.print(f"[yellow]Warning: Unknown category {label}")
        
        # Save updated annotations
        with open(self.annotations_file, "w") as f:
            json.dump(self.annotations, f, indent=2)
            
    def distribute_to_splits(self):
        """Distribute labeled images to train/valid/test splits."""
        # Get all labeled images
        labeled_images = {
            ann["image_id"] for ann in self.annotations["annotations"]
        }
        
        # Randomly assign to splits
        import random
        split_assignments = {}
        splits_list = list(self.splits.items())
        
        for img_id in labeled_images:
            r = random.random()
            cumsum = 0
            for split, ratio in splits_list:
                cumsum += ratio
                if r <= cumsum:
                    split_assignments[img_id] = split
                    break
        
        # Move images to appropriate split directories
        for img in self.annotations["images"]:
            if img["id"] in split_assignments:
                split = split_assignments[img["id"]]
                source = next(self.raw_dir.rglob(img["file_name"]))
                dest = self.processed_dir / split / img["file_name"]
                
                if not dest.exists():
                    shutil.copy2(source, dest)
        
        console.print("[green]Successfully distributed images to splits")
