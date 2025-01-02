import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from pathlib import Path
import json
from typing import Dict, List

from trailcam_data_manager import TrailCamDataManager

class LabelingTool:
    def __init__(self, images_to_label: List[Path], categories: List[str]):
        self.root = tk.Tk()
        self.root.title("Trail Camera Image Labeling Tool")
        
        self.images = images_to_label
        self.current_idx = 0
        self.categories = categories
        self.labels: Dict[Path, List[str]] = {}
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface."""
        # Image display
        self.image_label = ttk.Label(self.root)
        self.image_label.pack(pady=10)
        
        # Checkboxes for categories
        self.var_dict = {}
        checkbox_frame = ttk.Frame(self.root)
        checkbox_frame.pack(pady=10)
        
        for category in self.categories:
            var = tk.BooleanVar()
            self.var_dict[category] = var
            ttk.Checkbutton(
                checkbox_frame, 
                text=category,
                variable=var,
            ).pack(side=tk.LEFT, padx=5)
            
        # Navigation buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)
        
        ttk.Button(
            button_frame,
            text="Previous",
            command=self.prev_image
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Next",
            command=self.next_image
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Save",
            command=self.save_labels
        ).pack(side=tk.LEFT, padx=5)
        
        # Progress indicator
        self.progress_var = tk.StringVar()
        ttk.Label(
            self.root,
            textvariable=self.progress_var
        ).pack(pady=5)
        
        self.load_current_image()
        
    def load_current_image(self):
        """Load and display the current image."""
        if 0 <= self.current_idx < len(self.images):
            # Update progress
            self.progress_var.set(
                f"Image {self.current_idx + 1} of {len(self.images)}"
            )
            
            # Load image
            img_path = self.images[self.current_idx]
            img = Image.open(img_path)
            
            # Resize to fit screen while maintaining aspect ratio
            display_size = (800, 600)
            img.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage and display
            photo = ImageTk.PhotoImage(img)
            self.image_label.configure(image=photo)
            self.image_label.image = photo  # Keep a reference
            
            # Load existing labels if any
            self.reset_checkboxes()
            if img_path in self.labels:
                for category in self.labels[img_path]:
                    self.var_dict[category].set(True)
                    
    def reset_checkboxes(self):
        """Reset all checkboxes to unchecked."""
        for var in self.var_dict.values():
            var.set(False)
            
    def save_current_labels(self):
        """Save labels for the current image."""
        if 0 <= self.current_idx < len(self.images):
            img_path = self.images[self.current_idx]
            selected = [
                category for category, var in self.var_dict.items()
                if var.get()
            ]
            if selected:
                self.labels[img_path] = selected
            elif img_path in self.labels:
                # If nothing selected and image was previously labeled,
                # mark as "No Animal"
                self.labels[img_path] = ["No Animal"]
                
    def next_image(self):
        """Move to next image."""
        self.save_current_labels()
        if self.current_idx < len(self.images) - 1:
            self.current_idx += 1
            self.load_current_image()
            
    def prev_image(self):
        """Move to previous image."""
        self.save_current_labels()
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_current_image()
            
    def save_labels(self):
        """Save all labels and close the tool."""
        self.save_current_labels()
        self.root.quit()
        
    def run(self) -> Dict[Path, List[str]]:
        """Run the labeling tool and return collected labels."""
        self.root.mainloop()
        return self.labels

def main(args):
    # Example usage
    data_manager = TrailCamDataManager(Path("data/trailcam-dataset"))
    data_manager.ingest_images(source_dir=Path(args.file_name))
    images_to_label = data_manager.create_labeling_batch(batch_size=50)
    print(images_to_label)
    
    categories = [cat["name"] for cat in data_manager.annotations["categories"]]
    
    tool = LabelingTool(images_to_label, categories)
    labels = tool.run()
    
    if labels:
        data_manager.add_labels(labels)
        data_manager.distribute_to_splits()

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("file_name")

    args = parser.parse_args()
    main(args)
