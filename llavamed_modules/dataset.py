"""
Dataset module for loading histopathology images with labels and captions.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Optional
from PIL import Image
from torch.utils.data import Dataset


class HistopathologyDataset(Dataset):
    """Dataset class for loading histopathology images with labels and captions."""

    def __init__(self, csv_path: str, images_dir: str, limit: Optional[int] = None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

        self.df = pd.read_csv(csv_path)

        # Filter to only samples with images
        if 'has_image' in self.df.columns:
            self.df = self.df[self.df['has_image'] == True]

        if limit is not None:
            self.df = self.df.head(limit)

        self.images_dir = Path(images_dir)

        if len(self.df) == 0:
            raise ValueError(f"No valid samples found in {csv_path}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        # Support both 'scan_id' and 'id' column names
        scan_id = row.get('scan_id', row.get('id', None))
        if scan_id is None:
            raise ValueError(f"Row {idx} missing 'scan_id' or 'id' column")

        # Try to find image file
        # Images are named like: {scan_id}_{suffix}.jpg
        # We match by the part before the underscore
        image_path = None
        
        # First, try exact match with extensions
        for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG']:
            potential_path = self.images_dir / f"{scan_id}{ext}"
            if potential_path.exists():
                image_path = potential_path
                break
        
        # If not found, try pattern matching (for formats like {scan_id}_{suffix}.jpg)
        # This matches images where scan_id is the part before the underscore
        if image_path is None:
            if self.images_dir.exists():
                for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG']:
                    # Match files like: scan_id_anything.ext
                    matching_files = list(self.images_dir.glob(f"{scan_id}_*{ext}"))
                    if not matching_files:
                        # Also try without underscore (just in case)
                        matching_files = list(self.images_dir.glob(f"{scan_id}*{ext}"))
                    if matching_files:
                        image_path = matching_files[0]  # Use first match
                        break

        if image_path is None and 'image_filename' in row and pd.notna(row['image_filename']):
            image_path = self.images_dir / row['image_filename']

        if image_path is None or not image_path.exists():
            raise FileNotFoundError(f"Image not found for scan_id: {scan_id}")

        img = Image.open(image_path).convert("RGB")

        ground_truth = row.get('text', row.get('caption', row.get('description', '')))
        return {
            'image': img,
            'scan_id': scan_id,
            'label': row.get('subtype', row.get('label', '')),
            'caption': ground_truth,  # Main code expects 'caption'
            'ground_truth_caption': ground_truth,  # Also include for compatibility
            'image_path': str(image_path)
        }

