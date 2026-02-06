#!/usr/bin/env python3
"""
Standalone script to match images with captions.
Can be run directly on Lambda without importing the full module.
"""

import os
import json
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Configuration - adjust these paths for your Lambda instance
DATASET_CSV = "/home/ubuntu/data/histopathology_dataset/dataset_with_labels.csv"
# Try multiple possible image directories
IMAGES_DIRS = [
    "/home/ubuntu/data/histopathology_dataset/images/patches_captions",
    "/home/ubuntu/data/histopathology_dataset/images",
    "/home/ubuntu/data/histopathology_dataset",
]
OUTPUT_DIR = "/home/ubuntu/data/histopathology_outputs"
NUM_IMAGES = 20  # Set to None to process all


def find_image_file(images_dirs, scan_id: str):
    """Find all image files matching scan_id across multiple directories. Returns list of paths."""
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    found_images = []
    
    # Try each directory
    for images_dir_str in images_dirs:
        images_dir = Path(images_dir_str)
        if not images_dir.exists():
            continue
        
        # Try exact match first
        for ext in extensions:
            exact_path = images_dir / f"{scan_id}{ext}"
            if exact_path.exists():
                found_images.append(exact_path)
        
        # Try pattern matching (finds all images starting with scan_id)
        for ext in extensions:
            pattern_paths = list(images_dir.glob(f"{scan_id}*{ext}"))
            found_images.extend(pattern_paths)
        
        # Try nested directories (recursive search)
        for ext in extensions:
            pattern_paths = list(images_dir.rglob(f"{scan_id}*{ext}"))
            found_images.extend(pattern_paths)
    
    # Remove duplicates and sort
    found_images = sorted(list(set(found_images)))
    
    return found_images if found_images else None


def main():
    print("=" * 60)
    print("Image-Caption Matching Tool")
    print("=" * 60)
    
    # Load dataset
    print(f"\n📂 Loading dataset from: {DATASET_CSV}")
    df = pd.read_csv(DATASET_CSV)
    
    if NUM_IMAGES:
        df = df.head(NUM_IMAGES)
        print(f"   Processing first {NUM_IMAGES} images")
    
    # Find which images directory actually exists
    images_dir_path = None
    for img_dir in IMAGES_DIRS:
        if Path(img_dir).exists():
            images_dir_path = Path(img_dir)
            print(f"   ✅ Using images directory: {img_dir}")
            break
    
    if not images_dir_path:
        print(f"   ⚠️  None of the image directories exist!")
        print(f"   Tried: {IMAGES_DIRS}")
        return
    
    output_dir_path = Path(OUTPUT_DIR)
    captions_dir = output_dir_path / "captions"
    
    # Create mapping
    mapping = {}
    print(f"\n🔍 Matching {len(df)} images with captions...\n")
    
    for idx, row in df.iterrows():
        scan_id = str(row['scan_id'])
        label = str(row.get('label', ''))
        
        # Find images (may be multiple for same scan_id)
        image_paths = find_image_file(IMAGES_DIRS, scan_id)
        
        # Find caption
        caption_path = captions_dir / f"caption_{scan_id}.txt"
        caption = None
        if caption_path.exists():
            try:
                with open(caption_path, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
            except Exception as e:
                caption = f"Error: {e}"
        
        # Store all image paths
        image_path_list = [str(p) for p in image_paths] if image_paths else []
        primary_image = image_paths[0] if image_paths and len(image_paths) > 0 else None
        has_images = image_paths is not None and len(image_paths) > 0
        
        mapping[scan_id] = {
            'image_path': str(primary_image) if primary_image else None,
            'image_paths': image_path_list,  # All images for this scan_id
            'num_images': len(image_paths) if image_paths else 0,
            'caption_path': str(caption_path) if caption_path.exists() else None,
            'caption': caption,
            'label': label,
            'found_image': has_images,
            'found_caption': caption_path.exists()
        }
        
        # Print status
        status_icon = "✅" if (has_images and caption_path.exists()) else "⚠️"
        print(f"{status_icon} [{idx+1}/{len(df)}] {scan_id[:20]}...")
        if image_paths:
            print(f"      Images: ✅ Found {len(image_paths)} image(s)")
            for i, img_path in enumerate(image_paths[:3]):  # Show first 3
                print(f"        [{i+1}] {Path(img_path).name}")
            if len(image_paths) > 3:
                print(f"        ... and {len(image_paths) - 3} more")
        else:
            print(f"      Images: ❌ Not found")
        
        if caption_path.exists():
            caption_preview = caption[:60] + "..." if caption and len(caption) > 60 else caption
            print(f"      Caption: ✅ {caption_preview}")
        else:
            print(f"      Caption: ❌ Not found")
        print()
    
    # Save mapping
    mapping_path = output_dir_path / "image_caption_mapping.json"
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"✅ Mapping saved to: {mapping_path}")
    
    # Statistics
    total = len(mapping)
    found_images = sum(1 for data in mapping.values() if data['found_image'])
    found_captions = sum(1 for data in mapping.values() if data['found_caption'])
    found_both = sum(1 for data in mapping.values() 
                    if data['found_image'] and data['found_caption'])
    
    print(f"\n📊 Statistics:")
    print(f"   Total: {total}")
    print(f"   Images found: {found_images} ({found_images/total*100:.1f}%)")
    print(f"   Captions found: {found_captions} ({found_captions/total*100:.1f}%)")
    print(f"   Both found: {found_both} ({found_both/total*100:.1f}%)")
    
    # Create visualization if we have matches
    if found_both > 0:
        print(f"\n📊 Creating visualization...")
        valid_items = [
            (scan_id, data) for scan_id, data in mapping.items()
            if data['found_image'] and data['found_caption']
        ]
        
        # For visualization, we'll show one image per scan_id (the first one)
        # But we can show multiple if there are multiple scan_ids
        max_show = 20
        valid_items = valid_items[:max_show]
        
        # Count total images to show (may be more if multiple images per scan_id)
        total_images_to_show = 0
        items_to_visualize = []
        for scan_id, data in valid_items:
            image_paths = data.get('image_paths', [])
            if image_paths:
                # Show first image for each scan_id
                items_to_visualize.append((scan_id, data, image_paths[0]))
                total_images_to_show += 1
                if total_images_to_show >= max_show:
                    break
        
        if items_to_visualize:
            num_images = len(items_to_visualize)
            images_per_row = 4
            num_rows = (num_images + images_per_row - 1) // images_per_row
            
            fig, axes = plt.subplots(num_rows, images_per_row, figsize=(20, 5 * num_rows))
            if num_rows == 1:
                axes = axes.reshape(1, -1) if num_images > 1 else [[axes]]
            
            for idx, (scan_id, data, img_path) in enumerate(items_to_visualize):
                row = idx // images_per_row
                col = idx % images_per_row
                ax = axes[row, col] if num_rows > 1 else axes[col]
                
                try:
                    img = Image.open(img_path)
                    ax.imshow(img)
                    ax.axis('off')
                    
                    caption = data['caption'] or "No caption"
                    if len(caption) > 80:
                        caption = caption[:80] + "..."
                    
                    label = data.get('label', '')
                    num_imgs = data.get('num_images', 0)
                    title = f"{scan_id[:15]}...\n"
                    if num_imgs > 1:
                        title += f"({num_imgs} images)\n"
                    if label:
                        title += f"Label: {label}\n"
                    title += f"{caption}"
                    
                    ax.set_title(title, fontsize=7, pad=3)
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error\n{scan_id[:15]}", 
                           ha='center', va='center', transform=ax.transAxes, fontsize=8)
                    ax.axis('off')
        
        # Hide unused subplots
        for idx in range(num_images, num_rows * images_per_row):
            row = idx // images_per_row
            col = idx % images_per_row
            if num_rows > 1:
                axes[row, col].axis('off')
            else:
                axes[col].axis('off')
        
        plt.tight_layout()
        viz_path = output_dir_path / "image_caption_mapping.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Visualization saved to: {viz_path}")
    
    print("\n✅ Complete!")


if __name__ == "__main__":
    main()

