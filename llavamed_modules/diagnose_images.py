#!/usr/bin/env python3
"""
Diagnostic script to find where images are and what scan_ids they have.
"""

import os
from pathlib import Path
import pandas as pd

# Check these paths
IMAGES_DIR = "/home/ubuntu/data/histopathology_dataset/images/patches_captions"
DATASET_CSV = "/home/ubuntu/data/histopathology_dataset/dataset_with_labels.csv"
OUTPUT_DIR = "/home/ubuntu/data/histopathology_outputs"

print("=" * 60)
print("Image Location Diagnostic Tool")
print("=" * 60)

# 1. Check if images directory exists
print(f"\n1. Checking images directory: {IMAGES_DIR}")
images_dir = Path(IMAGES_DIR)
if images_dir.exists():
    print(f"   ✅ Directory exists")
    
    # List all files
    all_files = list(images_dir.glob("*"))
    print(f"   📁 Found {len(all_files)} items")
    
    # Filter to image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = [f for f in all_files if f.suffix in image_extensions]
    print(f"   🖼️  Found {len(image_files)} image files")
    
    if image_files:
        print(f"\n   First 10 image files:")
        for i, img_file in enumerate(image_files[:10]):
            print(f"      [{i+1}] {img_file.name}")
        
        # Extract scan_ids from filenames
        print(f"\n   Analyzing filenames...")
        scan_ids_found = set()
        for img_file in image_files:
            # Try to extract scan_id (first part before underscore or extension)
            name = img_file.stem  # filename without extension
            # Common patterns: scan_id_suffix.jpg or scan_id.jpg
            if '_' in name:
                scan_id = name.split('_')[0]
            else:
                scan_id = name
            scan_ids_found.add(scan_id)
        
        print(f"   📊 Found {len(scan_ids_found)} unique scan_id(s) in image filenames")
        print(f"   First 5 scan_ids from images:")
        for i, sid in enumerate(list(scan_ids_found)[:5]):
            print(f"      [{i+1}] {sid}")
else:
    print(f"   ❌ Directory does NOT exist!")
    print(f"   Checking parent directory...")
    parent = images_dir.parent
    if parent.exists():
        print(f"   ✅ Parent exists: {parent}")
        print(f"   Contents:")
        for item in list(parent.iterdir())[:10]:
            print(f"      - {item.name} ({'dir' if item.is_dir() else 'file'})")

# 2. Check CSV
print(f"\n2. Checking CSV: {DATASET_CSV}")
csv_path = Path(DATASET_CSV)
if csv_path.exists():
    print(f"   ✅ CSV exists")
    df = pd.read_csv(csv_path)
    print(f"   📊 CSV has {len(df)} rows")
    print(f"   Columns: {list(df.columns)}")
    
    if 'scan_id' in df.columns:
        print(f"\n   First 10 scan_ids from CSV:")
        for i, scan_id in enumerate(df['scan_id'].head(10)):
            print(f"      [{i+1}] {scan_id}")
        
        # Check if any CSV scan_ids match image scan_ids
        if image_files:
            csv_scan_ids = set(df['scan_id'].astype(str))
            print(f"\n   🔍 Comparing CSV scan_ids with image scan_ids...")
            matching = scan_ids_found.intersection(csv_scan_ids)
            print(f"   ✅ Found {len(matching)} matching scan_ids")
            if matching:
                print(f"   Matching scan_ids:")
                for sid in list(matching)[:5]:
                    print(f"      - {sid}")
            else:
                print(f"   ⚠️  NO matching scan_ids found!")
                print(f"   This means CSV scan_ids don't match image filenames")
                print(f"\n   CSV scan_ids (first 3): {list(csv_scan_ids)[:3]}")
                print(f"   Image scan_ids (first 3): {list(scan_ids_found)[:3]}")
else:
    print(f"   ❌ CSV does NOT exist!")

# 3. Check output directory for captions
print(f"\n3. Checking output directory: {OUTPUT_DIR}")
output_path = Path(OUTPUT_DIR)
if output_path.exists():
    print(f"   ✅ Output directory exists")
    
    captions_dir = output_path / "captions"
    if captions_dir.exists():
        caption_files = list(captions_dir.glob("caption_*.txt"))
        print(f"   📝 Found {len(caption_files)} caption files")
        
        if caption_files:
            print(f"\n   First 5 caption files:")
            for i, cap_file in enumerate(caption_files[:5]):
                scan_id_from_file = cap_file.stem.replace("caption_", "")
                print(f"      [{i+1}] {cap_file.name} (scan_id: {scan_id_from_file[:20]}...)")
    else:
        print(f"   ❌ Captions directory does NOT exist: {captions_dir}")
else:
    print(f"   ❌ Output directory does NOT exist!")

# 4. Try to find images in other common locations
print(f"\n4. Searching for images in common locations...")
search_paths = [
    "/home/ubuntu/data/histopathology_dataset/images",
    "/home/ubuntu/data/histopathology_dataset",
    "/home/ubuntu/data",
]

for search_path in search_paths:
    search_dir = Path(search_path)
    if search_dir.exists():
        # Look for image files recursively
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        found_images = []
        for ext in image_extensions:
            found_images.extend(search_dir.rglob(ext))
        
        if found_images:
            print(f"   ✅ Found {len(found_images)} images in: {search_path}")
            print(f"   First 3 images:")
            for img in found_images[:3]:
                print(f"      - {img}")
            break

print("\n" + "=" * 60)
print("Diagnostic complete!")
print("=" * 60)


