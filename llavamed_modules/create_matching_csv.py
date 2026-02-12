#!/usr/bin/env python3
"""
Create a CSV that matches available images.
This script will:
1. Find all images in the images directory
2. Extract scan_ids from image filenames
3. Create a new CSV with only scan_ids that have images
"""

import pandas as pd
from pathlib import Path
import sys

# Configuration
IMAGES_DIR = "/home/ubuntu/data/histopathology_dataset/images/patches_captions"
ORIGINAL_CSV = "/home/ubuntu/data/histopathology_dataset/dataset_with_labels.csv"
OUTPUT_CSV = "/home/ubuntu/data/histopathology_dataset/dataset_matching_images.csv"

print("=" * 70)
print("Creating CSV that matches available images")
print("=" * 70)

# Check images directory
images_dir = Path(IMAGES_DIR)
if not images_dir.exists():
    print(f"❌ Images directory does not exist: {IMAGES_DIR}")
    print(f"   Please check the path and ensure images are uploaded.")
    sys.exit(1)

# Get all image files
print(f"\n📂 Scanning images in: {IMAGES_DIR}")
image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
all_images = []
for ext in image_extensions:
    all_images.extend(images_dir.glob(f"*{ext}"))

if not all_images:
    print(f"❌ No images found in {IMAGES_DIR}")
    print(f"   Checked extensions: {image_extensions}")
    sys.exit(1)

print(f"✅ Found {len(all_images)} image files")

# Extract scan_ids from image filenames
print("\n🔍 Extracting scan_ids from image filenames...")
image_scan_ids = {}
for img_path in all_images:
    # Extract scan_id (part before first underscore)
    name = img_path.stem
    if '_' in name:
        scan_id = name.split('_')[0]
    else:
        scan_id = name
    
    if scan_id not in image_scan_ids:
        image_scan_ids[scan_id] = []
    image_scan_ids[scan_id].append(img_path.name)

print(f"✅ Found {len(image_scan_ids)} unique scan_ids in images")

# Show sample scan_ids
print(f"\n📋 Sample scan_ids found in images (first 10):")
for i, (scan_id, files) in enumerate(sorted(image_scan_ids.items())[:10], 1):
    print(f"   [{i:2d}] {scan_id} ({len(files)} image(s))")

# Load original CSV if it exists
if Path(ORIGINAL_CSV).exists():
    print(f"\n📊 Loading original CSV: {ORIGINAL_CSV}")
    df = pd.read_csv(ORIGINAL_CSV)
    print(f"   Original CSV has {len(df)} rows")
    
    # Try to match scan_ids
    csv_scan_ids = set(df['scan_id'].astype(str))
    matching_scan_ids = csv_scan_ids.intersection(set(image_scan_ids.keys()))
    
    print(f"   ✅ {len(matching_scan_ids)} scan_ids from CSV match available images")
    
    if matching_scan_ids:
        # Create filtered CSV with matching scan_ids
        filtered_df = df[df['scan_id'].isin(matching_scan_ids)].copy()
        filtered_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ Created CSV with {len(filtered_df)} matching rows: {OUTPUT_CSV}")
        print(f"   Use this CSV: {OUTPUT_CSV}")
    else:
        print(f"\n⚠️  No scan_ids from CSV match available images!")
        print(f"   Creating new CSV from available images...")
        
        # Create new CSV from available images
        rows = []
        for scan_id, files in sorted(image_scan_ids.items()):
            # Use first image as reference
            first_file = files[0]
            rows.append({
                'scan_id': scan_id,
                'label': 'Unknown',  # Will need to be filled manually or from another source
                'subtype': 'Unknown',
                'text': f'Image patch: {first_file}',
                'caption': f'Image patch: {first_file}',
                'has_image': True
            })
        
        new_df = pd.DataFrame(rows)
        new_df.to_csv(OUTPUT_CSV, index=False)
        print(f"✅ Created new CSV with {len(new_df)} rows from available images: {OUTPUT_CSV}")
        print(f"   Note: Labels are set to 'Unknown' - you may need to update them manually")
else:
    print(f"\n⚠️  Original CSV not found: {ORIGINAL_CSV}")
    print(f"   Creating new CSV from available images...")
    
    # Create new CSV from available images
    rows = []
    for scan_id, files in sorted(image_scan_ids.items()):
        rows.append({
            'scan_id': scan_id,
            'label': 'Unknown',
            'subtype': 'Unknown',
            'text': f'Image patch: {files[0]}',
            'caption': f'Image patch: {files[0]}',
            'has_image': True
        })
    
    new_df = pd.DataFrame(rows)
    new_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Created new CSV with {len(new_df)} rows: {OUTPUT_CSV}")

print("\n" + "=" * 70)
print("Next steps:")
print(f"1. Review the CSV: {OUTPUT_CSV}")
print(f"2. Update your config to use: cfg.DATASET_CSV = '{OUTPUT_CSV}'")
print("=" * 70)

