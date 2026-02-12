#!/usr/bin/env python3
"""
Check which scan_ids from the CSV actually have images available.
"""

import pandas as pd
from pathlib import Path

DATASET_CSV = "/home/ubuntu/data/histopathology_dataset/dataset_with_labels.csv"
IMAGES_DIR = "/home/ubuntu/data/histopathology_dataset/images/patches_captions"

print("=" * 60)
print("Checking Available Images")
print("=" * 60)

# Load CSV
df = pd.read_csv(DATASET_CSV)
print(f"\n📊 CSV has {len(df)} scan_ids")

# Check images directory
images_dir = Path(IMAGES_DIR)
if not images_dir.exists():
    print(f"❌ Images directory does not exist: {IMAGES_DIR}")
    exit(1)

# Get all image files
image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
all_images = []
for ext in image_extensions:
    all_images.extend(images_dir.glob(f"*{ext}"))

print(f"🖼️  Found {len(all_images)} image files")

# Extract scan_ids from image filenames
image_scan_ids = set()
for img_path in all_images:
    # Extract scan_id (part before first underscore or before extension)
    name = img_path.stem
    if '_' in name:
        scan_id = name.split('_')[0]
    else:
        scan_id = name
    image_scan_ids.add(scan_id)

print(f"📋 Found {len(image_scan_ids)} unique scan_ids in images")

# Check which CSV scan_ids have images
csv_scan_ids = set(df['scan_id'].astype(str))
matching_scan_ids = csv_scan_ids.intersection(image_scan_ids)

print(f"\n✅ {len(matching_scan_ids)} scan_ids from CSV have images available")
print(f"❌ {len(csv_scan_ids) - len(matching_scan_ids)} scan_ids from CSV do NOT have images")

# Show first 20 matching scan_ids
if matching_scan_ids:
    print(f"\n📝 First 20 scan_ids with images (ready to process):")
    matching_list = sorted(list(matching_scan_ids))[:20]
    for i, scan_id in enumerate(matching_list, 1):
        # Count how many images for this scan_id
        num_images = sum(1 for img in all_images if img.stem.startswith(scan_id))
        print(f"   [{i:2d}] {scan_id} ({num_images} image(s))")
    
    # Create a filtered CSV with only scan_ids that have images
    filtered_df = df[df['scan_id'].isin(matching_list)]
    filtered_csv = "/home/ubuntu/data/histopathology_dataset/dataset_20_available.csv"
    filtered_df.to_csv(filtered_csv, index=False)
    print(f"\n✅ Created filtered CSV with 20 available scan_ids: {filtered_csv}")
else:
    print("\n⚠️  No matching scan_ids found!")
    print("   You may need to upload more images or check the image directory path.")

print("\n" + "=" * 60)


