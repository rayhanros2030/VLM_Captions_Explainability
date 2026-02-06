#!/usr/bin/env python3
"""
Process only scan_ids that actually have images available.
This filters the CSV to only include scan_ids with matching images.
"""

import sys
sys.path.insert(0, '/home/ubuntu/projects/llavamed-integration')

import pandas as pd
from pathlib import Path
import tempfile

from llavamed_modules.config import cfg

# Configuration
DATASET_CSV = '/home/ubuntu/data/histopathology_dataset/dataset_with_labels.csv'
IMAGES_DIR = '/home/ubuntu/data/histopathology_dataset/images/patches_captions'
OUTPUT_DIR = '/home/ubuntu/data/histopathology_outputs'
MAX_IMAGES = 20  # Maximum number of scan_ids to process

print("=" * 60)
print("Finding Available Images")
print("=" * 60)

# Load CSV
df = pd.read_csv(DATASET_CSV)
print(f"📊 CSV has {len(df)} scan_ids")

# Find images
images_dir = Path(IMAGES_DIR)
image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
all_images = []
for ext in image_extensions:
    all_images.extend(images_dir.glob(f"*{ext}"))

print(f"🖼️  Found {len(all_images)} image files")

# Extract scan_ids from image filenames
image_scan_ids = {}
for img_path in all_images:
    # Extract scan_id (part before first underscore or before extension)
    name = img_path.stem
    if '_' in name:
        scan_id = name.split('_')[0]
    else:
        scan_id = name
    
    if scan_id not in image_scan_ids:
        image_scan_ids[scan_id] = []
    image_scan_ids[scan_id].append(str(img_path))

print(f"📋 Found {len(image_scan_ids)} unique scan_ids in images")

# Filter CSV to only scan_ids with images
csv_scan_ids = set(df['scan_id'].astype(str))
matching_scan_ids = csv_scan_ids.intersection(set(image_scan_ids.keys()))

print(f"✅ {len(matching_scan_ids)} scan_ids from CSV have images")

if not matching_scan_ids:
    print("❌ No matching scan_ids found!")
    print("   You need to upload images for the scan_ids in your CSV.")
    exit(1)

# Filter dataframe
filtered_df = df[df['scan_id'].isin(matching_scan_ids)].copy()

# Limit to MAX_IMAGES
if len(filtered_df) > MAX_IMAGES:
    filtered_df = filtered_df.head(MAX_IMAGES)
    print(f"📝 Limiting to first {MAX_IMAGES} scan_ids with images")

print(f"\n📋 Will process {len(filtered_df)} scan_ids:")
for idx, row in filtered_df.iterrows():
    scan_id = row['scan_id']
    num_images = len(image_scan_ids[scan_id])
    print(f"   [{idx+1}] {scan_id} ({num_images} image(s))")

# Create temporary filtered CSV
temp_csv = Path(OUTPUT_DIR) / "temp_filtered_dataset.csv"
temp_csv.parent.mkdir(parents=True, exist_ok=True)
filtered_df.to_csv(temp_csv, index=False)
print(f"\n✅ Created filtered CSV: {temp_csv}")

# Now run processing
print("\n" + "=" * 60)
print("Starting Processing")
print("=" * 60)

# Configure for processing
cfg.DATASET_CSV = str(temp_csv)
cfg.IMAGES_DIR = IMAGES_DIR
cfg.NUM_IMAGES = len(filtered_df)  # Process all in filtered CSV
cfg.PROMPT_STYLE = "descriptive"
cfg.DEBUG_MODE = True
cfg.ENABLE_ROBUSTNESS_EXPERIMENTS = False
cfg.ENABLE_MULTI_SCALE = False
cfg.ENABLE_TOKEN_LEVEL_SALIENCY = False
cfg.ENABLE_WORD_LEVEL_SALIENCY = False

# Run the main processing
from llavamed_modules.main import main

try:
    main()
    print("\n✅ Processing complete!")
finally:
    # Clean up temp CSV (optional - comment out if you want to keep it)
    # temp_csv.unlink()
    pass


