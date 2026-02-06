#!/usr/bin/env python3
"""
Verify that captions match images by ID (part before underscore).
Creates a filtered CSV with only matching entries.
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict

# Configuration
IMAGES_DIR = "/home/ubuntu/data/histopathology_dataset/images/patches_captions"
DATASET_CSV = "/home/ubuntu/data/histopathology_dataset/dataset_with_labels.csv"
OUTPUT_CSV = "/home/ubuntu/data/histopathology_dataset/dataset_matched.csv"

def extract_id_from_filename(filename: str) -> str:
    """Extract ID from filename (part before first underscore)."""
    name = Path(filename).stem  # Remove extension
    if '_' in name:
        return name.split('_')[0]
    return name

def main():
    print("=" * 60)
    print("Verifying Caption-Image Matching")
    print("=" * 60)
    
    # 1. Find all images and extract IDs
    print(f"\n1. Scanning images in: {IMAGES_DIR}")
    images_dir = Path(IMAGES_DIR)
    
    if not images_dir.exists():
        print(f"   ❌ Images directory does not exist!")
        return
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f"*{ext}"))
    
    print(f"   ✅ Found {len(image_files)} image files")
    
    # Group images by ID (before underscore)
    images_by_id = defaultdict(list)
    for img_file in image_files:
        image_id = extract_id_from_filename(img_file.name)
        images_by_id[image_id].append(img_file)
    
    print(f"   ✅ Found {len(images_by_id)} unique image IDs")
    
    # Show some examples
    print(f"\n   First 5 image IDs:")
    for i, (img_id, img_list) in enumerate(list(images_by_id.items())[:5]):
        print(f"      [{i+1}] {img_id} - {len(img_list)} image(s)")
        print(f"          Example: {img_list[0].name}")
    
    # 2. Load CSV and check scan_ids
    print(f"\n2. Loading CSV: {DATASET_CSV}")
    csv_path = Path(DATASET_CSV)
    
    if not csv_path.exists():
        print(f"   ❌ CSV file does not exist!")
        return
    
    df = pd.read_csv(csv_path)
    print(f"   ✅ CSV has {len(df)} rows")
    print(f"   Columns: {list(df.columns)}")
    
    # Get scan_ids from CSV
    if 'scan_id' not in df.columns:
        print(f"   ❌ CSV does not have 'scan_id' column!")
        print(f"   Available columns: {list(df.columns)}")
        return
    
    csv_scan_ids = set(df['scan_id'].astype(str))
    print(f"   ✅ CSV has {len(csv_scan_ids)} unique scan_ids")
    
    # 3. Match images to CSV scan_ids
    print(f"\n3. Matching images to CSV scan_ids...")
    
    matching_ids = []
    for scan_id in csv_scan_ids:
        if scan_id in images_by_id:
            matching_ids.append(scan_id)
    
    print(f"   ✅ Found {len(matching_ids)} matching IDs")
    print(f"   ❌ {len(csv_scan_ids) - len(matching_ids)} CSV scan_ids have no images")
    print(f"   ❌ {len(images_by_id) - len(matching_ids)} image IDs have no CSV entry")
    
    # Show matching examples
    if matching_ids:
        print(f"\n   First 5 matching IDs:")
        for i, match_id in enumerate(matching_ids[:5]):
            num_images = len(images_by_id[match_id])
            print(f"      [{i+1}] {match_id} - {num_images} image(s)")
    
    # 4. Create filtered CSV with only matching entries
    print(f"\n4. Creating filtered CSV...")
    
    filtered_df = df[df['scan_id'].isin(matching_ids)].copy()
    
    # Add image count for each row
    filtered_df['num_images'] = filtered_df['scan_id'].apply(
        lambda x: len(images_by_id.get(str(x), []))
    )
    
    # Save filtered CSV
    output_path = Path(OUTPUT_CSV)
    filtered_df.to_csv(output_path, index=False)
    
    print(f"   ✅ Created filtered CSV: {output_path}")
    print(f"   ✅ Filtered CSV has {len(filtered_df)} rows (down from {len(df)})")
    
    # 5. Summary
    print(f"\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total images found: {len(image_files)}")
    print(f"Unique image IDs: {len(images_by_id)}")
    print(f"CSV scan_ids: {len(csv_scan_ids)}")
    print(f"Matching IDs: {len(matching_ids)}")
    print(f"\nFiltered CSV saved to: {output_path}")
    print(f"   Use this CSV for processing: {output_path}")
    
    # Show unmatched examples
    unmatched_csv = csv_scan_ids - set(matching_ids)
    if unmatched_csv:
        print(f"\n⚠️  CSV scan_ids without images (first 5):")
        for sid in list(unmatched_csv)[:5]:
            print(f"   - {sid}")
    
    unmatched_images = set(images_by_id.keys()) - set(matching_ids)
    if unmatched_images:
        print(f"\n⚠️  Image IDs without CSV entries (first 5):")
        for img_id in list(unmatched_images)[:5]:
            print(f"   - {img_id} ({len(images_by_id[img_id])} images)")

if __name__ == "__main__":
    main()


