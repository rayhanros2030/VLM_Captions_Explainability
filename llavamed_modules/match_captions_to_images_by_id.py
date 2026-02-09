#!/usr/bin/env python3
"""
Match captions to images based on ID (part before underscore in image filenames).
This handles the case where:
- Captions have unique IDs
- Images have the same ID before underscore, but different suffixes after (e.g., ID_001.jpg, ID_002.jpg)
"""

import os
import json
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict

# Configuration - adjust for your Lambda setup
CAPTIONS_DIR = "/home/ubuntu/data/histopathology_dataset/captions"  # Where caption files are
IMAGES_DIR = "/home/ubuntu/data/histopathology_dataset/images/patches_captions"  # Where images are
OUTPUT_DIR = "/home/ubuntu/data/histopathology_outputs"

def extract_id_from_filename(filename: str) -> str:
    """Extract ID from filename (part before first underscore or before extension)."""
    # Remove extension
    name = Path(filename).stem
    # Split by underscore and take first part
    if '_' in name:
        return name.split('_')[0]
    return name

def find_caption_files(captions_dir: Path) -> dict:
    """Find all caption files and extract their IDs."""
    caption_files = {}
    
    if not captions_dir.exists():
        print(f"⚠️  Captions directory does not exist: {captions_dir}")
        return caption_files
    
    # Look for .txt files
    for caption_file in captions_dir.glob("*.txt"):
        # Extract ID from filename
        # Could be: caption_ID.txt or just ID.txt
        filename = caption_file.stem
        if filename.startswith("caption_"):
            caption_id = filename.replace("caption_", "")
        else:
            caption_id = filename
        
        caption_files[caption_id] = caption_file
    
    return caption_files

def find_image_files(images_dir: Path) -> dict:
    """Find all image files and group by ID (part before underscore)."""
    image_files = defaultdict(list)
    
    if not images_dir.exists():
        print(f"⚠️  Images directory does not exist: {images_dir}")
        return image_files
    
    # Look for image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    for ext in image_extensions:
        for img_file in images_dir.glob(f"*{ext}"):
            # Extract ID (part before underscore)
            image_id = extract_id_from_filename(img_file.name)
            image_files[image_id].append(img_file)
    
    # Sort images for each ID
    for image_id in image_files:
        image_files[image_id].sort()
    
    return image_files

def load_caption(caption_path: Path) -> str:
    """Load caption text from file."""
    try:
        with open(caption_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        return f"Error loading caption: {e}"

def main():
    print("=" * 60)
    print("Caption-to-Image Matching by ID")
    print("=" * 60)
    
    captions_dir = Path(CAPTIONS_DIR)
    images_dir = Path(IMAGES_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all caption files
    print(f"\n📝 Finding caption files in: {captions_dir}")
    caption_files = find_caption_files(captions_dir)
    print(f"   ✅ Found {len(caption_files)} caption files")
    
    # Find all image files
    print(f"\n🖼️  Finding image files in: {images_dir}")
    image_files = find_image_files(images_dir)
    print(f"   ✅ Found {len(image_files)} unique image IDs")
    total_images = sum(len(imgs) for imgs in image_files.values())
    print(f"   ✅ Total: {total_images} image files")
    
    # Match captions to images
    print(f"\n🔍 Matching captions to images...")
    matches = {}
    unmatched_captions = []
    unmatched_images = []
    
    # Match by ID
    for caption_id, caption_path in caption_files.items():
        if caption_id in image_files:
            caption_text = load_caption(caption_path)
            matches[caption_id] = {
                'caption_id': caption_id,
                'caption_path': str(caption_path),
                'caption': caption_text,
                'image_paths': [str(img) for img in image_files[caption_id]],
                'num_images': len(image_files[caption_id]),
                'matched': True
            }
        else:
            unmatched_captions.append(caption_id)
    
    # Find images without captions
    matched_ids = set(matches.keys())
    for image_id, img_list in image_files.items():
        if image_id not in matched_ids:
            unmatched_images.append(image_id)
    
    print(f"\n📊 Matching Results:")
    print(f"   ✅ Matched: {len(matches)} caption IDs")
    print(f"   ❌ Unmatched captions: {len(unmatched_captions)}")
    print(f"   ❌ Unmatched image IDs: {len(unmatched_images)}")
    
    # Show some examples
    if matches:
        print(f"\n📋 Example matches:")
        for i, (caption_id, data) in enumerate(list(matches.items())[:5]):
            print(f"   [{i+1}] Caption ID: {caption_id}")
            print(f"       Images: {data['num_images']} image(s)")
            print(f"       Caption preview: {data['caption'][:60]}...")
            if data['num_images'] > 0:
                print(f"       First image: {Path(data['image_paths'][0]).name}")
    
    # Save mapping
    mapping_path = output_dir / "caption_image_mapping_by_id.json"
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(matches, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Mapping saved to: {mapping_path}")
    
    # Create visualization for first few matches
    if matches:
        print(f"\n📊 Creating visualization...")
        num_to_show = min(5, len(matches))
        items_to_show = list(matches.items())[:num_to_show]
        
        for caption_id, data in items_to_show:
            if data['num_images'] == 0:
                continue
            
            # Show first image with caption
            try:
                img_path = Path(data['image_paths'][0])
                img = Image.open(img_path)
                caption = data['caption']
                
                fig, axes = plt.subplots(1, 1, figsize=(10, 8))
                axes.imshow(img)
                axes.axis('off')
                
                # Add caption as title
                caption_preview = caption[:100] + "..." if len(caption) > 100 else caption
                title = f"Caption ID: {caption_id}\n{data['num_images']} image(s)\n\nCaption:\n{caption_preview}"
                axes.set_title(title, fontsize=9, pad=10, wrap=True)
                
                viz_path = output_dir / f"caption_image_match_{caption_id}.png"
                plt.tight_layout()
                plt.savefig(viz_path, dpi=150, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"   ⚠️  Could not create visualization for {caption_id}: {e}")
        
        print(f"✅ Visualizations saved to: {output_dir}")
    
    # Create summary report
    report_path = output_dir / "matching_report_by_id.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Caption-to-Image Matching Report (by ID)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total caption files: {len(caption_files)}\n")
        f.write(f"Total image IDs: {len(image_files)}\n")
        f.write(f"Total image files: {total_images}\n\n")
        f.write(f"✅ Matched: {len(matches)}\n")
        f.write(f"❌ Unmatched captions: {len(unmatched_captions)}\n")
        f.write(f"❌ Unmatched image IDs: {len(unmatched_images)}\n\n")
        
        if unmatched_captions:
            f.write("Unmatched Captions (first 10):\n")
            for cid in unmatched_captions[:10]:
                f.write(f"  - {cid}\n")
        
        if unmatched_images:
            f.write("\nUnmatched Image IDs (first 10):\n")
            for iid in unmatched_images[:10]:
                f.write(f"  - {iid} ({len(image_files[iid])} images)\n")
    
    print(f"✅ Report saved to: {report_path}")
    print(f"\n✅ Complete!")

if __name__ == "__main__":
    main()


