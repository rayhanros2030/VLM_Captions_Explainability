#!/usr/bin/env python3
"""
Match multiple image patches (all from same scan_id) with their caption.
This handles the case where you have multiple patches from one scan_id.
"""

import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

# Configuration
IMAGES_DIR = "/home/ubuntu/data/histopathology_dataset/images/patches_captions"
OUTPUT_DIR = "/home/ubuntu/data/histopathology_outputs"
# SCAN_ID will be auto-detected from available images
SCAN_ID = None  # Will be auto-detected

def main():
    print("=" * 60)
    print("Patch-to-Caption Matching Tool")
    print("=" * 60)
    
    images_dir = Path(IMAGES_DIR)
    output_dir = Path(OUTPUT_DIR)
    captions_dir = output_dir / "captions"
    
    # Auto-detect scan_id from images
    print(f"\n🔍 Finding images...")
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(images_dir.glob(f"*{ext}"))
    
    if not all_images:
        print("   ❌ No images found!")
        return
    
    # Extract scan_id from all images and find the most common one
    scan_ids = []
    for img_path in all_images:
        if '_' in img_path.stem:
            scan_id = img_path.stem.split('_')[0]
        else:
            scan_id = img_path.stem
        scan_ids.append(scan_id)
    
    # Find most common scan_id (should be the one with most images)
    scan_id_counts = Counter(scan_ids)
    scan_id = scan_id_counts.most_common(1)[0][0]
    num_images_for_scan_id = scan_id_counts[scan_id]
    
    print(f"   ✅ Found {len(all_images)} total images")
    print(f"   ✅ Auto-detected scan_id: {scan_id} (appears in {num_images_for_scan_id} images)")
    
    # Find all images for this scan_id
    image_files = []
    for ext in image_extensions:
        pattern_paths = list(images_dir.glob(f"{scan_id}*{ext}"))
        image_files.extend(pattern_paths)
    
    image_files = sorted(list(set(image_files)))
    
    print(f"   ✅ Found {len(image_files)} image patches")
    
    # Try to find caption - could be for scan_id or for individual patches
    caption = None
    caption_path = None
    
    # First try scan_id caption
    caption_path = captions_dir / f"caption_{scan_id}.txt"
    if caption_path.exists():
        with open(caption_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        print(f"   ✅ Found caption for scan_id")
        print(f"   Caption preview: {caption[:80]}...")
    else:
        # Try to find any caption file for patches
        caption_files = list(captions_dir.glob(f"caption_{scan_id}_*.txt"))
        if caption_files:
            # Use first caption as example (or combine them)
            caption_path = caption_files[0]
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
            print(f"   ✅ Found {len(caption_files)} caption file(s) for patches")
            print(f"   Using first caption as example: {caption_path.name}")
        else:
            print(f"   ⚠️  No caption found for scan_id: {scan_id}")
            print(f"   Looking for captions in: {captions_dir}")
            print(f"   Available caption files: {list(captions_dir.glob('caption_*.txt'))[:5]}")
            return
    
    # Create mapping
    mapping = {
        'scan_id': scan_id,
        'caption': caption,
        'caption_path': str(caption_path) if caption_path else None,
        'num_patches': len(image_files),
        'patches': []
    }
    
    print(f"\n📝 Creating mapping for {len(image_files)} patches...")
    for idx, img_path in enumerate(image_files):
        patch_info = {
            'patch_index': idx + 1,
            'image_path': str(img_path),
            'filename': img_path.name,
            'suffix': img_path.stem.replace(scan_id, '').lstrip('_')  # Extract suffix like 'f16118'
        }
        mapping['patches'].append(patch_info)
        print(f"   [{idx+1}/{len(image_files)}] {img_path.name}")
    
    # Save mapping
    mapping_path = output_dir / "patch_caption_mapping.json"
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Mapping saved to: {mapping_path}")
    
    # Create visualization
    print(f"\n📊 Creating visualization...")
    num_images = len(image_files)
    images_per_row = 5
    num_rows = (num_images + images_per_row - 1) // images_per_row
    
    fig, axes = plt.subplots(num_rows, images_per_row, figsize=(25, 5 * num_rows))
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_path in enumerate(image_files):
        row = idx // images_per_row
        col = idx % images_per_row
        ax = axes[row, col]
        
        try:
            img = Image.open(img_path)
            ax.imshow(img)
            ax.axis('off')
            
            # Show patch number and suffix
            suffix = img_path.stem.replace(scan_id, '').lstrip('_')
            ax.set_title(f"Patch {idx+1}\n{suffix}", fontsize=8, pad=2)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error\n{img_path.name}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=6)
            ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_images, num_rows * images_per_row):
        row = idx // images_per_row
        col = idx % images_per_row
        axes[row, col].axis('off')
    
    # Add caption at the bottom
    if caption:
        fig.suptitle(f"Scan ID: {scan_id}\nCaption: {caption[:100]}...", 
                     fontsize=10, y=0.02)
    else:
        fig.suptitle(f"Scan ID: {scan_id}\n(No caption found)", 
                     fontsize=10, y=0.02)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    viz_path = output_dir / "patch_caption_mapping.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization saved to: {viz_path}")
    
    # Create individual patch files with caption
    print(f"\n📄 Creating individual patch files with caption...")
    patches_dir = output_dir / "patches_with_captions"
    patches_dir.mkdir(exist_ok=True)
    
    for idx, img_path in enumerate(image_files):
        patch_num = idx + 1
        suffix = img_path.stem.replace(scan_id, '').lstrip('_')
        
        # Try to find caption for this specific patch
        patch_caption_path = captions_dir / f"caption_{img_path.stem}.txt"
        patch_caption = caption  # Default to scan_id caption
        if patch_caption_path.exists():
            with open(patch_caption_path, 'r', encoding='utf-8') as f:
                patch_caption = f.read().strip()
        
        # Create a text file with patch info and caption
        patch_info_path = patches_dir / f"patch_{patch_num:02d}_{suffix}.txt"
        with open(patch_info_path, 'w', encoding='utf-8') as f:
            f.write(f"Scan ID: {scan_id}\n")
            f.write(f"Patch Number: {patch_num}\n")
            f.write(f"Patch Suffix: {suffix}\n")
            f.write(f"Image File: {img_path.name}\n")
            f.write(f"\n{'='*60}\n")
            f.write(f"CAPTION:\n")
            f.write(f"{'='*60}\n\n")
            f.write(patch_caption if patch_caption else "No caption available")
        
        # Copy image to patches directory (optional - comment out if you don't want copies)
        # import shutil
        # shutil.copy(img_path, patches_dir / f"patch_{patch_num:02d}_{suffix}.jpg")
    
    print(f"✅ Patch info files saved to: {patches_dir}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Scan ID: {scan_id}")
    print(f"Number of patches: {len(image_files)}")
    if caption:
        print(f"Caption: {caption[:80]}...")
    else:
        print(f"Caption: Not found")
    print(f"\nFiles created:")
    print(f"  - {mapping_path}")
    print(f"  - {viz_path}")
    print(f"  - {patches_dir}/ (with {len(image_files)} info files)")
    print(f"\n✅ Complete!")

if __name__ == "__main__":
    main()

