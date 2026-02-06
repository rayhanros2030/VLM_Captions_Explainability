"""
Script to match images with their generated captions.
Creates a mapping file and optionally generates a visualization.
"""

import os
import json
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

from .config import cfg


def find_image_file(images_dir: Path, scan_id: str) -> Optional[Path]:
    """Find image file matching scan_id."""
    # Try common extensions
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    # First try exact match with scan_id
    for ext in extensions:
        exact_path = images_dir / f"{scan_id}{ext}"
        if exact_path.exists():
            return exact_path
    
    # Try pattern matching: scan_id_* or scan_id*
    for ext in extensions:
        pattern_paths = list(images_dir.glob(f"{scan_id}*{ext}"))
        if pattern_paths:
            return pattern_paths[0]  # Return first match
    
    # Try nested directories
    for ext in extensions:
        pattern_paths = list(images_dir.rglob(f"{scan_id}*{ext}"))
        if pattern_paths:
            return pattern_paths[0]
    
    return None


def find_caption_file(output_dir: Path, scan_id: str) -> Optional[Path]:
    """Find caption file matching scan_id."""
    captions_dir = output_dir / "captions"
    caption_path = captions_dir / f"caption_{scan_id}.txt"
    
    if caption_path.exists():
        return caption_path
    
    return None


def load_caption(caption_path: Path) -> str:
    """Load caption from file."""
    try:
        with open(caption_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        return f"Error loading caption: {e}"


def create_image_caption_mapping(
    dataset_csv: str,
    images_dir: str,
    output_dir: str,
    max_images: Optional[int] = None
) -> Dict[str, Dict]:
    """
    Create mapping between images and captions.
    
    Returns:
        Dictionary mapping scan_id to {
            'image_path': Path,
            'caption_path': Path,
            'caption': str,
            'label': str,
            'found_image': bool,
            'found_caption': bool
        }
    """
    # Load dataset
    df = pd.read_csv(dataset_csv)
    
    if max_images:
        df = df.head(max_images)
    
    images_dir_path = Path(images_dir)
    output_dir_path = Path(output_dir)
    
    mapping = {}
    
    print(f"🔍 Matching {len(df)} images with captions...")
    
    for idx, row in df.iterrows():
        scan_id = str(row['scan_id'])
        label = str(row.get('label', ''))
        
        # Find image
        image_path = find_image_file(images_dir_path, scan_id)
        
        # Find caption
        caption_path = find_caption_file(output_dir_path, scan_id)
        caption = load_caption(caption_path) if caption_path else None
        
        mapping[scan_id] = {
            'image_path': str(image_path) if image_path else None,
            'caption_path': str(caption_path) if caption_path else None,
            'caption': caption,
            'label': label,
            'found_image': image_path is not None,
            'found_caption': caption_path is not None,
            'row_index': idx
        }
        
        status = []
        if image_path:
            status.append("✅ Image")
        else:
            status.append("❌ Image")
        
        if caption_path:
            status.append("✅ Caption")
        else:
            status.append("❌ Caption")
        
        print(f"  [{idx+1}/{len(df)}] {scan_id}: {' | '.join(status)}")
    
    return mapping


def save_mapping(mapping: Dict, output_path: Path):
    """Save mapping to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"✅ Mapping saved to: {output_path}")


def create_visualization(
    mapping: Dict,
    output_dir: Path,
    num_images: int = 20,
    images_per_row: int = 4
):
    """
    Create a visualization showing images with their captions.
    """
    # Filter to only images that have both image and caption
    valid_items = [
        (scan_id, data) for scan_id, data in mapping.items()
        if data['found_image'] and data['found_caption']
    ]
    
    if not valid_items:
        print("⚠️  No valid image-caption pairs found for visualization")
        return
    
    # Limit number of images
    valid_items = valid_items[:num_images]
    
    num_rows = (len(valid_items) + images_per_row - 1) // images_per_row
    
    fig, axes = plt.subplots(num_rows, images_per_row, figsize=(20, 5 * num_rows))
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (scan_id, data) in enumerate(valid_items):
        row = idx // images_per_row
        col = idx % images_per_row
        ax = axes[row, col]
        
        # Load and display image
        try:
            image_path = Path(data['image_path'])
            if image_path.exists():
                img = Image.open(image_path)
                ax.imshow(img)
                ax.axis('off')
                
                # Add caption (truncate if too long)
                caption = data['caption']
                if len(caption) > 100:
                    caption = caption[:100] + "..."
                
                # Add label if available
                label = data.get('label', '')
                title = f"{scan_id[:12]}...\n"
                if label:
                    title += f"Label: {label}\n"
                title += f"Caption: {caption}"
                
                ax.set_title(title, fontsize=8, pad=5)
            else:
                ax.text(0.5, 0.5, f"Image not found\n{scan_id}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading image\n{scan_id}\n{str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    # Hide unused subplots
    for idx in range(len(valid_items), num_rows * images_per_row):
        row = idx // images_per_row
        col = idx % images_per_row
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    viz_path = output_dir / "image_caption_mapping.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization saved to: {viz_path}")


def create_summary_report(mapping: Dict, output_path: Path):
    """Create a text summary report."""
    total = len(mapping)
    found_images = sum(1 for data in mapping.values() if data['found_image'])
    found_captions = sum(1 for data in mapping.values() if data['found_caption'])
    found_both = sum(1 for data in mapping.values() 
                    if data['found_image'] and data['found_caption'])
    
    report = f"""
Image-Caption Matching Report
==============================

Total scan_ids: {total}
✅ Images found: {found_images} ({found_images/total*100:.1f}%)
✅ Captions found: {found_captions} ({found_captions/total*100:.1f}%)
✅ Both found: {found_both} ({found_both/total*100:.1f}%)

Missing Images:
"""
    
    missing_images = [scan_id for scan_id, data in mapping.items() 
                     if not data['found_image']]
    if missing_images:
        for scan_id in missing_images[:10]:  # Show first 10
            report += f"  - {scan_id}\n"
        if len(missing_images) > 10:
            report += f"  ... and {len(missing_images) - 10} more\n"
    else:
        report += "  None\n"
    
    report += "\nMissing Captions:\n"
    missing_captions = [scan_id for scan_id, data in mapping.items() 
                       if not data['found_caption']]
    if missing_captions:
        for scan_id in missing_captions[:10]:  # Show first 10
            report += f"  - {scan_id}\n"
        if len(missing_captions) > 10:
            report += f"  ... and {len(missing_captions) - 10} more\n"
    else:
        report += "  None\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Summary report saved to: {output_path}")


def main():
    """Main function to match images with captions."""
    print("=" * 60)
    print("Image-Caption Matching Tool")
    print("=" * 60)
    
    # Create mapping
    mapping = create_image_caption_mapping(
        dataset_csv=cfg.DATASET_CSV,
        images_dir=cfg.IMAGES_DIR,
        output_dir=cfg.OUTPUT_DIR,
        max_images=cfg.NUM_IMAGES
    )
    
    output_dir = Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save mapping JSON
    mapping_path = output_dir / "image_caption_mapping.json"
    save_mapping(mapping, mapping_path)
    
    # Create visualization
    print("\n📊 Creating visualization...")
    create_visualization(mapping, output_dir, num_images=20)
    
    # Create summary report
    print("\n📝 Creating summary report...")
    report_path = output_dir / "matching_report.txt"
    create_summary_report(mapping, report_path)
    
    print("\n✅ Matching complete!")
    print(f"   - Mapping JSON: {mapping_path}")
    print(f"   - Visualization: {output_dir / 'image_caption_mapping.png'}")
    print(f"   - Summary report: {report_path}")


if __name__ == "__main__":
    main()


