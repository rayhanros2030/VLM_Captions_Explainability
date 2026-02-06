#!/usr/bin/env python3
"""
Process all image patches separately.
Each patch gets its own caption, even if they share the same scan_id.
"""

import os
# Disable accelerate/bitsandbytes auto-import (must be before any transformers imports)
os.environ["TRANSFORMERS_NO_ACCELERATE"] = "1"

import sys
sys.path.insert(0, '/home/ubuntu/projects/llavamed-integration')

import pandas as pd
from pathlib import Path
from PIL import Image
import json
import numpy as np

from llavamed_modules.config import cfg, set_seed
from llavamed_modules.model_loader import ensure_llava_registered, load_llava_med_model
from llavamed_modules.gradcam import GradCAM
from llavamed_modules.prompts import get_prompt
from llavamed_modules.evaluation import evaluate_caption_accuracy, post_process_caption

def extract_id_from_filename(filename: str) -> str:
    """Extract ID from filename (part before first underscore)."""
    name = Path(filename).stem
    if '_' in name:
        return name.split('_')[0]
    return name

def main():
    print("=" * 60)
    print("Processing All Image Patches")
    print("=" * 60)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Configuration
    IMAGES_DIR = "/home/ubuntu/data/histopathology_dataset/images/patches_captions"
    DATASET_CSV = "/home/ubuntu/data/histopathology_dataset/dataset_matched.csv"
    OUTPUT_DIR = "/home/ubuntu/data/histopathology_outputs"
    
    images_dir = Path(IMAGES_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    print(f"\n📂 Finding images in: {IMAGES_DIR}")
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    all_images = []
    for ext in image_extensions:
        all_images.extend(images_dir.glob(f"*{ext}"))
    
    all_images = sorted(all_images)
    print(f"   ✅ Found {len(all_images)} image files")
    
    # Load CSV to get labels/captions for the scan_id
    print(f"\n📂 Loading CSV: {DATASET_CSV}")
    df = pd.read_csv(DATASET_CSV)
    
    # Get the scan_id from first image
    if not all_images:
        print("   ❌ No images found!")
        return
    
    first_image_id = extract_id_from_filename(all_images[0].name)
    print(f"   ✅ Image ID: {first_image_id}")
    
    # Find matching CSV row
    csv_row = df[df['scan_id'] == first_image_id]
    if csv_row.empty:
        print(f"   ⚠️  No CSV entry found for {first_image_id}")
        label = ""
        ground_truth = ""
    else:
        label = csv_row.iloc[0].get('label', '')
        ground_truth = csv_row.iloc[0].get('caption', csv_row.iloc[0].get('text', ''))
        print(f"   ✅ Found CSV entry with label: {label}")
    
    # Set up config
    cfg.DATASET_CSV = DATASET_CSV
    cfg.IMAGES_DIR = str(images_dir)
    cfg.PROMPT_STYLE = "descriptive"
    cfg.ENABLE_POST_PROCESSING = True  # Enable post-processing
    cfg.ENABLE_ROBUSTNESS_EXPERIMENTS = False
    cfg.ENABLE_MULTI_SCALE = False
    cfg.ENABLE_TOKEN_LEVEL_SALIENCY = False
    cfg.ENABLE_WORD_LEVEL_SALIENCY = False
    
    # Create output directories
    captions_dir = output_dir / "captions"
    saliency_dir = output_dir / "saliency"
    captions_dir.mkdir(exist_ok=True)
    saliency_dir.mkdir(exist_ok=True)
    
    # Load model (same way as main.py)
    print(f"\n📦 Loading LLaVA-Med model...")
    ensure_llava_registered(cfg.LLAVA_MED_ROOT, cfg.MODEL_ID)
    model, tokenizer = load_llava_med_model(cfg.MODEL_ID)
    print("   ✅ Model loaded")
    
    # Initialize GradCAM (it will get image processor internally)
    print(f"\n🔧 Initializing GradCAM...")
    gradcam = GradCAM(model, tokenizer)
    print("   ✅ GradCAM initialized")
    
    # Get prompt
    prompt = get_prompt(cfg.PROMPT_STYLE, use_few_shot=cfg.USE_FEW_SHOT_EXAMPLES)
    
    # Process each image
    print(f"\n🚀 Processing {len(all_images)} image patches...")
    all_results = []
    
    for idx, image_path in enumerate(all_images):
        patch_id = image_path.stem  # Full filename without extension (e.g., scan_id_suffix)
        scan_id = extract_id_from_filename(image_path.name)
        
        print(f"\n[{idx+1}/{len(all_images)}] Processing: {image_path.name}")
        print(f"   Patch ID: {patch_id}")
        print(f"   Scan ID: {scan_id}")
        
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Generate caption and saliency
            saliency_map, caption = gradcam.generate_cam(image, prompt)
            
            # Post-process caption if enabled
            if cfg.ENABLE_POST_PROCESSING:
                caption_original = caption
                caption = post_process_caption(caption, label)
                if caption != caption_original:
                    print(f"   📝 Post-processed caption")
            
            # Normalize saliency map
            if saliency_map.max() > 0:
                saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-10)
            
            # Save caption (use patch_id to make it unique)
            caption_path = captions_dir / f"caption_{patch_id}.txt"
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(caption)
            
            # Save saliency map
            import matplotlib.pyplot as plt
            saliency_path = saliency_dir / f"saliency_{patch_id}.png"
            plt.figure(figsize=(8, 8))
            plt.imshow(saliency_map, cmap='jet')
            plt.axis('off')
            plt.colorbar()
            plt.title(f"Saliency: {caption[:50]}...")
            plt.savefig(saliency_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Evaluate if ground truth available
            eval_result = None
            if ground_truth:
                try:
                    eval_result = evaluate_caption_accuracy(caption, ground_truth, label)
                except Exception as e:
                    print(f"   ⚠️  Evaluation failed: {e}")
            
            # Store result
            result = {
                'patch_id': patch_id,
                'scan_id': scan_id,
                'image_path': str(image_path),
                'image_filename': image_path.name,
                'caption': caption,
                'label': label,
                'ground_truth': ground_truth,
            }
            
            if eval_result:
                result.update(eval_result)
            
            all_results.append(result)
            
            print(f"   ✅ Caption: {caption[:60]}...")
            if eval_result:
                print(f"   📊 BLEU: {eval_result.get('bleu_score', 0):.4f}")
            
        except Exception as e:
            print(f"   ❌ Error processing {image_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save all results
    print(f"\n💾 Saving results...")
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    results_path = output_dir / "all_patches_results.json"
    serializable_results = convert_to_serializable(all_results)
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Results saved to: {results_path}")
    
    # Create summary
    if all_results:
        eval_df = pd.DataFrame(all_results)
        summary = {
            'total_patches': len(all_results),
            'unique_scan_ids': len(set(r['scan_id'] for r in all_results)),
        }
        
        if 'bleu_score' in eval_df.columns:
            summary['average_bleu'] = eval_df['bleu_score'].mean()
        if 'semantic_similarity' in eval_df.columns:
            summary['average_semantic_similarity'] = eval_df['semantic_similarity'].mean()
        
        summary_path = output_dir / "patches_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"   ✅ Summary saved to: {summary_path}")
    
    print(f"\n✅ Processing complete!")
    print(f"   Processed: {len(all_results)}/{len(all_images)} patches")
    print(f"   Results: {results_path}")

if __name__ == "__main__":
    main()

