"""
Main execution script for LLaVA-Med integration with histopathology dataset.
Orchestrates the full pipeline: model loading, caption generation, saliency mapping, and evaluation.
"""

import os
import sys
import random
import json
import re
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

# Disable accelerate/bitsandbytes auto-import
os.environ["TRANSFORMERS_NO_ACCELERATE"] = "1"

# Import all modules (config handles optional dependency checks)
from .config import cfg, set_seed, SEMANTIC_AVAILABLE, PLI_AVAILABLE
from .dataset import HistopathologyDataset
from .model_loader import ensure_llava_registered, load_llava_med_model
from .prompts import get_prompt
from .gradcam import GradCAM
from .pli import gradcam_to_pli, generate_raw_gradcam, generate_attention_saliency
from .saliency import (
    generate_token_level_saliency,
    generate_word_level_saliency_and_pli,
    compute_cross_token_consistency,
    compute_cross_token_grounding_consistency,
    deletion_insertion_test,
    compute_readability_metric
)
from .experiments import (
    test_pathology_robustness,
    test_multi_scale_token_grounding,
    fuzzy_logic_ablation_study,
    compare_fuzzy_ablation,
    test_robustness,
    test_multi_scale
)
from .evaluation import evaluate_caption_accuracy, post_process_caption


def main():
    """Main execution function for LLaVA-Med integration pipeline."""
    # Set reproducibility
    set_seed(cfg.RANDOM_SEED)
    
    # Create output directories
    output_dir = Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    captions_dir = output_dir / "captions"
    saliency_dir = output_dir / "saliency_maps"
    pli_dir = output_dir / "pli_maps"
    token_saliency_dir = output_dir / "token_saliency"
    word_saliency_dir = output_dir / "word_saliency"
    
    for dir_path in [captions_dir, saliency_dir, pli_dir, token_saliency_dir, word_saliency_dir]:
        dir_path.mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Processing {cfg.NUM_IMAGES or 'all'} images...")
    
    # Register LLaVA model config
    ensure_llava_registered(cfg.LLAVA_MED_ROOT, cfg.MODEL_ID)
    
    # Load model and tokenizer
    print("\n📦 Loading LLaVA-Med model...")
    model, tokenizer = load_llava_med_model(cfg.MODEL_ID)
    
    # Load semantic similarity model if available
    semantic_model = None
    if SEMANTIC_AVAILABLE and cfg.EVALUATE_ACCURACY:
        try:
            print("Loading semantic similarity model...")
            from sentence_transformers import SentenceTransformer
            semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Semantic model loaded")
        except Exception as e:
            print(f"⚠️ Could not load semantic model: {e}")
    
    # Initialize GradCAM
    print("\n🔧 Initializing GradCAM...")
    gradcam = GradCAM(model, tokenizer)
    # Hooks are automatically registered in GradCAM.__init__
    
    # Load dataset
    print(f"\n📂 Loading dataset from {cfg.DATASET_CSV}...")
    dataset = HistopathologyDataset(cfg.DATASET_CSV, cfg.IMAGES_DIR, limit=cfg.NUM_IMAGES)
    print(f"✅ Loaded {len(dataset)} images")
    
    # Get prompt
    prompt = get_prompt(cfg.PROMPT_STYLE, use_few_shot=cfg.USE_FEW_SHOT_EXAMPLES)
    
    # Results storage
    all_results = []
    all_evaluations = []
    
    # Process each image
    print(f"\n🚀 Processing images...")
    for idx in range(len(dataset)):
        scan_id = None
        try:
            sample = dataset[idx]
            scan_id = sample['scan_id']
            image = sample['image']
            label = sample['label']
            ground_truth = sample.get('caption', '')
            
            print(f"\n[{idx+1}/{len(dataset)}] Processing scan_id: {scan_id}")
            
            # Generate caption and saliency map
            if cfg.USE_SALIENCY_FOR_REFINEMENT:
                saliency_map, caption = gradcam.generate_cam_with_saliency_guidance(
                    image, prompt, use_saliency_for_refinement=True
                )
            else:
                saliency_map, caption = gradcam.generate_cam(image, prompt)
            
            # Post-process caption if enabled
            if cfg.ENABLE_POST_PROCESSING:
                caption_original = caption
                caption = post_process_caption(caption, label)
                if caption != caption_original:
                    print(f"  📝 Post-processed caption")
            
            # Normalize saliency map
            if saliency_map.max() > 0:
                saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-10)
            
            # Generate PLI map
            pli_map = None
            if cfg.ENABLE_WORD_LEVEL_SALIENCY or cfg.ENABLE_BASELINE_COMPARISON:
                try:
                    pli_map = gradcam_to_pli(saliency_map)
                except Exception as e:
                    print(f"  ⚠️ PLI generation failed: {e}")
            
            # Save caption
            caption_path = captions_dir / f"caption_{scan_id}.txt"
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(caption)
            
            # Save saliency map
            saliency_path = saliency_dir / f"saliency_{scan_id}.png"
            plt.figure(figsize=(8, 8))
            plt.imshow(saliency_map, cmap='jet')
            plt.axis('off')
            plt.colorbar()
            plt.title(f"Saliency Map: {caption[:50]}...")
            plt.savefig(saliency_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Save PLI map if available
            if pli_map is not None:
                pli_path = pli_dir / f"pli_{scan_id}.png"
                plt.figure(figsize=(8, 8))
                plt.imshow(pli_map, cmap='jet')
                plt.axis('off')
                plt.colorbar()
                plt.title(f"PLI Map: {caption[:50]}...")
                plt.savefig(pli_path, dpi=150, bbox_inches='tight')
                plt.close()
            
            # Token-level saliency if enabled
            token_saliency_maps = {}
            if cfg.ENABLE_TOKEN_LEVEL_SALIENCY:
                try:
                    token_saliency_maps = generate_token_level_saliency(gradcam, image, prompt, caption)
                    if cfg.SAVE_TOKEN_SALIENCY:
                        for token, token_map in token_saliency_maps.items():
                            token_safe = re.sub(r'[^\w\s-]', '', token.replace(' ', '_'))[:50]
                            token_path = token_saliency_dir / f"token_{scan_id}_{token_safe}.png"
                            
                            # Create better visualization: overlay saliency on original image
                            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                            
                            # Left: Original image
                            axes[0].imshow(image)
                            axes[0].axis('off')
                            axes[0].set_title('Original Image', fontsize=10)
                            
                            # Right: Saliency overlay on image
                            if token_map is not None and token_map.size > 0:
                                # Resize saliency map to match image if needed
                                if token_map.shape[:2] != (image.size[1], image.size[0]):
                                    import cv2
                                    token_map_resized = cv2.resize(
                                        token_map, 
                                        (image.size[0], image.size[1]), 
                                        interpolation=cv2.INTER_CUBIC
                                    )
                                else:
                                    token_map_resized = token_map
                                
                                # Normalize saliency map
                                if token_map_resized.max() > token_map_resized.min():
                                    token_map_resized = (token_map_resized - token_map_resized.min()) / (token_map_resized.max() - token_map_resized.min() + 1e-10)
                                
                                # Overlay saliency on image
                                axes[1].imshow(image)
                                overlay = axes[1].imshow(token_map_resized, cmap='jet', alpha=0.5, interpolation='bilinear')
                                axes[1].axis('off')
                                axes[1].set_title(f'Token: "{token}"\nSaliency Overlay', fontsize=10)
                                plt.colorbar(overlay, ax=axes[1], fraction=0.046, pad=0.04)
                            else:
                                axes[1].imshow(image)
                                axes[1].axis('off')
                                axes[1].set_title(f'Token: "{token}"\n(No saliency data)', fontsize=10)
                            
                            plt.tight_layout()
                            plt.savefig(token_path, dpi=150, bbox_inches='tight')
                            plt.close()
                except Exception as e:
                    print(f"  ⚠️ Token-level saliency failed: {e}")
            
            # Word-level saliency if enabled
            if cfg.ENABLE_WORD_LEVEL_SALIENCY:
                try:
                    generate_word_level_saliency_and_pli(
                        gradcam, image, prompt, caption, scan_id, 
                        str(word_saliency_dir), save_outputs=cfg.SAVE_WORD_SALIENCY
                    )
                except Exception as e:
                    print(f"  ⚠️ Word-level saliency failed: {e}")
            
            # Evaluate accuracy if ground truth available
            eval_result = None
            if cfg.EVALUATE_ACCURACY and ground_truth:
                eval_result = evaluate_caption_accuracy(caption, ground_truth, label, semantic_model)
                all_evaluations.append({
                    'scan_id': scan_id,
                    **eval_result
                })
            
            # Store results
            result = {
                'scan_id': scan_id,
                'caption': caption,
                'label': label,
                'ground_truth': ground_truth,
            }
            if eval_result:
                result.update(eval_result)
            all_results.append(result)
            
            # Extended experiments if enabled
            if cfg.ENABLE_ROBUSTNESS_EXPERIMENTS:
                try:
                    robustness_results = test_robustness(gradcam, image, prompt, saliency_map, cfg.ROBUSTNESS_NOISE_LEVELS)
                    result['robustness'] = robustness_results
                except Exception as e:
                    print(f"  ⚠️ Robustness experiments failed: {e}")
            
            if cfg.ENABLE_MULTI_SCALE:
                try:
                    multi_scale_results = test_multi_scale(gradcam, image, prompt, cfg.MULTI_SCALE_SIZES)
                    result['multi_scale'] = multi_scale_results
                except Exception as e:
                    print(f"  ⚠️ Multi-scale experiments failed: {e}")
            
            if cfg.ENABLE_CROSS_TOKEN_CONSISTENCY and token_saliency_maps:
                try:
                    consistency_results = compute_cross_token_consistency(token_saliency_maps)
                    result['cross_token_consistency'] = consistency_results
                except Exception as e:
                    print(f"  ⚠️ Cross-token consistency failed: {e}")
            
            print(f"  ✅ Completed: {caption[:60]}...")
            
        except Exception as e:
            scan_id_str = scan_id if scan_id else f"image {idx+1}"
            print(f"  ❌ Error processing {scan_id_str}: {e}")
            if cfg.DEBUG_MODE:
                import traceback
                traceback.print_exc()
            continue
    
    # Save results
    print(f"\n💾 Saving results...")
    
    # Convert numpy arrays to lists for JSON serialization
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
    
    # Save all results JSON
    results_path = output_dir / "all_results.json"
    serializable_results = convert_to_serializable(all_results)
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    # Save evaluation summary
    if all_evaluations:
        eval_df = pd.DataFrame(all_evaluations)
        eval_summary_path = output_dir / "evaluation_summary.json"
        
        summary = {
            'total_images': len(all_evaluations),
            'average_bleu': eval_df['bleu_score'].mean() if 'bleu_score' in eval_df else 0,
            'average_word_overlap': eval_df['word_overlap'].mean() if 'word_overlap' in eval_df else 0,
            'average_semantic_similarity': eval_df['semantic_similarity'].mean() if 'semantic_similarity' in eval_df else 0,
            'average_key_term_overlap': eval_df['key_term_overlap'].mean() if 'key_term_overlap' in eval_df else 0,
            'label_mention_rate': eval_df['label_mentioned'].mean() if 'label_mentioned' in eval_df else 0,
            'false_negative_rate': eval_df['false_negative'].mean() if 'false_negative' in eval_df else 0,
        }
        
        with open(eval_summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        # Save CSV for easy analysis
        eval_csv_path = output_dir / "evaluation_results.csv"
        eval_df.to_csv(eval_csv_path, index=False)
        
        print(f"✅ Evaluation summary saved")
        print(f"\n📊 Evaluation Summary:")
        print(f"   Average BLEU: {summary['average_bleu']:.4f}")
        print(f"   Average Word Overlap: {summary['average_word_overlap']:.4f}")
        print(f"   Average Semantic Similarity: {summary['average_semantic_similarity']:.4f}")
        print(f"   Label Mention Rate: {summary['label_mention_rate']:.4f}")
        print(f"   False Negative Rate: {summary['false_negative_rate']:.4f}")
    
    print(f"\n✅ Processing complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()

