#!/usr/bin/env python3
"""
Script to process 20 images from the dataset.
This will process 20 different scan_ids from the CSV.
"""

import sys
sys.path.insert(0, '/home/ubuntu/projects/llavamed-integration')

from llavamed_modules.config import cfg

# Configure for 20 images
cfg.DATASET_CSV = '/home/ubuntu/data/histopathology_dataset/dataset_with_labels.csv'
cfg.IMAGES_DIR = '/home/ubuntu/data/histopathology_dataset/images/patches_captions'
cfg.NUM_IMAGES = 20  # Process 20 images

# Use improved prompts
cfg.PROMPT_STYLE = "descriptive"  # Better than "diagnostic" for avoiding codes
cfg.DEBUG_MODE = True
cfg.ENABLE_ROBUSTNESS_EXPERIMENTS = False
cfg.ENABLE_MULTI_SCALE = False
cfg.ENABLE_TOKEN_LEVEL_SALIENCY = False
cfg.ENABLE_WORD_LEVEL_SALIENCY = False

# Run the main processing
from llavamed_modules.main import main

if __name__ == "__main__":
    print("=" * 60)
    print("Processing 20 Images")
    print("=" * 60)
    print(f"Dataset CSV: {cfg.DATASET_CSV}")
    print(f"Images directory: {cfg.IMAGES_DIR}")
    print(f"Number of images: {cfg.NUM_IMAGES}")
    print(f"Prompt style: {cfg.PROMPT_STYLE}")
    print("=" * 60)
    print()
    
    main()


