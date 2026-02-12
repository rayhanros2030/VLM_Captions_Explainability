"""
Local Windows configuration for LLaVA-Med integration.
Use this file to override default Lambda paths for local Windows development.
"""

import os
from pathlib import Path
from llavamed_modules.config import ConfigPaths

# Get the parent directory of this config file
CONFIG_DIR = Path(__file__).parent.parent

# Local dataset paths
LOCAL_DATASET_DIR = Path(r"C:\Users\PC\Downloads\patchgastricadc222_prepared")
LOCAL_DATASET_CSV = LOCAL_DATASET_DIR / "dataset_with_labels.csv"
LOCAL_IMAGES_DIR = LOCAL_DATASET_DIR / "images"

# Local output directory
LOCAL_OUTPUT_DIR = Path(r"C:\Users\PC\Downloads\histopathology_outputs")

# LLaVA-Med root (adjust if different on your machine)
# If you have LLaVA-Med cloned locally, set this path
LOCAL_LLAVA_MED_ROOT = Path(r"C:\Users\PC\Downloads\LLaVA-Med-main")  # Update this if needed

# Create local config by extending the base ConfigPaths
class LocalConfig(ConfigPaths):
    """Local Windows configuration overriding Lambda paths."""
    
    # Override with local paths
    LLAVA_MED_ROOT: str = str(LOCAL_LLAVA_MED_ROOT) if LOCAL_LLAVA_MED_ROOT.exists() else ConfigPaths.LLAVA_MED_ROOT
    DATASET_DIR: str = str(LOCAL_DATASET_DIR)
    DATASET_CSV: str = str(LOCAL_DATASET_CSV)
    IMAGES_DIR: str = str(LOCAL_IMAGES_DIR)
    OUTPUT_DIR: str = str(LOCAL_OUTPUT_DIR)
    
    # Local settings (adjust as needed)
    NUM_IMAGES: int = 50  # Start with fewer images for testing
    
    # Other settings can be adjusted here
    DEBUG_MODE: bool = True  # Enable debugging for local development

# Create local config instance
cfg = LocalConfig()

# Print configuration
print(f"📁 Local Configuration:")
print(f"   Dataset CSV: {cfg.DATASET_CSV}")
print(f"   Images Dir: {cfg.IMAGES_DIR}")
print(f"   Output Dir: {cfg.OUTPUT_DIR}")
print(f"   LLaVA-Med Root: {cfg.LLAVA_MED_ROOT}")
print(f"   Num Images: {cfg.NUM_IMAGES}")



