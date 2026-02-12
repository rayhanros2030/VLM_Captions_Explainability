#!/usr/bin/env python3
"""
Local Windows runner script for LLaVA-Med integration.
This uses local paths instead of Lambda paths.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to Python path so we can import llavamed_modules
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Set up local paths BEFORE importing any modules
LOCAL_DATASET_DIR = Path(r"C:\Users\PC\Downloads\patchgastricadc222_prepared")
LOCAL_DATASET_CSV = LOCAL_DATASET_DIR / "dataset_with_labels.csv"
# Images are in the extracted_archive location
LOCAL_IMAGES_DIR = Path(r"C:\Users\PC\Downloads\extracted_archive\patches_captions\patches_captions")
LOCAL_OUTPUT_DIR = Path(r"C:\Users\PC\Downloads\histopathology_outputs")
LOCAL_LLAVA_MED_ROOT = Path(r"C:\Users\PC\Downloads\LLaVA-Med-main")

# Now import config and override it
import llavamed_modules.config as config_mod

# Override the config instance with local paths
config_mod.cfg.DATASET_DIR = str(LOCAL_DATASET_DIR)
config_mod.cfg.DATASET_CSV = str(LOCAL_DATASET_CSV)
config_mod.cfg.IMAGES_DIR = str(LOCAL_IMAGES_DIR)
config_mod.cfg.OUTPUT_DIR = str(LOCAL_OUTPUT_DIR)
if LOCAL_LLAVA_MED_ROOT.exists():
    config_mod.cfg.LLAVA_MED_ROOT = str(LOCAL_LLAVA_MED_ROOT)

# Local settings
config_mod.cfg.NUM_IMAGES = 50  # Start with fewer images for testing
config_mod.cfg.DEBUG_MODE = True

# Now import and run main
if __name__ == "__main__":
    from llavamed_modules.main import main
    
    print("=" * 70)
    print("LLaVA-Med Integration - Local Windows Pipeline")
    print("=" * 70)
    print(f"Working directory: {os.getcwd()}")
    print(f"Dataset: {config_mod.cfg.DATASET_CSV}")
    print(f"Images: {config_mod.cfg.IMAGES_DIR}")
    print(f"Output: {config_mod.cfg.OUTPUT_DIR}")
    print("=" * 70)
    print()
    
    # Verify dataset exists
    if not Path(config_mod.cfg.DATASET_CSV).exists():
        print(f"❌ ERROR: Dataset CSV not found: {config_mod.cfg.DATASET_CSV}")
        sys.exit(1)
    
    if not Path(config_mod.cfg.IMAGES_DIR).exists():
        print(f"⚠️  WARNING: Images directory not found: {config_mod.cfg.IMAGES_DIR}")
        print("   Continuing anyway, but image processing may fail.")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

