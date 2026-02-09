# Complete Lambda Setup Guide - LLaVA-Med Integration

This guide will help you set up and run the LLaVA-Med integration system on Lambda Labs A100 (40 GB SXM4) with Lambda Stack 22.04.

## Prerequisites

- Lambda Labs account with SSH key configured
- Your Lambda instance IP address
- All your files ready on your local machine

---

## Step 1: Connect to Lambda Instance

```bash
# From your local machine (PowerShell or terminal)
ssh ubuntu@YOUR_LAMBDA_IP
# Example: ssh ubuntu@150.136.219.166
```

If you get "Permission denied", make sure your SSH key is added to your Lambda Labs account.

---

## Step 2: Create Directory Structure

Once connected to Lambda, run:

```bash
# Create main project directory
mkdir -p ~/projects/llavamed-integration
cd ~/projects/llavamed-integration

# Create data directories
mkdir -p ~/data/histopathology_dataset/images/patches_captions
mkdir -p ~/data/histopathology_outputs
```

---

## Step 3: Clone LLaVA-Med Repository

```bash
cd ~/projects
git clone https://github.com/microsoft/LLaVA-Med.git LLaVA-Med-main
cd ~/projects/llavamed-integration
```

---

## Step 4: Install Python Dependencies

```bash
# Update pip first
pip install --upgrade pip

# Install core dependencies (fixing version conflicts)
pip install "numpy<2.0"  # Important: NumPy 2.x causes compatibility issues
pip install transformers>=4.36.0
pip install pillow>=9.0.0
pip install sentencepiece
pip install tf-keras  # Required for transformers compatibility

# Install other dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install accelerate
pip install opencv-python
pip install matplotlib
pip install pandas
pip install scikit-learn
pip install scikit-fuzzy  # For PLI maps
pip install sentence-transformers  # For semantic similarity
pip install nltk

# Download NLTK data
python3 -c "import nltk; nltk.download('punkt', quiet=True)"
```

---

## Step 5: Upload Your Files

From your **local machine** (PowerShell), upload files:

### Upload Python Modules

```powershell
# Navigate to your modules directory
cd C:\Users\PC\Downloads\llavamed_modules

# Upload all Python files
scp *.py ubuntu@YOUR_LAMBDA_IP:~/projects/llavamed-integration/llavamed_modules/
```

### Upload Dataset CSV

```powershell
# Upload your CSV file
scp "C:\Users\PC\Downloads\patchgastricadc222_prepared\dataset_with_labels.csv" ubuntu@YOUR_LAMBDA_IP:~/data/histopathology_dataset/
```

### Upload Images

```powershell
# Upload images (this may take a while)
# Option 1: Upload entire directory
scp -r "C:\Users\PC\Downloads\extracted_archive\patches_captions\patches_captions\*" ubuntu@YOUR_LAMBDA_IP:~/data/histopathology_dataset/images/patches_captions/

# Option 2: If you have a lot of images, use rsync (faster, resumable)
# First install rsync on Windows or use WSL
```

---

## Step 6: Set Up Python Path

Back on Lambda, create the `__init__.py` file:

```bash
cd ~/projects/llavamed-integration/llavamed_modules
touch __init__.py
```

The `__init__.py` should import the config. If it doesn't exist, create it:

```bash
cat > __init__.py << 'EOF'
"""
LLaVA-Med Integration Package
"""
from .config import cfg, ConfigPaths, set_seed, BLEU_AVAILABLE, PLI_AVAILABLE, SEMANTIC_AVAILABLE
__all__ = ['cfg', 'ConfigPaths', 'set_seed', 'BLEU_AVAILABLE', 'PLI_AVAILABLE', 'SEMANTIC_AVAILABLE']
EOF
```

---

## Step 7: Verify Setup

```bash
cd ~/projects/llavamed-integration

# Test import
python3 << 'EOF'
import sys
sys.path.insert(0, '/home/ubuntu/projects/llavamed-integration')
from llavamed_modules.config import cfg
print("✅ Config loaded successfully!")
print(f"   LLaVA-Med root: {cfg.LLAVA_MED_ROOT}")
print(f"   Dataset CSV: {cfg.DATASET_CSV}")
print(f"   Images dir: {cfg.IMAGES_DIR}")
EOF
```

---

## Step 8: Run the System

### Option A: Quick Test (1 image)

```bash
cd ~/projects/llavamed-integration

python3 << 'EOF'
import sys
sys.path.insert(0, '/home/ubuntu/projects/llavamed-integration')
from llavamed_modules.config import cfg

# Test settings
cfg.DATASET_CSV = '/home/ubuntu/data/histopathology_dataset/dataset_with_labels.csv'
cfg.IMAGES_DIR = '/home/ubuntu/data/histopathology_dataset/images/patches_captions'
cfg.NUM_IMAGES = 1
cfg.DEBUG_MODE = True
cfg.PROMPT_STYLE = "descriptive"  # Better captions

# Disable experiments for faster test
cfg.ENABLE_ROBUSTNESS_EXPERIMENTS = False
cfg.ENABLE_MULTI_SCALE = False
cfg.ENABLE_TOKEN_LEVEL_SALIENCY = False
cfg.ENABLE_WORD_LEVEL_SALIENCY = False

from llavamed_modules.main import main
main()
EOF
```

### Option B: Process Multiple Images

```bash
cd ~/projects/llavamed-integration

python3 << 'EOF'
import sys
sys.path.insert(0, '/home/ubuntu/projects/llavamed-integration')
from llavamed_modules.config import cfg

# Configure
cfg.DATASET_CSV = '/home/ubuntu/data/histopathology_dataset/dataset_with_labels.csv'
cfg.IMAGES_DIR = '/home/ubuntu/data/histopathology_dataset/images/patches_captions'
cfg.NUM_IMAGES = 20  # Process 20 images
cfg.PROMPT_STYLE = "descriptive"

# Enable features you want
cfg.ENABLE_TOKEN_LEVEL_SALIENCY = True
cfg.SAVE_TOKEN_SALIENCY = True
cfg.ENABLE_WORD_LEVEL_SALIENCY = False  # Can enable if needed
cfg.ENABLE_ROBUSTNESS_EXPERIMENTS = False
cfg.ENABLE_MULTI_SCALE = False

from llavamed_modules.main import main
main()
EOF
```

---

## Step 9: Check Results

```bash
# List output files
ls -lh ~/data/histopathology_outputs/

# View captions
ls ~/data/histopathology_outputs/captions/

# View a specific caption
cat ~/data/histopathology_outputs/captions/caption_*.txt | head -1

# View evaluation summary
cat ~/data/histopathology_outputs/evaluation_summary.json
```

---

## Step 10: Match Captions to Images

```bash
cd ~/projects/llavamed-integration

python3 llavamed_modules/match_captions_to_images_by_id.py
```

This will create:
- `caption_image_mapping_by_id.json` - Complete mapping
- Visualizations showing matched images and captions

---

## Troubleshooting

### Issue: "Permission denied (publickey)"
**Solution:** Make sure your SSH public key is added to your Lambda Labs account settings.

### Issue: "NumPy version incompatibility"
**Solution:** 
```bash
pip install "numpy<2.0"
pip install --force-reinstall scipy scikit-learn
```

### Issue: "Image not found for scan_id"
**Solution:** 
- Check that images are in the correct directory
- Verify image filenames match the pattern: `{scan_id}_*.jpg`
- Run the diagnostic script: `python3 llavamed_modules/diagnose_images.py`

### Issue: "ImportError: cannot import name 'MistralModel'"
**Solution:**
```bash
pip install --upgrade transformers>=4.36.0
```

### Issue: "ValueError: Your currently installed version of Keras is Keras 3"
**Solution:**
```bash
pip install tf-keras
```

### Issue: "ImportError: LlamaTokenizer requires the SentencePiece library"
**Solution:**
```bash
pip install sentencepiece
```

---

## File Structure on Lambda

```
/home/ubuntu/
├── projects/
│   ├── LLaVA-Med-main/          # Cloned LLaVA-Med repo
│   └── llavamed-integration/
│       └── llavamed_modules/    # Your Python modules
│           ├── __init__.py
│           ├── config.py
│           ├── dataset.py
│           ├── main.py
│           └── ... (all other modules)
└── data/
    ├── histopathology_dataset/
    │   ├── dataset_with_labels.csv
    │   └── images/
    │       └── patches_captions/  # Your images here
    └── histopathology_outputs/   # Results saved here
        ├── captions/
        ├── saliency/
        ├── token_saliency/
        └── all_results.json
```

---

## Quick Reference Commands

```bash
# Connect to Lambda
ssh ubuntu@YOUR_LAMBDA_IP

# Navigate to project
cd ~/projects/llavamed-integration

# Run test
python3 llavamed_modules/run_20_images.py

# Check GPU
nvidia-smi

# Check disk space
df -h

# View logs
tail -f ~/data/histopathology_outputs/*.log
```

---

## Notes

- The system processes images one at a time
- Each image takes ~30-60 seconds depending on features enabled
- Token saliency adds significant time (processes each token separately)
- Results are saved incrementally, so you can stop and resume
- Use `screen` or `tmux` for long-running jobs:
  ```bash
  screen -S llavamed
  # Run your command
  # Press Ctrl+A then D to detach
  # Reattach with: screen -r llavamed
  ```

---

## Success Indicators

✅ Model loads without errors
✅ "Processing images..." message appears
✅ Captions are generated and saved
✅ Saliency maps are created
✅ Results JSON file is created
✅ No "Image not found" errors

If you see these, the system is working correctly!


