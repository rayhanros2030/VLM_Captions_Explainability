# Lambda Labs Setup Guide - LLaVA-Med Integration

## Prerequisites
- Lambda Labs instance with A100 (40GB SXM4)
- Lambda Stack 22.04 installed
- SSH access to your instance

## Step 1: Upload Your Files to Lambda

### Option A: Using SCP (from your local Windows machine)

Open PowerShell and run:

```powershell
# Navigate to your Downloads folder
cd C:\Users\PC\Downloads

# Upload the modules directory
scp -r llavamed_modules ubuntu@<your-lambda-ip>:/home/ubuntu/projects/llavamed-integration/

# Upload the dataset CSV
scp patchgastricadc222_prepared\dataset_with_labels.csv ubuntu@<your-lambda-ip>:/home/ubuntu/data/histopathology_dataset/

# Upload images (this may take a while - 262k+ files)
# Option 1: Compress first (recommended)
# On Windows, create a zip of the images folder, then:
scp extracted_archive.zip ubuntu@<your-lambda-ip>:/home/ubuntu/data/
# Then SSH in and extract: unzip extracted_archive.zip

# Option 2: Use rsync (if available on Windows via WSL or Git Bash)
# rsync -avz extracted_archive/patches_captions/patches_captions/ ubuntu@<your-lambda-ip>:/home/ubuntu/data/histopathology_dataset/images/
```

### Option B: Using Lambda's File Manager (if available)

1. Log into Lambda Labs dashboard
2. Use the file manager to upload files
3. Upload to appropriate directories

### Option C: Clone/Download on Lambda

SSH into your Lambda instance and download directly:

```bash
# SSH into Lambda
ssh ubuntu@<your-lambda-ip>

# Create directories
mkdir -p /home/ubuntu/projects/llavamed-integration
mkdir -p /home/ubuntu/data/histopathology_dataset/images
mkdir -p /home/ubuntu/data/histopathology_outputs

# If you have the files in a cloud storage (S3, Google Drive, etc.)
# Download them here
```

## Step 2: Run Setup Script

Once files are uploaded, SSH into Lambda and run:

```bash
# SSH into Lambda
ssh ubuntu@<your-lambda-ip>

# Navigate to project
cd /home/ubuntu/projects/llavamed-integration

# Make setup script executable
chmod +x llavamed_modules/lambda_setup.sh

# Run setup
./llavamed_modules/lambda_setup.sh
```

This will:
- Install all Python dependencies
- Download NLTK data
- Verify GPU availability
- Set up PYTHONPATH
- Check for dataset and LLaVA-Med

## Step 3: Verify Dataset Structure

Ensure your dataset is structured correctly:

```bash
# Check CSV exists
ls -lh /home/ubuntu/data/histopathology_dataset/dataset_with_labels.csv

# Check images directory
ls /home/ubuntu/data/histopathology_dataset/images/ | head -5

# Count images
find /home/ubuntu/data/histopathology_dataset/images -type f | wc -l
```

## Step 4: Configure LLaVA-Med Path

If LLaVA-Med is in a different location, update the config:

```bash
cd /home/ubuntu/projects/llavamed-integration
python3 -c "
import sys
sys.path.insert(0, '/home/ubuntu/projects/llavamed-integration')
from llavamed_modules.config import cfg
cfg.LLAVA_MED_ROOT = '/path/to/your/LLaVA-Med-main'
print(f'LLaVA-Med path set to: {cfg.LLAVA_MED_ROOT}')
"
```

Or edit `llavamed_modules/config.py` directly.

## Step 5: Run the Pipeline

```bash
cd /home/ubuntu/projects/llavamed-integration

# Test with a small number of images first
python3 -c "
import sys
sys.path.insert(0, '/home/ubuntu/projects/llavamed-integration')
from llavamed_modules.config import cfg
cfg.NUM_IMAGES = 5  # Test with 5 images
from llavamed_modules.main import main
main()
"

# Or use the runner script
python3 llavamed_modules/run.py
```

## Step 6: Monitor Progress

In a separate terminal, monitor GPU usage:

```bash
watch -n 1 nvidia-smi
```

Check output directory:

```bash
ls -lh /home/ubuntu/data/histopathology_outputs/
tail -f /path/to/output.log  # if running with logging
```

## Expected Directory Structure on Lambda

```
/home/ubuntu/
├── projects/
│   ├── llavamed-integration/
│   │   └── llavamed_modules/  (all your Python modules)
│   └── LLaVA-Med-main/  (LLaVA-Med repository)
└── data/
    ├── histopathology_dataset/
    │   ├── dataset_with_labels.csv
    │   └── images/  (all your image files)
    └── histopathology_outputs/  (results will be saved here)
        ├── captions/
        ├── saliency_maps/
        ├── pli_maps/
        └── evaluation_summary.json
```

## Troubleshooting

### Import Errors
```bash
export PYTHONPATH=/home/ubuntu/projects/llavamed-integration:$PYTHONPATH
```

### CUDA Out of Memory
- Reduce `NUM_IMAGES` in config
- Disable extended experiments
- Process images in smaller batches

### Missing Dependencies
```bash
pip install --upgrade -r llavamed_modules/requirements.txt
```

### Dataset Not Found
- Verify paths in config match your actual directory structure
- Check file permissions: `chmod -R 755 /home/ubuntu/data/histopathology_dataset`

## Quick Start (After Setup)

```bash
cd /home/ubuntu/projects/llavamed-integration
python3 llavamed_modules/run.py
```


