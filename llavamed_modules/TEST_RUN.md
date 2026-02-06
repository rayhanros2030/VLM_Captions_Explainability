# Test Run on Lambda

## Step 1: SSH into Lambda

SSH into your Lambda instance:
```
ssh ubuntu@YOUR_LAMBDA_IP
```

## Step 2: Run Setup (if not done yet)

```bash
cd /home/ubuntu/projects/llavamed-integration
chmod +x llavamed_modules/lambda_setup.sh
./llavamed_modules/lambda_setup.sh
```

This installs dependencies and checks GPU.

## Step 3: Test with 1-2 Images

Once setup is complete, test with just 1-2 images:

```bash
cd /home/ubuntu/projects/llavamed-integration

python3 -c "
import sys
sys.path.insert(0, '/home/ubuntu/projects/llavamed-integration')
from llavamed_modules.config import cfg

# Test settings
cfg.NUM_IMAGES = 1  # Just test with 1 image
cfg.DEBUG_MODE = True
cfg.ENABLE_ROBUSTNESS_EXPERIMENTS = False  # Disable for faster test
cfg.ENABLE_MULTI_SCALE = False  # Disable for faster test
cfg.ENABLE_TOKEN_LEVEL_SALIENCY = False  # Disable for faster test
cfg.ENABLE_WORD_LEVEL_SALIENCY = False  # Disable for faster test

from llavamed_modules.main import main
main()
"
```

## What This Will Do

1. Load the LLaVA-Med model (first time will download ~13GB - takes 5-10 minutes)
2. Process 1 image from your dataset
3. Generate a caption
4. Create saliency maps
5. Save results to `/home/ubuntu/data/histopathology_outputs/`

## Expected Output

You should see:
- Model loading messages
- Image processing progress
- Caption generation
- Results saved to output directory

## Check Results

After it completes:

```bash
# Check output directory
ls -lh /home/ubuntu/data/histopathology_outputs/

# View generated caption
cat /home/ubuntu/data/histopathology_outputs/captions/caption_*.txt

# Check if saliency maps were created
ls /home/ubuntu/data/histopathology_outputs/saliency_maps/
```

## Troubleshooting

**If you get "Image not found" errors:**
- Some images might not be uploaded yet - that's okay for testing
- The system will skip images it can't find

**If you get import errors:**
- Make sure setup script ran successfully
- Check PYTHONPATH: `echo $PYTHONPATH`

**If model download is slow:**
- First run downloads the model (~13GB)
- Subsequent runs will be faster (model is cached)


