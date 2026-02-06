# Image-Caption Matching Guide

This guide helps you match your processed images with their generated captions.

## Quick Start

### Option 1: Standalone Script (Easiest)

Upload `match_captions_standalone.py` to Lambda and run:

```bash
cd /home/ubuntu/projects/llavamed-integration

# Make sure paths are correct in the script, then run:
python3 llavamed_modules/match_captions_standalone.py
```

This will:
- ✅ Match all images with their captions
- ✅ Create a JSON mapping file
- ✅ Generate a visualization (PNG) showing images with captions
- ✅ Print statistics

### Option 2: Using the Module

If you want to use it as part of the module:

```bash
cd /home/ubuntu/projects/llavamed-integration

python3 << 'EOF'
import sys
sys.path.insert(0, '/home/ubuntu/projects/llavamed-integration')
from llavamed_modules.match_images_captions import main
main()
EOF
```

## Output Files

After running, you'll get:

1. **`image_caption_mapping.json`** - Complete mapping of scan_id → image path, caption path, caption text, label
2. **`image_caption_mapping.png`** - Visualization showing images with their captions
3. **`matching_report.txt`** - Summary report with statistics

## Understanding the Mapping JSON

The JSON file structure:

```json
{
  "scan_id_1": {
    "image_path": "/path/to/image.jpg",
    "caption_path": "/path/to/caption_scan_id_1.txt",
    "caption": "The histopathology image shows...",
    "label": "Well differentiated tubular adenocarcinoma",
    "found_image": true,
    "found_caption": true,
    "row_index": 0
  },
  ...
}
```

## Viewing Results

### On Lambda:

```bash
# View the mapping JSON
cat /home/ubuntu/data/histopathology_outputs/image_caption_mapping.json | head -50

# View a specific caption
cat /home/ubuntu/data/histopathology_outputs/captions/caption_YOUR_SCAN_ID.txt

# View the visualization (if you have X11 forwarding)
# Or download it to your local machine
```

### Download to Local Machine:

```bash
# From your local machine (Windows PowerShell)
scp ubuntu@150.136.219.166:/home/ubuntu/data/histopathology_outputs/image_caption_mapping.json .
scp ubuntu@150.136.219.166:/home/ubuntu/data/histopathology_outputs/image_caption_mapping.png .
scp ubuntu@150.136.219.166:/home/ubuntu/data/histopathology_outputs/matching_report.txt .
```

## Troubleshooting

### Images Not Found

If images aren't being found:
1. Check the `IMAGES_DIR` path in the script matches where your images actually are
2. Verify image filenames match the scan_id pattern
3. Check if images are in subdirectories (the script searches recursively)

### Captions Not Found

If captions aren't being found:
1. Make sure you've run the main processing script first
2. Check that captions are saved in `OUTPUT_DIR/captions/`
3. Verify caption files are named `caption_{scan_id}.txt`

### Customizing

To process a different number of images, edit the script:

```python
NUM_IMAGES = 20  # Change this number
```

Or set it to `None` to process all images in the CSV.

## Example Usage

```bash
# Process 20 images and create mapping
python3 match_captions_standalone.py

# Check results
ls -lh /home/ubuntu/data/histopathology_outputs/
cat /home/ubuntu/data/histopathology_outputs/matching_report.txt
```

## Next Steps

After matching:
1. Review the visualization to see image-caption pairs
2. Check the mapping JSON for detailed information
3. Use the mapping to create evaluation reports or further analysis


