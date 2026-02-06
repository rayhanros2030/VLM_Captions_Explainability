#!/bin/bash
# Lambda Labs Setup Script for LLaVA-Med Integration
# Run this on your Lambda instance to set up the environment

set -e  # Exit on error

echo "=========================================="
echo "LLaVA-Med Integration - Lambda Setup"
echo "=========================================="
echo ""

# Check if running on Lambda Stack
if [ ! -f /opt/lambda-stack/version ]; then
    echo "⚠️  Warning: Lambda Stack not detected. Proceeding anyway..."
fi

# Check Python version
echo "Checking Python version..."
python3 --version
echo ""

# Create project directory structure
echo "Creating project directories..."
mkdir -p /home/ubuntu/projects/llavamed-integration
mkdir -p /home/ubuntu/data/histopathology_dataset
mkdir -p /home/ubuntu/data/histopathology_outputs
echo "✅ Directories created"
echo ""

# Install Python dependencies
echo "Installing Python dependencies..."
cd /home/ubuntu/projects/llavamed-integration
pip install --upgrade pip
if [ -f "llavamed_modules/requirements.txt" ]; then
    pip install -r llavamed_modules/requirements.txt
else
    echo "⚠️  requirements.txt not found, installing core packages..."
    pip install torch transformers accelerate pillow opencv-python matplotlib pandas numpy nltk sentence-transformers scikit-fuzzy
fi
echo "✅ Dependencies installed"
echo ""

# Download NLTK data
echo "Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt_tab', quiet=True)" || \
python3 -c "import nltk; nltk.download('punkt', quiet=True)" || \
echo "⚠️  NLTK data download skipped (will be downloaded on first use)"
echo ""

# Verify GPU
echo "Checking GPU availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Set PYTHONPATH (add to bashrc if not already there)
if ! grep -q "llavamed-integration" ~/.bashrc 2>/dev/null; then
    echo "Adding PYTHONPATH to ~/.bashrc..."
    echo 'export PYTHONPATH=/home/ubuntu/projects/llavamed-integration:$PYTHONPATH' >> ~/.bashrc
    export PYTHONPATH=/home/ubuntu/projects/llavamed-integration:$PYTHONPATH
    echo "✅ PYTHONPATH updated"
else
    echo "✅ PYTHONPATH already configured"
fi
echo ""

# Check LLaVA-Med repository
echo "Checking for LLaVA-Med repository..."
if [ -d "/home/ubuntu/projects/LLaVA-Med-main" ]; then
    echo "✅ LLaVA-Med repository found at /home/ubuntu/projects/LLaVA-Med-main"
else
    echo "⚠️  LLaVA-Med repository not found at /home/ubuntu/projects/LLaVA-Med-main"
    echo "   You may need to clone it or create a symlink"
fi
echo ""

# Check dataset
echo "Checking dataset..."
if [ -f "/home/ubuntu/data/histopathology_dataset/dataset_with_labels.csv" ]; then
    echo "✅ Dataset CSV found"
    IMG_COUNT=$(find /home/ubuntu/data/histopathology_dataset/images -type f 2>/dev/null | wc -l)
    echo "   Found $IMG_COUNT image files"
else
    echo "⚠️  Dataset CSV not found at /home/ubuntu/data/histopathology_dataset/dataset_with_labels.csv"
    echo "   You need to upload your dataset CSV and images"
fi
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Upload your dataset CSV to: /home/ubuntu/data/histopathology_dataset/dataset_with_labels.csv"
echo "2. Upload your images to: /home/ubuntu/data/histopathology_dataset/images/"
echo "3. Ensure LLaVA-Med is at: /home/ubuntu/projects/LLaVA-Med-main"
echo "4. Run: cd /home/ubuntu/projects/llavamed-integration && python3 llavamed_modules/run.py"
echo ""


