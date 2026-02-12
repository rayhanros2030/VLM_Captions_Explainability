#!/bin/bash
# Quick setup script for Lambda Labs instance
# Run this after uploading your files

set -e  # Exit on error

echo "=========================================="
echo "LLaVA-Med Integration Setup"
echo "=========================================="

# Create directories
echo "📁 Creating directories..."
mkdir -p ~/projects/llavamed-integration/llavamed_modules
mkdir -p ~/data/histopathology_dataset/images/patches_captions
mkdir -p ~/data/histopathology_outputs

# Clone LLaVA-Med if not exists
if [ ! -d ~/projects/LLaVA-Med-main ]; then
    echo "📦 Cloning LLaVA-Med repository..."
    cd ~/projects
    git clone https://github.com/microsoft/LLaVA-Med.git LLaVA-Med-main
fi

# Install/upgrade dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip --quiet

# Fix NumPy version (critical for compatibility)
echo "   Installing NumPy < 2.0..."
pip install "numpy<2.0" --quiet

# Core dependencies
echo "   Installing core packages..."
pip install transformers>=4.36.0 --quiet
pip install pillow>=9.0.0 --quiet
pip install sentencepiece --quiet
pip install tf-keras --quiet

# Other dependencies
echo "   Installing additional packages..."
pip install opencv-python matplotlib pandas scikit-learn --quiet
pip install scikit-fuzzy sentence-transformers --quiet
pip install nltk --quiet

# Download NLTK data
echo "   Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt', quiet=True)" 2>/dev/null || true

# Create __init__.py if it doesn't exist
if [ ! -f ~/projects/llavamed-integration/llavamed_modules/__init__.py ]; then
    echo "📝 Creating __init__.py..."
    cat > ~/projects/llavamed-integration/llavamed_modules/__init__.py << 'INITEOF'
"""
LLaVA-Med Integration Package
"""
from .config import cfg, ConfigPaths, set_seed, BLEU_AVAILABLE, PLI_AVAILABLE, SEMANTIC_AVAILABLE
__all__ = ['cfg', 'ConfigPaths', 'set_seed', 'BLEU_AVAILABLE', 'PLI_AVAILABLE', 'SEMANTIC_AVAILABLE']
INITEOF
fi

# Test import
echo "🧪 Testing setup..."
cd ~/projects/llavamed-integration
python3 << 'TESTEOF'
import sys
sys.path.insert(0, '/home/ubuntu/projects/llavamed-integration')
try:
    from llavamed_modules.config import cfg
    print("✅ Setup successful!")
    print(f"   LLaVA-Med root: {cfg.LLAVA_MED_ROOT}")
    print(f"   Output dir: {cfg.OUTPUT_DIR}")
except Exception as e:
    print(f"❌ Setup failed: {e}")
    sys.exit(1)
TESTEOF

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Upload your CSV file to: ~/data/histopathology_dataset/"
echo "2. Upload your images to: ~/data/histopathology_dataset/images/patches_captions/"
echo "3. Run: python3 llavamed_modules/run_20_images.py"
echo ""


