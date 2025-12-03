# VLM Pathology Project - Caption Explainability

A comprehensive vision-language model (VLM) project for histopathology image captioning and explainability analysis. This project integrates LLaVA-Med and BLIP models to generate medical image captions with saliency map visualization and comprehensive evaluation metrics.

## Overview

This project focuses on:
- **Medical Image Captioning**: Generating descriptive and diagnostic captions for histopathology images
- **Explainability**: Visualizing model attention through Grad-CAM saliency maps and Pixel-Level Interpretability (PLI) maps
- **Evaluation**: Comprehensive assessment using BLEU scores, semantic similarity, and faithfulness metrics
- **Dataset Support**: PCAM (PatchCamelyon) histopathology dataset integration

## Features

- 🤖 **Multiple VLM Support**: 
  - LLaVA-Med (microsoft/llava-med-v1.5-mistral-7b) for medical-specific captioning
  - BLIP (Salesforce/blip-image-captioning-base) for general image captioning

- 🔍 **Saliency Map Generation**:
  - Grad-CAM implementation for attention visualization
  - Activation-based fallback methods
  - Pixel-Level Interpretability (PLI) maps using fuzzy logic

- 📊 **Comprehensive Evaluation**:
  - BLEU score calculation
  - Semantic similarity using sentence transformers
  - Continuity metrics (SSIM, MAE)
  - Lipschitz stability
  - Faithfulness testing through masking

- 🏥 **Medical Dataset Support**:
  - PCAM histopathology images
  - Custom dataset with labels integration
  - Support for multiple image formats (PNG, JPG, TIFF)

## Installation

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended) or CPU
- Conda (for environment management)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/rayhanros2030/VLM_Captions_Explainability.git
cd VLM_Captions_Explainability
```

2. **Create conda environment**:
```bash
conda env create -f environment.yml
conda activate vlm_pathology
```

3. **Install additional dependencies** (if needed):
```bash
pip install nltk scikit-fuzzy sentence-transformers scikit-image opencv-python
```

4. **Download NLTK data**:
```python
import nltk
nltk.download('punkt')
```

## Project Structure

```
VLM_Captions_Explainability/
├── llavamed_integration_with_labels.py  # Main LLaVA-Med integration script
├── caption_image.py                      # Simple BLIP captioning example
├── caption_pcam_images.py                # Batch PCAM image captioning
├── saliency_map.py                       # Saliency map generation utilities
├── extract_saliency_maps.py              # Extract saliency maps for dataset
├── extract_pcam_patches.py               # PCAM dataset preprocessing
├── eval_metrics.py                       # Evaluation metrics implementation
├── example_usage.py                      # Complete evaluation pipeline example
├── eval_metrics.py                       # Evaluation utilities
├── environment.yml                       # Conda environment configuration
├── data/
│   ├── pcam/
│   │   └── sample_images/               # Sample PCAM images
│   └── histopathology_dataset/          # Custom dataset directory
└── README.md
```

## Usage

### Basic Image Captioning

**Using BLIP for simple captioning**:
```python
python caption_image.py
```

**Using LLaVA-Med for medical captioning**:
```python
python llavamed_integration_with_labels.py
```

### Generating Saliency Maps

**Single image saliency**:
```python
python saliency_single.py
```

**Batch processing**:
```python
python saliency_loop.py
```

### Complete Evaluation Pipeline

Run the complete evaluation pipeline including PLI maps:
```python
python example_usage.py
```

### Configuration

Edit `llavamed_integration_with_labels.py` to configure paths and settings:

```python
@dataclass
class ConfigPaths:
    LLAVA_MED_ROOT: str = "/path/to/LLaVA-Med-main"
    DATASET_DIR: str = "/path/to/dataset"
    DATASET_CSV: str = "/path/to/dataset.csv"
    IMAGES_DIR: str = "/path/to/images"
    OUTPUT_DIR: str = "/path/to/outputs"
    MODEL_ID: str = "microsoft/llava-med-v1.5-mistral-7b"
    NUM_IMAGES: Optional[int] = 750
    PROMPT_STYLE: str = "diagnostic"  # Options: "descriptive", "diagnostic", "classification"
```

### Prompt Styles

The project supports three prompt styles:

1. **Descriptive**: General description of the image
2. **Diagnostic**: Medical diagnosis-focused descriptions
3. **Classification**: Classification-focused output

## Evaluation Metrics

The project implements several evaluation metrics:

### Caption Quality Metrics
- **BLEU Score**: N-gram overlap between generated and reference captions
- **Semantic Similarity**: Cosine similarity using sentence embeddings
- **Word Overlap**: Simple word-level similarity

### Interpretability Metrics
- **Continuity (SSIM)**: Structural similarity of saliency maps under perturbations
- **Continuity (MAE)**: Mean absolute error of saliency maps
- **Lipschitz Stability**: Smoothness measure of saliency maps
- **Faithfulness**: Correlation between saliency importance and prediction impact

### Image Perturbations
- Gaussian noise addition
- Brightness variation
- Rotation

## Datasets

### PCAM (PatchCamelyon)
The project includes sample PCAM histopathology images in `data/pcam/sample_images/`. PCAM is a binary classification dataset for detecting metastatic tissue in histopathologic scans of lymph node sections.

### Custom Dataset Format
For custom datasets, provide a CSV file with the following columns:
- `scan_id`: Unique identifier for each image
- `image_filename`: Name of the image file (optional)
- `label` or `subtype`: Classification label
- `text` or `caption`: Ground truth caption/description
- `has_image`: Boolean indicating if image file exists

## Output

The project generates:
- **Captions**: Generated descriptions for each image
- **Saliency Maps**: Visual heatmaps showing model attention
- **PLI Maps**: Pixel-level interpretability maps
- **Evaluation Reports**: JSON files with detailed metrics
- **Comparison Visualizations**: Side-by-side comparisons of images, captions, and saliency maps

## Requirements

Key dependencies include:
- `torch` (2.1.0)
- `transformers` (4.35.0)
- `pillow` (11.3.0)
- `matplotlib` (3.10.6)
- `numpy` (1.25.0)
- `pandas`
- `opencv-python`
- `nltk` (for BLEU scores)
- `scikit-fuzzy` (for PLI maps)
- `sentence-transformers` (for semantic similarity)
- `scikit-image` (for evaluation metrics)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is provided as-is for research and educational purposes.

## Citation

If you use this project in your research, please cite the relevant papers:
- LLaVA-Med: [LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day](https://arxiv.org/abs/2306.00890)
- BLIP: [BLIP: Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2201.12086)
- PCAM: [PatchCamelyon: A New Benchmark for Image Classification](https://github.com/basveeling/pcam)

## Contact

For questions or issues, please open an issue on the GitHub repository.

## Acknowledgments

- Microsoft LLaVA-Med team for the medical vision-language model
- Salesforce Research for BLIP
- PCAM dataset contributors

