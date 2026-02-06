"""
Configuration and setup module for LLaVA-Med integration.
Contains configuration dataclass and optional dependency setup.
"""

import os
import random
from dataclasses import dataclass, field
from typing import Optional, List

# Disable accelerate/bitsandbytes auto-import
os.environ["TRANSFORMERS_NO_ACCELERATE"] = "1"

# ==========================
# REPRODUCIBILITY SETUP
# ==========================
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ Random seed set to {seed} for reproducibility")

# ==========================
# OPTIONAL DEPENDENCIES
# ==========================

# Import evaluation metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk

    # Download required NLTK data (try both punkt and punkt_tab for compatibility)
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        try:
            nltk.download('punkt', quiet=True)
        except:
            pass
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    print("[WARNING] nltk not available. BLEU scores will not be computed.")

# Import PLI conversion (fuzzy logic)
try:
    import skfuzzy as fuzz
    PLI_AVAILABLE = True
except ImportError:
    PLI_AVAILABLE = False
    print("[WARNING] skfuzzy not available. PLI maps will not be computed.")
    print("         Install with: pip install scikit-fuzzy")

# Import semantic similarity (sentence embeddings)
try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("[WARNING] sentence-transformers not available. Semantic similarity will not be computed.")
    print("         Install with: pip install sentence-transformers")


# ==========================
# CONFIG
# ==========================

@dataclass
class ConfigPaths:
    # Root of your LLaVA-Med repo on Lambda
    LLAVA_MED_ROOT: str = "/home/ubuntu/projects/LLaVA-Med-main"

    # New histopathology dataset on Lambda
    DATASET_DIR: str = "/home/ubuntu/data/histopathology_dataset"
    DATASET_CSV: str = "/home/ubuntu/data/histopathology_dataset/dataset_with_labels.csv"
    IMAGES_DIR: str = "/home/ubuntu/data/histopathology_dataset/images"

    # Output dir on Lambda
    OUTPUT_DIR: str = "/home/ubuntu/data/histopathology_outputs"

    MODEL_ID: str = "microsoft/llava-med-v1.5-mistral-7b"
    NUM_IMAGES: Optional[int] = 750  # Process 750 samples

    # Activation-based saliency method
    ACTIVATION_METHOD: str = "l2_norm"

    # Prompt options: "descriptive", "diagnostic", "classification"
    PROMPT_STYLE: str = "diagnostic"  # Options: "descriptive", "diagnostic", "classification"

    # Generation settings
    GENERATION_TEMPERATURE: float = 0.85  # 0.0 = deterministic, 0.7-1.0 = more variation
    MAX_NEW_TOKENS: int = 80  # Maximum tokens to generate

    # Evaluation settings
    EVALUATE_ACCURACY: bool = True
    SAVE_DETAILED_RESULTS: bool = True
    SHOW_SAMPLE_COMPARISONS: bool = True  # Show sample comparisons at the end
    DEBUG_MODE: bool = False  # Enable verbose debugging
    ENABLE_POST_PROCESSING: bool = True  # Enable post-processing to fix false negatives
    USE_SALIENCY_FOR_REFINEMENT: bool = True  # Use saliency maps to guide caption generation (two-pass)
    USE_FEW_SHOT_EXAMPLES: bool = True  # Include few-shot examples in prompts
    
    # Reproducibility settings
    RANDOM_SEED: int = 42  # Random seed for reproducibility
    
    # Token-level saliency settings
    ENABLE_TOKEN_LEVEL_SALIENCY: bool = True  # Generate token-level saliency maps
    SAVE_TOKEN_SALIENCY: bool = True  # Save individual token saliency maps
    
    # Word-level saliency settings (NEW - for enhanced explainability)
    ENABLE_WORD_LEVEL_SALIENCY: bool = True  # Generate word-level saliency and PLI maps
    SAVE_WORD_SALIENCY: bool = True  # Save individual word saliency and PLI maps
    
    # Baseline comparison settings
    ENABLE_BASELINE_COMPARISON: bool = True  # Compare PLI vs baseline methods
    BASELINE_METHODS: List[str] = field(default_factory=lambda: ["raw_gradcam", "attention"])  # Baseline methods to compare
    
    # Extended experiments settings
    ENABLE_CROSS_TOKEN_CONSISTENCY: bool = True  # Analyze cross-token consistency
    ENABLE_ROBUSTNESS_EXPERIMENTS: bool = True  # Run robustness experiments
    ENABLE_MULTI_SCALE: bool = True  # Multi-scale grounding experiments
    ROBUSTNESS_NOISE_LEVELS: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1])  # Noise levels for robustness tests
    MULTI_SCALE_SIZES: List[int] = field(default_factory=lambda: [224, 336, 448])  # Different image sizes for multi-scale
    
    # Advanced evaluation settings
    ENABLE_CROSS_TOKEN_GROUNDING_CONSISTENCY: bool = True  # Check for non-overlapping token regions
    ENABLE_DELETION_INSERTION_TEST: bool = True  # Caption faithfulness via deletion-insertion
    ENABLE_STAIN_NORMALIZATION_ROBUSTNESS: bool = True  # Pathology-specific robustness
    ENABLE_MULTI_SCALE_TOKEN_GROUNDING: bool = True  # Token grounding across magnifications
    ENABLE_FUZZY_LOGIC_ABLATION: bool = True  # Ablation study of fuzzy logic components
    DELETION_PERCENTAGES: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.5])  # Percentages of pixels to delete


# Global configuration instance
cfg = ConfigPaths()

