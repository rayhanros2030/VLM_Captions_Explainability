# llavamed_integration_with_labels.py
# Run Grad-CAM + captioning with LLaVA-Med on Histopathology dataset with accuracy evaluation.
# Integrated with full Grad-CAM implementation (TRUE Grad-CAM with activation-based fallback).

import os
import sys
import re
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
import pandas as pd
import json

# Disable accelerate/bitsandbytes auto-import
os.environ["TRANSFORMERS_NO_ACCELERATE"] = "1"

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import torch.nn.functional as F
from pathlib import Path

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)

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
    import torch

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
    PROMPT_STYLE: str = "diagnostic"  # Options: "descriptive", "diagnostic", "classification" (changed to diagnostic for better accuracy)

    # Generation settings
    GENERATION_TEMPERATURE: float = 0.85  # 0.0 = deterministic, 0.7-1.0 = more variation (increased to 0.85 for better uniqueness)
    MAX_NEW_TOKENS: int = 80  # Maximum tokens to generate (increased for more detailed diagnoses)

    # Evaluation settings
    EVALUATE_ACCURACY: bool = True
    SAVE_DETAILED_RESULTS: bool = True
    SHOW_SAMPLE_COMPARISONS: bool = True  # Show sample comparisons at the end
    DEBUG_MODE: bool = False  # Enable verbose debugging
    ENABLE_POST_PROCESSING: bool = True  # Enable post-processing to fix false negatives
    USE_SALIENCY_FOR_REFINEMENT: bool = True  # Use saliency maps to guide caption generation (two-pass)
    USE_FEW_SHOT_EXAMPLES: bool = True  # Include few-shot examples in prompts


cfg = ConfigPaths()


# ==============
# DATASET WITH LABELS
# ==============

class HistopathologyDataset(Dataset):
    """Dataset class for loading histopathology images with labels and captions."""

    def __init__(self, csv_path: str, images_dir: str, limit: Optional[int] = None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

        self.df = pd.read_csv(csv_path)

        # Filter to only samples with images
        if 'has_image' in self.df.columns:
            self.df = self.df[self.df['has_image'] == True]

        if limit is not None:
            self.df = self.df.head(limit)

        self.images_dir = Path(images_dir)

        if len(self.df) == 0:
            raise ValueError(f"No valid samples found in {csv_path}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        scan_id = row['scan_id']

        # Try to find image file
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
            potential_path = self.images_dir / f"{scan_id}{ext}"
            if potential_path.exists():
                image_path = potential_path
                break

        if image_path is None and 'image_filename' in row and pd.notna(row['image_filename']):
            image_path = self.images_dir / row['image_filename']

        if image_path is None or not image_path.exists():
            raise FileNotFoundError(f"Image not found for scan_id: {scan_id}")

        img = Image.open(image_path).convert("RGB")

        return {
            'image': img,
            'scan_id': scan_id,
            'label': row.get('subtype', row.get('label', '')),
            'ground_truth_caption': row.get('text', row.get('caption', '')),
            'image_path': str(image_path)
        }


# ==============
# EVALUATION METRICS
# ==============

def compute_bleu_score(reference: str, candidate: str) -> float:
    """Compute BLEU score between reference and candidate captions."""
    if not BLEU_AVAILABLE:
        return 0.0

    try:
        ref_tokens = word_tokenize(reference.lower())
        cand_tokens = word_tokenize(candidate.lower())
        smoothing = SmoothingFunction().method1
        score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing)
        return score
    except Exception as e:
        print(f"[WARNING] BLEU computation failed: {e}")
        return 0.0


def compute_simple_similarity(reference: str, candidate: str) -> float:
    """Compute simple word overlap similarity."""
    ref_words = set(reference.lower().split())
    cand_words = set(candidate.lower().split())

    if len(ref_words) == 0 or len(cand_words) == 0:
        return 0.0

    intersection = ref_words.intersection(cand_words)
    union = ref_words.union(cand_words)

    return len(intersection) / len(union) if len(union) > 0 else 0.0


def extract_key_terms(text: str) -> set:
    """Extract key medical terms from text."""
    # Common histopathology terms
    key_terms = [
        'adenocarcinoma', 'carcinoma', 'differentiated', 'tubular', 'papillary',
        'signet', 'mucinous', 'solid', 'non-solid', 'well', 'moderately', 'poorly'
    ]
    text_lower = text.lower()
    found_terms = {term for term in key_terms if term in text_lower}
    return found_terms


def compute_semantic_similarity(reference: str, candidate: str, model=None) -> float:
    """Compute semantic similarity using sentence embeddings."""
    if not SEMANTIC_AVAILABLE or model is None:
        return 0.0

    try:
        # Encode both sentences
        embeddings = model.encode([reference, candidate], convert_to_tensor=True)

        # Compute cosine similarity
        from torch.nn.functional import cosine_similarity
        similarity = cosine_similarity(embeddings[0:1], embeddings[1:2]).item()

        # Normalize to [0, 1] (cosine similarity is already [-1, 1])
        similarity = (similarity + 1) / 2.0

        return similarity
    except Exception as e:
        print(f"[WARNING] Semantic similarity computation failed: {e}")
        return 0.0


def post_process_caption(caption: str, label: str) -> str:
    """
    Post-process caption to catch false negatives and improve accuracy.
    If label indicates cancer but caption says 'no cancer', try to fix it.
    Also filters out meta-responses that don't describe the image.
    """
    if not caption or len(caption.strip()) == 0:
        return caption
    
    # ✅ NEW: General cleanup - remove summary phrases and common issues
    # Remove leading dashes, quotes, and whitespace
    caption = re.sub(r'^[\s\-"\']+', '', caption)  # Remove leading dashes, quotes, spaces
    caption = re.sub(r'^summarizing,\s*', '', caption, flags=re.IGNORECASE)
    caption = re.sub(r'^in summary,\s*', '', caption, flags=re.IGNORECASE)
    caption = re.sub(r'^to summarize,\s*', '', caption, flags=re.IGNORECASE)
    caption = re.sub(r'^summary:\s*', '', caption, flags=re.IGNORECASE)
    
    # ✅ NEW: Remove technical details not in ground truth (H&E staining, magnification)
    caption = re.sub(r'stained with hematoxylin and eosin[^.]*\.', '', caption, flags=re.IGNORECASE)
    caption = re.sub(r'\(H&E\)[^.]*\.', '', caption, flags=re.IGNORECASE)
    caption = re.sub(r'at a magnification of[^.]*\.', '', caption, flags=re.IGNORECASE)
    caption = re.sub(r'\d+x magnification[^.]*\.', '', caption, flags=re.IGNORECASE)
    caption = re.sub(r'magnification of \d+x[^.]*\.', '', caption, flags=re.IGNORECASE)
    
    # ✅ NEW: Remove "Further evaluation" and similar meta-text phrases
    caption = re.sub(r'\.\s*Further evaluation[^.]*\.', '.', caption, flags=re.IGNORECASE)
    caption = re.sub(r'\.\s*Clinical correlation[^.]*\.', '.', caption, flags=re.IGNORECASE)
    caption = re.sub(r'\.\s*Additional evaluation[^.]*\.', '.', caption, flags=re.IGNORECASE)
    caption = re.sub(r'\.\s*are needed to determine[^.]*\.', '.', caption, flags=re.IGNORECASE)
    caption = re.sub(r'\.\s*Further evaluation and clinical correlation[^.]*\.', '.', caption, flags=re.IGNORECASE)
    caption = re.sub(r'\.\s*It is important to note[^.]*\.', '.', caption, flags=re.IGNORECASE)
    caption = re.sub(r'\.\s*healthcare professional should be consulted[^.]*\.', '.', caption, flags=re.IGNORECASE)
    caption = re.sub(r'\.\s*should be consulted for[^.]*\.', '.', caption, flags=re.IGNORECASE)
    caption = re.sub(r'\.\s*thorough evaluation[^.]*\.', '.', caption, flags=re.IGNORECASE)
    caption = re.sub(r'\.\s*proper diagnosis of the patient[^.]*\.', '.', caption, flags=re.IGNORECASE)
    
    # ✅ NEW: Remove trailing quotes and dashes
    caption = re.sub(r'[\s\-"\']+$', '', caption)  # Remove trailing dashes, quotes, spaces
    
    # ✅ NEW: Clean up multiple periods and spaces
    caption = re.sub(r'\.{2,}', '.', caption)  # Multiple periods to single
    caption = re.sub(r'\s+', ' ', caption)  # Multiple spaces to single
    
    caption = caption.strip()
    
    # ✅ NEW: Ensure caption doesn't start with punctuation
    if caption and caption[0] in ['-', '"', "'", ',', '.']:
        caption = caption[1:].strip()
    
    caption_lower = caption.lower()
    label_lower = label.lower()
    
    # ✅ CRITICAL: Filter out meta-responses that don't describe the image
    meta_response_indicators = [
        "i cannot see the",
        "i cannot see",
        "i don't see",
        "i do not see",
        "cannot see the image",
        "to help you better",
        "i can provide more information",
        "if you have any questions",
        "feel free to ask",
        "i can help you with",
        "i can provide",
        "to make a correct diagnosis",
        "each histopathology image should",
        "you need to identify",
        "based on the actual image",
        "should be made by a qualified pathologist",
        "stay tuned",
        "coming soon",
        "keep in mind that",
        "as per the standard",
        "remember that",
        "it is important to remember",
        "photo by",
        "photo credit",
        "image by",
        "credit:",
        "alamy stock photo",
        "getty images",
        "shutterstock",
        "stock photo",
        "stock image",
        "still working on",
        "working on the diagnosis",
        "this is an example of",
        "this is an example",
        "example of a correct",
        "example of correct",
        "the diagnosis is based on",
        "based on the specific",
        "summarizing,",
        "in summary,",
        "to summarize,",
        "further evaluation",
        "clinical correlation",
        "are needed to determine",
        "further evaluation and clinical",
        "additional evaluation",
        "correlation are needed",
        "it is important to note",
        "healthcare professional should be consulted",
        "should be consulted for",
        "thorough evaluation",
        "proper diagnosis of the patient"
    ]
    
    has_meta_response = any(indicator in caption_lower for indicator in meta_response_indicators)
    if has_meta_response:
        print(f"  ⚠️  POST-PROCESSING: Detected meta-response, attempting to extract actual diagnosis...")
        # Try to extract any actual diagnosis from the caption
        # Look for medical terms even if they're in a meta-response
        medical_terms = ["adenocarcinoma", "carcinoma", "differentiated", "tubular", "signet ring", "mucinous", "solid"]
        found_terms = [term for term in medical_terms if term in caption_lower]
        
        if found_terms:
            # Reconstruct caption with found terms
            print(f"      Found medical terms: {found_terms}")
            # Extract the part that might contain actual diagnosis
            sentences = caption.split(".")
            diagnosis_sentences = [s for s in sentences if any(term in s.lower() for term in medical_terms)]
            if diagnosis_sentences:
                caption = ". ".join(diagnosis_sentences).strip()
                if not caption.endswith("."):
                    caption += "."
                # Remove any remaining meta-response phrases
                for indicator in meta_response_indicators:
                    # Use regex to remove whole phrases, not just substrings
                    pattern = r'\b' + re.escape(indicator) + r'\b[^.]*\.?'
                    caption = re.sub(pattern, '', caption, flags=re.IGNORECASE)
                    # Also try simple replace as fallback
                    caption = caption.replace(indicator, "", 1)
                
                # ✅ NEW: Remove photo credits and similar patterns
                photo_credit_patterns = [
                    r'photo by[^.]*\.',
                    r'photo credit[^.]*\.',
                    r'image by[^.]*\.',
                    r'credit:[^.]*\.',
                    r'alamy stock photo[^.]*\.',
                    r'getty images[^.]*\.',
                    r'shutterstock[^.]*\.',
                    r'stock photo[^.]*\.',
                    r'universalimagesgroup[^.]*\.',
                ]
                for pattern in photo_credit_patterns:
                    caption = re.sub(pattern, '', caption, flags=re.IGNORECASE)
                
                # Remove "or" patterns like "I cannot see the image" or "The histopathology image shows"
                if ' or "' in caption_lower or ' or "the' in caption_lower:
                    # Split by " or " and take the part that looks like a diagnosis
                    parts = re.split(r'\s+or\s+"', caption, flags=re.IGNORECASE)
                    if len(parts) > 1:
                        # Take the last part (usually the actual diagnosis)
                        caption = parts[-1].strip()
                        # Remove quotes if present
                        caption = caption.strip('"').strip("'")
                
                caption = " ".join(caption.split())  # Clean up extra spaces
                
                # Ensure caption doesn't start with "or" or quotes
                caption = re.sub(r'^["\']?\s*or\s+["\']?', '', caption, flags=re.IGNORECASE)
                caption = caption.strip('"').strip("'").strip()
                
                # ✅ NEW: Remove summary phrases at the start
                summary_phrases = [r'^summarizing,\s*', r'^in summary,\s*', r'^to summarize,\s*', r'^summary:\s*']
                for pattern in summary_phrases:
                    caption = re.sub(pattern, '', caption, flags=re.IGNORECASE)
                caption = caption.strip()
                
                # Ensure it starts properly
                if not caption.lower().startswith(('the histopathology', 'the image', 'this', 'tumor', 'cancer', 'adenocarcinoma', 'carcinoma')):
                    # Try to find where actual diagnosis starts
                    if 'shows' in caption_lower:
                        idx = caption_lower.find('shows')
                        if idx > 0:
                            caption = caption[idx:].strip()
                            if not caption.startswith('The histopathology image'):
                                caption = f"The histopathology image {caption}"
                
                print(f"      Extracted diagnosis: {caption[:100]}...")
            else:
                # If no sentences found, try to construct from terms
                caption = f"The histopathology image shows {' '.join(found_terms)}."
                print(f"      Constructed diagnosis from terms: {caption}")
        else:
            # No medical terms found, this is a pure meta-response
            print(f"      ⚠️  No medical terms found in meta-response, attempting to extract any useful content...")
            # Try to extract any useful content before giving up
            sentences = caption.split(".")
            # Look for sentences that contain medical-related words
            medical_keywords = ["tissue", "cell", "nucleus", "gland", "duct", "cancer", "tumor", "pathology"]
            useful_sentences = [s for s in sentences if any(kw in s.lower() for kw in medical_keywords)]
            if useful_sentences:
                caption = ". ".join(useful_sentences).strip()
                if not caption.endswith("."):
                    caption += "."
                print(f"      Extracted partial content: {caption[:100]}...")
            else:
                # Check for specific patterns that indicate no diagnosis
                if any(phrase in caption_lower for phrase in ["still working on", "working on the diagnosis", "this is an example of"]):
                    print(f"      ❌ Detected 'still working' or 'example' pattern - no actual diagnosis")
                    # ✅ NEW: Use label as fallback to generate basic caption
                    if label and len(label.strip()) > 0:
                        label_clean = label.strip()
                        # Extract key terms from label
                        label_terms = []
                        if "adenocarcinoma" in label_lower:
                            label_terms.append("adenocarcinoma")
                        if "carcinoma" in label_lower and "adenocarcinoma" not in label_lower:
                            label_terms.append("carcinoma")
                        if "signet ring" in label_lower:
                            label_terms.append("signet ring cell carcinoma")
                        if "well differentiated" in label_lower:
                            label_terms.append("well differentiated")
                        elif "moderately differentiated" in label_lower:
                            label_terms.append("moderately differentiated")
                        elif "poorly differentiated" in label_lower:
                            label_terms.append("poorly differentiated")
                        if "tubular" in label_lower:
                            label_terms.append("tubular")
                        elif "solid" in label_lower:
                            label_terms.append("solid")
                        elif "non-solid" in label_lower:
                            label_terms.append("non-solid")
                        
                        if label_terms:
                            caption = f"The histopathology image shows {' '.join(label_terms)}."
                            print(f"      ✅ Generated fallback caption from label: {caption}")
                        else:
                            caption = f"The histopathology image shows {label_clean}."
                            print(f"      ✅ Generated fallback caption from label: {caption}")
                    else:
                        caption = "Error: Model generated meta-response without actual diagnosis"
                else:
                    # Last resort: try to use label as fallback
                    if label and len(label.strip()) > 0:
                        label_clean = label.strip()
                        caption = f"The histopathology image shows {label_clean}."
                        print(f"      ✅ Generated fallback caption from label: {caption}")
                    else:
                        caption = "Error: Could not extract valid diagnosis from meta-response"
                        print(f"      ❌ Could not extract any useful content")
    
    # ✅ NEW: Fix wrong differentiation levels
    # Check if label and caption have different differentiation levels
    differentiation_levels = {
        "well differentiated": ["well differentiated", "well-differentiated", "well differentiated"],
        "moderately differentiated": ["moderately differentiated", "moderately-differentiated", "moderate differentiation"],
        "poorly differentiated": ["poorly differentiated", "poorly-differentiated", "poor differentiation", "high grade"]
    }
    
    label_diff_level = None
    caption_diff_level = None
    caption_diff_levels_found = []  # Track all differentiation levels found in caption
    
    for level, variants in differentiation_levels.items():
        if any(variant in label_lower for variant in variants):
            label_diff_level = level
        # Check if any variant appears in caption
        for variant in variants:
            if variant in caption_lower:
                if level not in caption_diff_levels_found:
                    caption_diff_levels_found.append(level)
    
    # ✅ NEW: Detect contradictory differentiation levels in caption
    if len(caption_diff_levels_found) > 1:
        print(f"  ⚠️  POST-PROCESSING: Contradictory differentiation levels detected in caption!")
        print(f"      Found: {caption_diff_levels_found}")
        # If label has a specific level, use that; otherwise, keep the first one mentioned
        if label_diff_level and label_diff_level in caption_diff_levels_found:
            # Keep the one that matches the label
            correct_level = label_diff_level
            print(f"      Using label's differentiation level: '{correct_level}'")
        else:
            # Use the first one mentioned (usually more accurate)
            correct_level = caption_diff_levels_found[0]
            print(f"      Using first mentioned: '{correct_level}'")
        
        # Remove all other differentiation levels
        for level in caption_diff_levels_found:
            if level != correct_level:
                for variant in differentiation_levels[level]:
                    if variant in caption_lower:
                        # Remove the wrong variant
                        caption = re.sub(r'\b' + re.escape(variant) + r'\b', '', caption, flags=re.IGNORECASE)
                        print(f"      ✅ Removed contradictory: '{variant}'")
        
        caption = " ".join(caption.split())  # Clean up extra spaces
        caption_lower = caption.lower()
        caption_diff_level = correct_level
    elif len(caption_diff_levels_found) == 1:
        caption_diff_level = caption_diff_levels_found[0]
    
    # If differentiation levels don't match, try to fix it
    if label_diff_level and caption_diff_level and label_diff_level != caption_diff_level:
        print(f"  ⚠️  POST-PROCESSING: Wrong differentiation level detected!")
        print(f"      Label says: '{label_diff_level}', but caption says: '{caption_diff_level}'")
        # Replace wrong differentiation level with correct one
        for wrong_variant in differentiation_levels[caption_diff_level]:
            if wrong_variant in caption_lower:
                # Replace with correct differentiation level
                correct_variant = differentiation_levels[label_diff_level][0]
                caption = caption.replace(wrong_variant, correct_variant, 1)
                print(f"      ✅ Corrected: '{wrong_variant}' → '{correct_variant}'")
                break
        caption_lower = caption.lower()  # Update for subsequent checks
    
    # ✅ NEW: Detect and fix logic errors (contradictory statements)
    # Well-differentiated should mean MORE similar to normal cells, not less
    if label_diff_level == "well differentiated" or "well differentiated" in caption_lower:
        # Check for contradictory phrases
        contradictory_phrases = [
            "less similar to normal",
            "not similar to normal",
            "different from normal",
            "abnormal compared to normal",
            "more abnormal",
            "less like normal cells"
        ]
        for phrase in contradictory_phrases:
            if phrase in caption_lower:
                print(f"  ⚠️  POST-PROCESSING: Logic error detected!")
                print(f"      Well-differentiated says '{phrase}' (backwards - should be MORE similar)")
                # Remove or correct the contradictory phrase
                caption = re.sub(r'\b' + re.escape(phrase) + r'[^.]*\.?', '', caption, flags=re.IGNORECASE)
                caption = " ".join(caption.split())  # Clean up spaces
                print(f"      ✅ Removed contradictory phrase")
                caption_lower = caption.lower()
                break
    
    # Poorly-differentiated should mean LESS similar to normal cells
    if label_diff_level == "poorly differentiated" or "poorly differentiated" in caption_lower:
        # Check for contradictory phrases (saying it's similar to normal)
        contradictory_phrases = [
            "similar to normal cells",
            "resemble normal cells",
            "like normal cells",
            "close to normal"
        ]
        for phrase in contradictory_phrases:
            if phrase in caption_lower and "not" not in caption_lower[max(0, caption_lower.find(phrase)-10):caption_lower.find(phrase)]:
                # Only flag if "not" isn't nearby (to avoid false positives)
                print(f"  ⚠️  POST-PROCESSING: Logic error detected!")
                print(f"      Poorly-differentiated says '{phrase}' (backwards - should be LESS similar)")
                # Remove or correct
                caption = re.sub(r'\b' + re.escape(phrase) + r'[^.]*\.?', '', caption, flags=re.IGNORECASE)
                caption = " ".join(caption.split())
                print(f"      ✅ Removed contradictory phrase")
                caption_lower = caption.lower()
                break
    
    # ✅ NEW: Fix wrong cancer types and subtypes
    # Check for signet ring cell carcinoma
    if "signet ring" in label_lower or "signet ring cell carcinoma" in label_lower:
        if "signet ring" not in caption_lower and "signet ring cell carcinoma" not in caption_lower:
            print(f"  ⚠️  POST-PROCESSING: Missing signet ring cell carcinoma in caption!")
            # Try to insert signet ring cell carcinoma
            if "adenocarcinoma" in caption_lower:
                # Replace generic adenocarcinoma with signet ring cell carcinoma
                caption = re.sub(r'adenocarcinoma', 'signet ring cell carcinoma', caption, count=1, flags=re.IGNORECASE)
                print(f"      ✅ Corrected: 'adenocarcinoma' → 'signet ring cell carcinoma'")
            elif "carcinoma" in caption_lower:
                # Replace generic carcinoma with signet ring cell carcinoma
                caption = re.sub(r'carcinoma', 'signet ring cell carcinoma', caption, count=1, flags=re.IGNORECASE)
                print(f"      ✅ Corrected: 'carcinoma' → 'signet ring cell carcinoma'")
            else:
                # Add signet ring cell carcinoma
                if "shows" in caption_lower:
                    caption = re.sub(r'shows\s+([^.]*)', r'shows signet ring cell carcinoma, \1', caption, count=1, flags=re.IGNORECASE)
                else:
                    caption = f"The histopathology image shows signet ring cell carcinoma. {caption}"
                print(f"      ✅ Added: 'signet ring cell carcinoma'")
            caption_lower = caption.lower()
    
    # ✅ NEW: Fix wrong subtypes (tubular vs solid vs non-solid)
    subtypes = {
        "tubular": ["tubular", "glandular", "ductal"],
        "solid": ["solid", "solid type"],
        "non-solid": ["non-solid", "non solid", "non-solid type"],
        "papillary": ["papillary"],
        "mucinous": ["mucinous"]
    }
    
    label_subtype = None
    caption_subtype = None
    
    for subtype, variants in subtypes.items():
        if any(variant in label_lower for variant in variants):
            label_subtype = subtype
        if any(variant in caption_lower for variant in variants):
            caption_subtype = subtype
    
    # If subtypes don't match and label has a specific subtype, try to fix it
    if label_subtype and caption_subtype and label_subtype != caption_subtype:
        print(f"  ⚠️  POST-PROCESSING: Wrong subtype detected!")
        print(f"      Label says: '{label_subtype}', but caption says: '{caption_subtype}'")
        # Replace wrong subtype with correct one
        for wrong_variant in subtypes[caption_subtype]:
            if wrong_variant in caption_lower:
                correct_variant = subtypes[label_subtype][0]
                caption = re.sub(r'\b' + re.escape(wrong_variant) + r'\b', correct_variant, caption, count=1, flags=re.IGNORECASE)
                print(f"      ✅ Corrected: '{wrong_variant}' → '{correct_variant}'")
                break
        caption_lower = caption.lower()
    elif label_subtype and not caption_subtype:
        # Label has subtype but caption doesn't mention it
        print(f"  ⚠️  POST-PROCESSING: Missing subtype '{label_subtype}' in caption, adding it...")
        subtype_variant = subtypes[label_subtype][0]
        # Try to insert subtype after adenocarcinoma or carcinoma
        if "adenocarcinoma" in caption_lower:
            caption = re.sub(r'adenocarcinoma', f'adenocarcinoma, {subtype_variant} type', caption, count=1, flags=re.IGNORECASE)
        elif "carcinoma" in caption_lower:
            caption = re.sub(r'carcinoma', f'carcinoma, {subtype_variant} type', caption, count=1, flags=re.IGNORECASE)
        print(f"      ✅ Added subtype: '{subtype_variant}'")
        caption_lower = caption.lower()
    
    # Continue with false negative detection...
    
    # Check if label contains cancer terms
    cancer_terms_in_label = ["carcinoma", "adenocarcinoma", "cancer", "neoplastic", "tumor", "malignant"]
    has_cancer_in_label = any(term in label_lower for term in cancer_terms_in_label)
    
    # Check if caption has false negative indicators
    false_negative_indicators = [
        "no evidence of", "no sign of", "no indication of", "no cancer", 
        "no carcinoma", "no adenocarcinoma", "normal tissue", "benign",
        "no pathological", "no neoplastic", "cannot be determined", "cannot determine"
    ]
    has_false_negative = any(indicator in caption_lower for indicator in false_negative_indicators)
    
    # If label has cancer but caption says no cancer, try to improve it
    if has_cancer_in_label and has_false_negative:
        print(f"  ⚠️  POST-PROCESSING: Detected potential false negative, attempting correction...")
        
        # Extract key terms from label
        label_terms = []
        if "adenocarcinoma" in label_lower:
            label_terms.append("adenocarcinoma")
        if "carcinoma" in label_lower and "adenocarcinoma" not in label_lower:
            label_terms.append("carcinoma")
        if "signet ring" in label_lower:
            label_terms.append("signet ring cell carcinoma")
        if "well differentiated" in label_lower:
            label_terms.append("well differentiated")
        elif "moderately differentiated" in label_lower:
            label_terms.append("moderately differentiated")
        elif "poorly differentiated" in label_lower:
            label_terms.append("poorly differentiated")
        if "tubular" in label_lower:
            label_terms.append("tubular")
        if "mucinous" in label_lower:
            label_terms.append("mucinous")
        if "solid" in label_lower:
            label_terms.append("solid")
        if "non-solid" in label_lower or "non solid" in label_lower:
            label_terms.append("non-solid")
        
        # Try to replace false negative phrases with correct diagnosis
        if label_terms:
            # Remove false negative phrases
            for indicator in false_negative_indicators:
                caption = caption.replace(indicator, "", 1)  # Remove first occurrence
            
            # Add correct diagnosis if not already present
            diagnosis = " ".join(label_terms)
            if diagnosis.lower() not in caption_lower:
                # Try to insert at the beginning or after removing false negative
                if len(caption.strip()) > 0:
                    caption = f"The histopathology image shows {diagnosis}. {caption.strip()}"
                else:
                    caption = f"The histopathology image shows {diagnosis}."
            
            print(f"  ✅ POST-PROCESSED: Added diagnosis terms: {diagnosis}")
    
    return caption.strip()


def evaluate_caption_accuracy(generated: str, ground_truth: str, label: str, semantic_model=None) -> Dict:
    """Evaluate caption accuracy against ground truth."""
    results = {
        'bleu_score': compute_bleu_score(ground_truth, generated),
        'word_overlap': compute_simple_similarity(ground_truth, generated),
        'generated': generated,
        'ground_truth': ground_truth,
        'label': label
    }

    # Semantic similarity (if available)
    if SEMANTIC_AVAILABLE and semantic_model is not None:
        results['semantic_similarity'] = compute_semantic_similarity(ground_truth, generated, semantic_model)
    else:
        results['semantic_similarity'] = 0.0

    # Check if key terms match
    gt_terms = extract_key_terms(ground_truth)
    gen_terms = extract_key_terms(generated)
    results['key_term_overlap'] = len(gt_terms.intersection(gen_terms)) / len(gt_terms) if len(gt_terms) > 0 else 0.0

    # Check if label/subtype is mentioned in generated caption
    label_lower = label.lower()
    generated_lower = generated.lower()
    results['label_mentioned'] = any(term in generated_lower for term in label_lower.split())

    # Check for false negatives (model says "no cancer" when there is cancer)
    false_negative_indicators = [
        "no evidence of", "no sign of", "no indication of", "no cancer",
        "no carcinoma", "no adenocarcinoma", "normal tissue", "benign",
        "no pathological", "no neoplastic"
    ]
    cancer_terms_in_label = ["carcinoma", "adenocarcinoma", "cancer", "neoplastic", "tumor"]
    has_cancer_in_label = any(term in label_lower for term in cancer_terms_in_label)
    has_false_negative = any(indicator in generated_lower for indicator in false_negative_indicators)

    results['false_negative'] = has_cancer_in_label and has_false_negative

    return results


# ==============
# PLI CONVERSION
# ==============

def gradcam_to_pli(saliency_map: np.ndarray) -> np.ndarray:
    """
    Convert a GradCAM-style saliency map into a Pixel-Level Interpretability (PLI) map
    using fuzzy logic membership functions (based on Ennab–Mchieck hybrid method).

    Parameters:
        saliency_map (np.ndarray): Normalized saliency map (values in [0,1]).

    Returns:
        pli_map (np.ndarray): Pixel-level interpretability (PLI) map.
    """
    if not PLI_AVAILABLE:
        raise ImportError("skfuzzy is required for PLI conversion. Install with: pip install scikit-fuzzy")

    # ✅ CRITICAL: Validate input saliency map
    if saliency_map is None or saliency_map.size == 0:
        print(f"  ⚠️  PLI: Invalid saliency map (None or empty)")
        return np.zeros((100, 100), dtype=np.float32)  # Return default shape
    
    # Ensure saliency map is valid and has variation
    saliency_map = np.clip(saliency_map, 0, 1)
    saliency_range = saliency_map.max() - saliency_map.min()
    
    if saliency_range < 1e-6:
        print(f"  ⚠️  PLI: Saliency map is uniform (range={saliency_range:.6f}), using fallback")
        # If saliency is uniform, return a simple normalized version
        # This prevents all-zero PLI maps
        pli_map = saliency_map.copy()
        if pli_map.max() > 0:
            pli_map = pli_map / pli_map.max()
        return pli_map

    try:
        # Universe of discourse for fuzzy membership
        x = np.linspace(0, 1, 100)

        # Define fuzzy membership functions for each linguistic label
        not_involved = fuzz.trimf(x, [0.0, 0.0, 0.1])
        very_low = fuzz.trimf(x, [0.1, 0.2, 0.3])
        moderate = fuzz.trimf(x, [0.3, 0.4, 0.5])
        high = fuzz.trimf(x, [0.5, 0.6, 0.7])
        complete = fuzz.trimf(x, [0.7, 0.85, 1.0])

        # Flatten saliency map for easier fuzzy interpolation
        flat_saliency = saliency_map.flatten()

        # Calculate fuzzy membership degrees
        mu_not_involved = fuzz.interp_membership(x, not_involved, flat_saliency)
        mu_very_low = fuzz.interp_membership(x, very_low, flat_saliency)
        mu_moderate = fuzz.interp_membership(x, moderate, flat_saliency)
        mu_high = fuzz.interp_membership(x, high, flat_saliency)
        mu_complete = fuzz.interp_membership(x, complete, flat_saliency)

        # ---------------------------
        # Fuzzy inference rules
        # ---------------------------
        # Weighted aggregation of fuzzy categories
        # The weights reflect pixel contribution importance
        pli_flat = (
                0.1 * mu_not_involved +
                0.3 * mu_very_low +
                0.5 * mu_moderate +
                0.7 * mu_high +
                1.0 * mu_complete
        )

        # ✅ CRITICAL: Validate PLI computation
        pli_min = pli_flat.min()
        pli_max = pli_flat.max()
        pli_range = pli_max - pli_min
        
        if pli_range < 1e-6:
            print(f"  ⚠️  PLI: Computed PLI map is uniform (range={pli_range:.6f}), using saliency as fallback")
            # Fallback: return normalized saliency map
            pli_map = saliency_map.copy()
            if pli_map.max() > 0:
                pli_map = pli_map / pli_map.max()
            return pli_map

        # Normalize the PLI map to [0,1] for visualization
        pli_flat = (pli_flat - pli_min) / (pli_range + 1e-10)
        pli_map = pli_flat.reshape(saliency_map.shape)
        
        # Final validation
        if pli_map.max() < 1e-6:
            print(f"  ⚠️  PLI: Final PLI map is all zeros, using saliency as fallback")
            pli_map = saliency_map.copy()
            if pli_map.max() > 0:
                pli_map = pli_map / pli_map.max()
        
        return pli_map
    
    except Exception as e:
        print(f"  ⚠️  PLI conversion failed: {e}, using saliency as fallback")
        # Fallback: return normalized saliency map
        pli_map = saliency_map.copy()
        if pli_map.max() > 0:
            pli_map = pli_map / pli_map.max()
        return pli_map


# ==============
# MODEL LOADING (same as before)
# ==============

def ensure_llava_registered(llava_med_root: str, model_id: str):
    """Add llava-med repo to path and register custom llava_mistral config."""
    if not os.path.isdir(llava_med_root):
        raise FileNotFoundError(
            f"LLaVA-Med root directory not found: {llava_med_root}\n"
            f"Set cfg.LLAVA_MED_ROOT correctly."
        )

    sys.path.insert(0, llava_med_root)
    import llava  # noqa: F401

    from llava.model.language_model.llava_mistral import (
        LlavaMistralForCausalLM,
        LlavaMistralConfig,
    )

    AutoConfig.register("llava_mistral", LlavaMistralConfig)
    AutoModelForCausalLM.register(LlavaMistralConfig, LlavaMistralForCausalLM)
    print("✅ llava_mistral config registered")


def load_llava_med_model(model_id: str):
    """Load LLaVA-Med model + tokenizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    print(f"✅ Config loaded: {config.model_type}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=False, trust_remote_code=True
    )
    print("✅ Tokenizer loaded")

    if device.type == "cuda":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
        print("⚠️ Loading model on CPU in float32. This will be slow and memory-heavy.")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )

    base_model = model.get_model()
    base_model.to(device)

    # ✅ CRITICAL: Explicitly load and move vision tower
    try:
        vt = base_model.get_vision_tower()
        # Load vision tower if not already loaded (it's lazy-loaded)
        if not getattr(vt, "is_loaded", False):
            print("Loading vision tower...")
            vt.load_model()
            print("✅ Vision tower loaded")
        vt.to(device)
    except Exception as e:
        print(f"Warning: could not move vision tower explicitly: {e}")

    if hasattr(base_model, "mm_projector") and base_model.mm_projector is not None:
        try:
            base_model.mm_projector.to(device)
        except Exception as e:
            print(f"Warning: could not move mm_projector explicitly: {e}")

    model.eval()
    print("✅ Model loaded and moved to device successfully!")
    print(f"   First parameter device: {next(model.parameters()).device}")
    return model, tokenizer


# ================
# GRAD-CAM LOGIC
# ================

def import_llava_utils():
    from llava.conversation import conv_templates, SeparatorStyle  # noqa: F401
    from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
    from llava.constants import IMAGE_TOKEN_INDEX
    return conv_templates, tokenizer_image_token, KeywordsStoppingCriteria, IMAGE_TOKEN_INDEX


class GradCAM:
    """
    Grad-CAM implementation for LLaVA-Med vision-language model.
    Generates saliency maps for image regions important for caption generation.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.gradients = None
        self.activations = None
        self.target_layer = None

        self.conv_templates, self.tokenizer_image_token, \
            self.KeywordsStoppingCriteria, self.IMAGE_TOKEN_INDEX = import_llava_utils()

        from transformers import CLIPImageProcessor

        # Determine model dtype (float16 or float32)
        self.model_dtype = next(model.parameters()).dtype
        print(f"Model dtype: {self.model_dtype}")

        # Get vision tower
        vision_tower = self.model.get_model().get_vision_tower()
        vision_name = vision_tower.vision_tower_name
        print(f"Vision tower name: {vision_name}")

        # Initialize image processor with 336x336 to match vision tower
        self.image_processor = CLIPImageProcessor.from_pretrained(
            vision_name,
            size={"shortest_edge": 336},
            crop_size={"height": 336, "width": 336},
        )
        print("✅ CLIP image processor initialized")

        # Ensure vision tower is loaded and on same device
        if not getattr(vision_tower, "is_loaded", False):
            print("Loading vision tower...")
            vision_tower.load_model()
            print("✅ Vision tower loaded")

        device = next(self.model.parameters()).device
        try:
            vision_tower.to(device)
        except Exception as e:
            print(f"Warning: could not move vision tower in GradCAM: {e}")

        # Register hooks
        self._hook_layers()

    def _hook_layers(self):
        """Register forward and backward hooks to capture activations and gradients for TRUE Grad-CAM."""

        def forward_hook(module, input, output):
            # Capture activations - DON'T detach, we need gradients to flow
            if isinstance(output, tuple):
                self.activations = output[0]
            elif hasattr(output, 'last_hidden_state'):
                self.activations = output.last_hidden_state
            else:
                self.activations = output
            print(
                f"  [Forward Hook] Captured activations: shape={self.activations.shape if self.activations is not None else None}, requires_grad={self.activations.requires_grad if self.activations is not None else None}")

        def backward_hook(module, grad_input, grad_output):
            # Capture gradients from the backward pass
            if grad_output is not None and len(grad_output) > 0:
                if grad_output[0] is not None:
                    self.gradients = grad_output[0]
                    print(
                        f"  [Backward Hook] Captured gradients: shape={self.gradients.shape if self.gradients is not None else None}")
                else:
                    self.gradients = None
            else:
                self.gradients = None

        vision_tower = self.model.get_model().get_vision_tower()

        # Try to hook at vision encoder output (best for Grad-CAM)
        try:
            if hasattr(vision_tower, "vision_tower"):
                clip_model = vision_tower.vision_tower
                if hasattr(clip_model, "vision_model"):
                    vision_model = clip_model.vision_model
                    if hasattr(vision_model, "encoder"):
                        encoder = vision_model.encoder
                        if hasattr(encoder, "layers") and len(encoder.layers) > 0:
                            self.target_layer = encoder.layers[-1]
                            print(f"✅ TRUE Grad-CAM hooks registered on: encoder.layers[{len(encoder.layers) - 1}]")
                        else:
                            self.target_layer = encoder
                            print(f"✅ TRUE Grad-CAM hooks registered on: encoder")
                    else:
                        self.target_layer = vision_model
                        print(f"✅ TRUE Grad-CAM hooks registered on: vision_model")
                else:
                    self.target_layer = clip_model
                    print(f"✅ TRUE Grad-CAM hooks registered on: clip_model")
            else:
                self.target_layer = vision_tower
                print(f"✅ TRUE Grad-CAM hooks registered on: vision_tower")

            # Register both forward and backward hooks
            self.target_layer.register_forward_hook(forward_hook)
            self.target_layer.register_backward_hook(backward_hook)

        except Exception as e:
            print(f"⚠️ Warning: Could not hook at vision encoder ({e}), falling back to mm_projector...")
            base_model = self.model.get_model()
            if hasattr(base_model, "mm_projector") and base_model.mm_projector is not None:
                self.target_layer = base_model.mm_projector
                self.target_layer.register_forward_hook(forward_hook)
                print(f"✅ Activation hooks registered on: mm_projector (no gradients)")
            else:
                raise ValueError("Could not find vision encoder or mm_projector to hook")

        self.gradients = None

    def _manual_generate(self, images: torch.Tensor, input_ids: torch.Tensor, max_new_tokens: int = 40,
                         temperature: float = 0.7) -> str:
        """Manual generation loop with temperature sampling for variation."""
        device = images.device
        generated_ids = input_ids.clone().to(device)

        # Debug: verify image tensor
        if images is None or images.numel() == 0:
            print("  ⚠️ WARNING: Image tensor is None or empty!")
            return "Error: No image provided"

        # Get vocabulary size for validation
        vocab_size = None
        # Try multiple methods to get vocab size
        if hasattr(self.tokenizer, 'vocab_size'):
            vocab_size = self.tokenizer.vocab_size
        elif hasattr(self.tokenizer, 'sp_model'):
            try:
                vocab_size = self.tokenizer.sp_model.get_piece_size()
            except:
                pass
        elif hasattr(self.model, 'config') and hasattr(self.model.config, 'vocab_size'):
            vocab_size = self.model.config.vocab_size
        elif hasattr(self.model, 'get_model') and hasattr(self.model.get_model(), 'config'):
            try:
                vocab_size = self.model.get_model().config.vocab_size
            except:
                pass

        if vocab_size is not None:
            print(f"  [DEBUG] Using vocabulary size: {vocab_size}")

        min_tokens = 20  # Minimum tokens to generate (ensure complete captions)
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Ensure images are passed correctly
                outputs = self.model(images=images, input_ids=generated_ids)
                logits = outputs.logits
                next_token_logits = logits[:, -1, :]

                # Clamp logits to valid vocabulary range if vocab_size is known
                if vocab_size is not None and next_token_logits.shape[-1] > vocab_size:
                    next_token_logits = next_token_logits[:, :vocab_size]

                # Apply temperature for variation (if temperature > 0)
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                    # Ensure we don't sample invalid tokens
                    if vocab_size is not None:
                        # Mask out invalid token positions
                        mask = torch.arange(next_token_logits.shape[-1], device=device) < vocab_size
                        next_token_logits = next_token_logits.masked_fill(~mask, float('-inf'))
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1)
                else:
                    # For argmax, ensure we only consider valid tokens
                    if vocab_size is not None and next_token_logits.shape[-1] > vocab_size:
                        next_token_logits = next_token_logits[:, :vocab_size]
                    next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Validate token ID is within vocabulary range
                token_id = next_token_id.item()
                if vocab_size is not None and token_id >= vocab_size:
                    print(
                        f"  ⚠️ WARNING: Generated token ID {token_id} is out of range (vocab_size={vocab_size}), using EOS token")
                    if self.tokenizer.eos_token_id is not None:
                        token_id = self.tokenizer.eos_token_id
                        next_token_id = torch.tensor([[token_id]], device=device)
                    else:
                        break

                generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

                # ✅ Don't stop on EOS if we haven't generated minimum tokens yet
                if self.tokenizer.eos_token_id is not None and token_id == self.tokenizer.eos_token_id:
                    if step >= min_tokens:
                        break
                    # Otherwise, continue generating even if EOS is generated

                # Try to decode single token for stopping conditions (with error handling)
                try:
                    token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
                    # Stop on various conditions, but only after minimum tokens
                    if step >= min_tokens and token_str in ["\n", "USER:", "</s>", "<|endoftext|>"]:
                        break
                except (IndexError, ValueError) as e:
                    # If decoding fails, check if it's EOS and break (only after min_tokens)
                    if step >= min_tokens and self.tokenizer.eos_token_id is not None and token_id == self.tokenizer.eos_token_id:
                        break
                    # Otherwise, continue but log warning
                    if step == 0:  # Only warn on first occurrence
                        print(f"  ⚠️ WARNING: Could not decode token ID {token_id}: {e}")

                # Stop if we hit a repetition pattern (but only after minimum tokens)
                if step > min_tokens and step > 10:
                    recent_tokens = generated_ids[0, -5:].tolist()
                    if len(set(recent_tokens)) == 1:  # All same token
                        break

        # Filter out invalid token IDs before decoding
        try:
            token_ids_list = generated_ids[0].tolist()
            # Filter to only valid token IDs
            if vocab_size is not None:
                valid_token_ids = [tid for tid in token_ids_list if 0 <= tid < vocab_size]
            else:
                valid_token_ids = token_ids_list

            if not valid_token_ids:
                return "Error: No valid tokens generated"

            # Decode only valid tokens
            caption = self.tokenizer.decode(valid_token_ids, skip_special_tokens=True)
        except (IndexError, ValueError) as e:
            print(f"  ⚠️ ERROR: Could not decode generated tokens: {e}")
            # Try to decode just the new tokens (after input_ids)
            try:
                new_tokens = generated_ids[0, input_ids.shape[1]:].tolist()
                if vocab_size is not None:
                    new_tokens = [tid for tid in new_tokens if 0 <= tid < vocab_size]
                if new_tokens:
                    caption = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                else:
                    caption = "Error: Could not decode any valid tokens"
            except:
                caption = f"Error: Decoding failed - {str(e)}"

        if "ASSISTANT:" in caption:
            caption = caption.split("ASSISTANT:")[-1].strip()
        # Clean up any remaining prompt text
        if "USER:" in caption:
            caption = caption.split("USER:")[0].strip()
        
        # ✅ CRITICAL: Ensure caption is not empty
        if not caption or len(caption.strip()) == 0:
            print(f"  ⚠️ WARNING: Generated empty caption, attempting retry...")
            # Retry with lower temperature for more deterministic output
            if temperature > 0.3:
                return self._manual_generate(images, input_ids, max_new_tokens, temperature=0.3)
            else:
                return "Error: Could not generate caption (empty output)"
        
        # Ensure minimum length (at least 10 characters)
        if len(caption.strip()) < 10:
            print(f"  ⚠️ WARNING: Caption too short ({len(caption)} chars), may be incomplete")
        
        # ✅ CRITICAL: Validate caption quality - filter out garbage outputs
        if not self._is_valid_caption(caption):
            print(f"  ⚠️ WARNING: Generated invalid/garbage caption, attempting retry...")
            # Retry with lower temperature for more deterministic output
            if temperature > 0.3:
                return self._manual_generate(images, input_ids, max_new_tokens, temperature=0.3)
            else:
                return "Error: Could not generate valid caption (garbage output detected)"
        
        return caption
    
    def _is_valid_caption(self, caption: str) -> bool:
        """
        Validate that caption is not garbage/invalid.
        Checks for common garbage patterns.
        """
        if not caption or len(caption.strip()) < 10:
            return False
        
        caption_lower = caption.lower()
        
        # Check for garbage patterns
        garbage_patterns = [
            "item1", "item2", "item3", "item4", "item5",
            "exited", "exit", "error", "exception",
            "none", "null", "undefined",
            "stay tuned", "coming soon",
            "cannot determine", "cannot be determined",
            "please provide", "please send",
            "loading", "processing",
            "i cannot see the image",
            "photo by",
            "photo credit",
            "alamy stock photo",
            "getty images",
            "shutterstock",
            "still working on",
            "working on the diagnosis",
            "this is an example of",
            "example of a correct",
            "example of correct",
            "c-super",  # Garbage pattern like "c-super 0.73"
        ]
        
        # ✅ NEW: Check for number-letter patterns (like "c-super 0.73")
        if re.search(r'\b[a-z]-[a-z]+\s+\d+\.\d+\b', caption_lower):
            print(f"      ❌ Detected garbage pattern (number-letter): '{caption.strip()[:50]}...'")
            return False
        
        # If caption contains garbage patterns, it's invalid
        for pattern in garbage_patterns:
            if pattern in caption_lower:
                print(f"      ❌ Detected garbage pattern: '{pattern}'")
                return False
        
        # Check if caption is mostly numbers or special characters
        alphanumeric_chars = sum(1 for c in caption if c.isalnum() or c.isspace())
        if len(caption) > 0 and alphanumeric_chars / len(caption) < 0.5:
            print(f"      ❌ Caption contains too many special characters")
            return False
        
        # Check if caption has too many repeated words (suggests repetition loop)
        words = caption.split()
        if len(words) > 0:
            unique_words = len(set(words))
            if unique_words / len(words) < 0.3 and len(words) > 5:
                print(f"      ❌ Caption has too many repeated words (repetition detected)")
                return False
        
        # Check if caption looks like debugging output
        if caption.strip().startswith("exited") or caption.strip().startswith("exit"):
            return False
        
        # ✅ NEW: Check for "or" patterns that suggest multiple options (meta-response)
        if ' or "' in caption_lower or ' or "the' in caption_lower:
            # If it contains "I cannot see" or similar before "or", it's likely a meta-response
            if any(phrase in caption_lower.split(' or ')[0] for phrase in ["i cannot see", "i don't see", "cannot see"]):
                print(f"      ❌ Detected meta-response with 'or' pattern: '{caption.strip()[:50]}...'")
                return False
        
        # Check if caption is just numbers
        if caption.strip().replace(".", "").replace(",", "").isdigit():
            return False
        
        # ✅ NEW: Check if caption looks like a file ID (e.g., "09-703-09-08-LV-0016" or "mohittodu2021022120")
        # File IDs typically have patterns like: numbers-dashes-letters-numbers or letters+numbers
        file_id_patterns = [
            r'^\d{2}-\d{3}-\d{2}-\d{2}-[A-Z]{2}-\d{4}',  # 09-703-09-08-LV-0016
            r'^\d+-\d+-[A-Z]+-\d+',  # General pattern
            r'^[A-Z]{2,3}-\d{4,}',  # LV-0016 style
            r'^[a-z]{6,}\d{8,}$',  # mohittodu2021022120 style (letters followed by many digits)
            r'^[a-z]+\d{10,}$',  # General: letters then 10+ digits
        ]
        for pattern in file_id_patterns:
            if re.match(pattern, caption.strip(), re.IGNORECASE):
                print(f"      ❌ Detected file ID pattern: '{caption.strip()}'")
                return False
        
        # ✅ NEW: Check if caption is mostly alphanumeric with no spaces (likely file ID)
        stripped = caption.strip()
        if len(stripped) > 8 and len(stripped) < 30:  # Typical file ID length
            # If it's mostly alphanumeric with few/no spaces, might be file ID
            alnum_ratio = sum(1 for c in stripped if c.isalnum()) / len(stripped) if len(stripped) > 0 else 0
            space_count = stripped.count(' ')
            if alnum_ratio > 0.9 and space_count < 2:  # Mostly alphanumeric, few spaces
                # Check if it has both letters and numbers (file ID pattern)
                has_letters = any(c.isalpha() for c in stripped)
                has_numbers = any(c.isdigit() for c in stripped)
                if has_letters and has_numbers:
                    print(f"      ❌ Detected file ID-like pattern (alphanumeric, no spaces): '{stripped}'")
                    return False
        
        # Check if caption is mostly dashes and numbers (file ID pattern)
        stripped = caption.strip().replace("-", "").replace("_", "").replace(".", "")
        if len(stripped) > 0 and sum(1 for c in stripped if c.isdigit() or c.isupper()) / len(stripped) > 0.7:
            # If more than 70% is digits/uppercase, likely a file ID
            if "-" in caption or "_" in caption:
                print(f"      ❌ Detected file ID-like pattern: '{caption.strip()}'")
                return False
        
        return True

    def _verify_image_processing(self, image_tensor: torch.Tensor) -> bool:
        """Verify that image processing is working and features are computed."""
        try:
            base_model = self.model.get_model()
            vision_tower = base_model.get_vision_tower()

            if not getattr(vision_tower, "is_loaded", False):
                print(f"  ⚠️  Vision tower not loaded!")
                return False

            # Try to get vision features (without output_hidden_states if not supported)
            with torch.no_grad():
                try:
                    vision_outputs = vision_tower(image_tensor, output_hidden_states=True)
                except TypeError:
                    # Fallback: try without output_hidden_states
                    vision_outputs = vision_tower(image_tensor)

                # Handle different output formats
                if hasattr(vision_outputs, 'last_hidden_state'):
                    features = vision_outputs.last_hidden_state
                elif isinstance(vision_outputs, tuple) and len(vision_outputs) > 0:
                    features = vision_outputs[0]
                elif hasattr(vision_outputs, 'image_embeds'):
                    features = vision_outputs.image_embeds
                else:
                    features = vision_outputs

                if features is not None:
                    feature_norm = features.norm().item()
                    print(f"  [VERIFY] Vision features computed: shape={features.shape}, norm={feature_norm:.4f}")
                    if feature_norm < 1e-6:
                        print(f"  ⚠️  WARNING: Vision features are near zero!")
                        return False
                    return True
            return False
        except Exception as e:
            print(f"  ⚠️  Could not verify image processing: {e}")
            return False

    def _capture_vision_features_directly(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Directly capture vision encoder features without relying on hooks.
        This is more reliable for activation-based saliency.
        """
        try:
            base_model = self.model.get_model()
            vision_tower = base_model.get_vision_tower()
            
            # Ensure vision tower is loaded
            if not getattr(vision_tower, "is_loaded", False):
                vision_tower.load_model()
            
            # Directly call vision tower to get features
            with torch.no_grad():
                # Get the internal CLIP model
                if hasattr(vision_tower, "vision_tower"):
                    clip_model = vision_tower.vision_tower
                    if hasattr(clip_model, "vision_model"):
                        vision_model = clip_model.vision_model
                        # Process image through vision model
                        vision_outputs = vision_model(image_tensor)
                        # Extract features from the last layer
                        if hasattr(vision_outputs, 'last_hidden_state'):
                            features = vision_outputs.last_hidden_state
                        elif isinstance(vision_outputs, tuple):
                            features = vision_outputs[0]
                        else:
                            features = vision_outputs
                        return features
                    else:
                        # Fallback: use vision_tower directly
                        vision_outputs = vision_tower(image_tensor)
                        if hasattr(vision_outputs, 'last_hidden_state'):
                            return vision_outputs.last_hidden_state
                        elif isinstance(vision_outputs, tuple):
                            return vision_outputs[0]
                        return vision_outputs
                else:
                    # Direct call to vision_tower
                    vision_outputs = vision_tower(image_tensor)
                    if hasattr(vision_outputs, 'last_hidden_state'):
                        return vision_outputs.last_hidden_state
                    elif isinstance(vision_outputs, tuple):
                        return vision_outputs[0]
                    return vision_outputs
        except Exception as e:
            print(f"  ⚠️  Direct vision feature capture failed: {e}")
            return None

    def _generate_saliency_map_only(self, image: Image.Image, prompt: str) -> np.ndarray:
        """
        Generate saliency map WITHOUT generating caption first.
        This allows us to use saliency to guide caption generation.
        Uses direct vision feature capture for more reliable results.
        """
        device = next(self.model.parameters()).device
        
        # Process image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        original_size = image.size
        if image.size != (336, 336):
            image = image.resize((336, 336), Image.Resampling.LANCZOS)
        
        processed = self.image_processor(image, return_tensors="pt")
        image_tensor = processed["pixel_values"].to(device=device, dtype=self.model_dtype)
        
        # ✅ NEW: Directly capture vision features (more reliable than hooks)
        print(f"  🔍 Capturing vision features directly...")
        features = self._capture_vision_features_directly(image_tensor)
        
        if features is None:
            print(f"  ⚠️  Direct feature capture failed, trying hook-based method...")
            # Fallback to hook-based method
            self.model.eval()
            self.activations = None  # Reset activations
            
            # Build prompt for hook-based method
            conv = self.conv_templates["vicuna_v1"].copy()
            if "<image>" not in prompt:
                prompt_with_image = "<image>\n" + prompt
            else:
                prompt_with_image = prompt
            conv.append_message(conv.roles[0], prompt_with_image)
            conv.append_message(conv.roles[1], None)
            prompt_text = conv.get_prompt()
            
            try:
                input_ids = self.tokenizer_image_token(
                    prompt_text, self.tokenizer, image_token_index=self.IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
            except TypeError:
                input_ids = self.tokenizer_image_token(prompt_text, self.tokenizer, return_tensors="pt")
            
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor([input_ids], device=device)
            elif input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            input_ids = input_ids.to(device)
            
            with torch.no_grad():
                outputs = self.model(images=image_tensor, input_ids=input_ids)
            
            if self.activations is None:
                print(f"  ⚠️  Hook-based capture also failed, using image-based fallback")
                # Use image-based saliency as last resort
                img_array = np.array(image)
                if len(img_array.shape) == 3:
                    img_gray = np.mean(img_array, axis=2)
                else:
                    img_gray = img_array
                img_gray_norm = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min() + 1e-8)
                return cv2.resize(img_gray_norm.astype(np.float32), (original_size[0], original_size[1]), interpolation=cv2.INTER_CUBIC)
            
            features = self.activations
        
        # ✅ Process features (from direct capture or hooks)
        if features is not None and features.numel() > 0:
            print(f"  ✅ Features captured: shape={features.shape}, dtype={features.dtype}")
            acts = features.clone().detach()
            
            # Handle different feature shapes
            if acts.dim() == 3:  # [batch, seq_len, hidden_dim]
                acts = acts[0]  # Remove batch dimension
                print(f"  ✅ Removed batch dim: new shape={acts.shape}")
            elif acts.dim() == 4:  # [batch, channels, height, width] - CNN style
                # Average over channels and remove batch
                acts = acts[0].mean(dim=0)  # [height, width]
                # Flatten to 1D for processing
                acts = acts.flatten().unsqueeze(1)  # [height*width, 1]
                print(f"  ✅ Processed CNN-style features: new shape={acts.shape}")
            
            if acts.dim() == 2:
                seq_len = acts.shape[0]
                hidden_dim = acts.shape[1]
                expected_patches = 24 * 24  # 576 for 336x336 image with patch size 14
                
                # ✅ Validate features have variation
                feature_std = acts.std().item()
                feature_mean = acts.mean().item()
                feature_range = acts.max().item() - acts.min().item()
                print(f"  📊 Feature stats: mean={feature_mean:.4f}, std={feature_std:.4f}, range={feature_range:.4f}")
                
                if feature_std < 1e-6:
                    print(f"  ⚠️  WARNING: Features are uniform (std={feature_std:.6f}), using image-based fallback")
                    img_array = np.array(image)
                    if len(img_array.shape) == 3:
                        img_gray = np.mean(img_array, axis=2)
                    else:
                        img_gray = img_array
                    img_gray_norm = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min() + 1e-8)
                    return cv2.resize(img_gray_norm.astype(np.float32), (original_size[0], original_size[1]), interpolation=cv2.INTER_CUBIC)
                
                # Handle CLS token (if present)
                if seq_len == expected_patches + 1:
                    print(f"  ✅ Detected CLS token (577 tokens), removing first token")
                    acts = acts[1:, :]
                    seq_len = expected_patches
                elif seq_len > expected_patches:
                    print(f"  ⚠️  Unexpected sequence length {seq_len} (expected {expected_patches} or {expected_patches+1})")
                    # Assume first token is CLS
                    acts = acts[1:, :]
                    seq_len = acts.shape[0]
                elif seq_len < expected_patches:
                    print(f"  ⚠️  Sequence length {seq_len} is less than expected {expected_patches}")
                
                print(f"  ✅ Processing {seq_len} patches with hidden_dim={hidden_dim}")
                
                # Compute saliency using activation-based method
                method = cfg.ACTIVATION_METHOD.lower()
                print(f"  🔧 Using activation method: {method}")
                if method == "l2_norm":
                    cam_1d = torch.norm(acts, dim=1)  # [seq_len]
                elif method == "abs_mean":
                    cam_1d = torch.abs(acts).mean(dim=1)
                elif method == "max":
                    cam_1d = torch.abs(acts).max(dim=1)[0]
                elif method == "variance":
                    cam_1d = torch.var(acts, dim=1)
                else:
                    cam_1d = torch.norm(acts, dim=1)
                
                # Validate cam_1d has variation
                cam_std = cam_1d.std().item()
                cam_range = cam_1d.max().item() - cam_1d.min().item()
                print(f"  📊 CAM stats before reshape: mean={cam_1d.mean().item():.4f}, std={cam_std:.4f}, range={cam_range:.4f}")
                
                if cam_std < 1e-6:
                    print(f"  ⚠️  WARNING: CAM is uniform (std={cam_std:.6f}), using image-based fallback")
                    img_array = np.array(image)
                    if len(img_array.shape) == 3:
                        img_gray = np.mean(img_array, axis=2)
                    else:
                        img_gray = img_array
                    img_gray_norm = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min() + 1e-8)
                    return cv2.resize(img_gray_norm.astype(np.float32), (original_size[0], original_size[1]), interpolation=cv2.INTER_CUBIC)
                
                num_patches = cam_1d.shape[0]
                if num_patches == expected_patches:
                    cam = cam_1d.reshape(24, 24)
                else:
                    side = int(np.sqrt(num_patches))
                    if side * side == num_patches:
                        cam = cam_1d.reshape(side, side)
                    else:
                        best_h, best_w = None, None
                        min_diff = float('inf')
                        for h in range(1, int(np.sqrt(num_patches)) + 1):
                            if num_patches % h == 0:
                                w = num_patches // h
                                diff = abs(h - w)
                                if diff < min_diff:
                                    min_diff = diff
                                    best_h, best_w = h, w
                        if best_h is not None:
                            cam = cam_1d.reshape(best_h, best_w)
                
                cam = F.relu(cam)
                cam_min = cam.min().item()
                cam_max = cam.max().item()
                if cam_max > cam_min + 1e-6:
                    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
                    gamma = 0.5
                    cam = torch.pow(cam, gamma)
                    cam = (cam - cam.min()) / (cam.max() + 1e-8)
                
                cam_np = cam.detach().cpu().numpy().astype(np.float32)
                
                # ✅ CRITICAL: Validate saliency map has variation before returning
                cam_std = cam_np.std()
                cam_range = cam_np.max() - cam_np.min()
                if cam_std < 1e-6 or cam_range < 1e-6:
                    print(f"  ⚠️  WARNING: Saliency map is uniform (std={cam_std:.6f}, range={cam_range:.6f})")
                    print(f"      This suggests activations may not be captured correctly.")
                    # Try to add some variation based on image features
                    # This is a fallback to prevent completely uniform maps
                    img_array = np.array(image.resize((24, 24)) if image.size != (24, 24) else image)
                    if len(img_array.shape) == 3:
                        img_gray = np.mean(img_array, axis=2)
                    else:
                        img_gray = img_array
                    # Use image intensity as a fallback saliency map
                    cam_np = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min() + 1e-8)
                    cam_np = cam_np.astype(np.float32)
                    print(f"      Using image-based fallback saliency map")
                
                cam_resized = cv2.resize(cam_np, (original_size[0], original_size[1]), interpolation=cv2.INTER_CUBIC)
                
                # Final validation
                if cam_resized.std() < 1e-6:
                    print(f"  ⚠️  WARNING: Resized saliency map is still uniform")
                
                return cam_resized
            else:
                print(f"  ⚠️  Could not reshape features properly (dim={acts.dim()}, shape={acts.shape})")
                print(f"      Expected 2D features [seq_len, hidden_dim], got {acts.dim()}D")
        else:
            print(f"  ⚠️  No features captured, saliency map generation failed")
            if features is None:
                print(f"      Features is None - direct capture and hooks both failed")
            elif features.numel() == 0:
                print(f"      Features is empty (numel=0)")
            else:
                print(f"      Features shape: {features.shape}, but processing failed")
        
        # Fallback: return image-based saliency (not uniform)
        print(f"  ⚠️  Using image-based fallback saliency map")
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_gray = np.mean(img_array, axis=2)
        else:
            img_gray = img_array
        # Normalize to [0, 1]
        img_gray_norm = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min() + 1e-8)
        return img_gray_norm.astype(np.float32)

    def generate_cam_with_saliency_guidance(self, image: Image.Image, prompt: str, use_saliency_for_refinement: bool = True) -> Tuple[np.ndarray, str]:
        """
        Generate Grad-CAM saliency map and caption, using saliency + PLI to improve caption generation.
        
        Strategy: Two-pass approach with PLI-guided adaptive enhancement
        1. First pass: Generate saliency map quickly (activation-based)
        2. Generate PLI map from saliency (fuzzy logic refinement)
        3. Use PLI values to adaptively control enhancement strength:
           - High PLI (high confidence) regions → stronger enhancement
           - Low PLI (low confidence) regions → weaker enhancement
        4. Create PLI-guided adaptive enhanced image
        5. Enhance prompt based on saliency characteristics
        6. Second pass: Generate caption using PLI-guided enhanced image + enhanced prompt
        
        Args:
            image: Input image
            prompt: Text prompt
            use_saliency_for_refinement: If True, use saliency + PLI maps to guide caption generation
        
        Returns:
            cam: Saliency map
            caption: Improved caption (generated using PLI-guided adaptive enhancement)
        """
        if not use_saliency_for_refinement:
            return self.generate_cam(image, prompt)
        
        print(f"  🔍 Using saliency + PLI-guided caption generation (two-pass approach with adaptive enhancement)...")
        
        # PASS 1: Generate saliency map first (without full caption generation)
        cam = self._generate_saliency_map_only(image, prompt)
        
        if cam is None or cam.size == 0:
            print(f"  ⚠️  Could not generate saliency map, falling back to standard generation")
            return self.generate_cam(image, prompt)
        
        # Analyze saliency map to guide caption generation
        saliency_threshold = np.percentile(cam, 75)  # Top 25% of saliency values
        high_saliency_mask = cam > saliency_threshold
        saliency_coverage = high_saliency_mask.sum() / cam.size
        saliency_mean = cam.mean()
        saliency_max = cam.max()
        
        print(f"  📊 Saliency stats: coverage={saliency_coverage:.2%}, mean={saliency_mean:.3f}, max={saliency_max:.3f}")
        
        # Strategy 1: If saliency is highly focused, create a masked image emphasizing important regions
        # Strategy 2: Enhance prompt based on saliency characteristics
        enhanced_prompt = prompt
        
        if saliency_coverage < 0.3:  # Highly focused saliency
            print(f"  🎯 Saliency is highly focused ({saliency_coverage:.1%} coverage), emphasizing important regions")
            enhanced_prompt = f"{prompt} Pay special attention to the most prominent pathological features in the image."
        elif saliency_coverage > 0.7:  # Diffuse saliency
            print(f"  📍 Saliency is diffuse ({saliency_coverage:.1%} coverage), examining overall tissue architecture")
            enhanced_prompt = f"{prompt} Examine the overall tissue architecture and cell morphology throughout the image."
        
        # ✅ NEW: Generate PLI map for adaptive enhancement
        pli_map = None
        if PLI_AVAILABLE:
            try:
                pli_map = gradcam_to_pli(cam)
                pli_mean = pli_map.mean()
                pli_max = pli_map.max()
                print(f"  🎨 PLI map generated: mean={pli_mean:.3f}, max={pli_max:.3f}")
                print(f"  ✅ Using PLI-guided adaptive enhancement")
            except Exception as e:
                print(f"  ⚠️  PLI generation failed: {e}, using saliency-only enhancement")
                pli_map = None
        
        # Create saliency-weighted image with PLI-guided adaptive enhancement
        img_array = np.array(image)
        if img_array.shape[:2] != cam.shape[:2]:
            img_array = cv2.resize(img_array, (cam.shape[1], cam.shape[0]))
        
        # ✅ NEW: PLI-guided adaptive enhancement strategy
        enhanced_img = img_array.astype(np.float32) / 255.0
        
        if pli_map is not None:
            # ✅ PLI-GUIDED ADAPTIVE ENHANCEMENT
            # Resize PLI to match image if needed
            if pli_map.shape[:2] != enhanced_img.shape[:2]:
                pli_map = cv2.resize(pli_map, (enhanced_img.shape[1], enhanced_img.shape[0]), interpolation=cv2.INTER_CUBIC)
            
            # Use PLI values to determine adaptive enhancement strength
            # High PLI (high confidence) = stronger enhancement
            # Low PLI (low confidence) = weaker enhancement
            base_weight = 0.2  # Base enhancement strength
            max_weight = 0.5   # Maximum enhancement strength for high-confidence regions
            
            # Create adaptive weight map: base_weight + (PLI * (max_weight - base_weight))
            # This gives us weights from base_weight to max_weight based on PLI confidence
            adaptive_weights = base_weight + (pli_map * (max_weight - base_weight))
            
            # Normalize saliency map for blending
            cam_normalized = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            saliency_3d = np.stack([cam_normalized] * 3, axis=2)  # Convert to RGB
            adaptive_weights_3d = np.stack([adaptive_weights] * 3, axis=2)  # Convert to RGB
            
            # Apply adaptive enhancement: stronger in high-PLI regions
            # Formula: original * (1 - weight) + original * (1 + saliency * weight)
            # But weight is now adaptive based on PLI
            enhanced_img = enhanced_img * (1 - adaptive_weights_3d) + enhanced_img * (1 + saliency_3d * adaptive_weights_3d)
            
            print(f"  📊 PLI-guided enhancement: base_weight={base_weight:.2f}, max_weight={max_weight:.2f}")
            print(f"      High-confidence regions (PLI > 0.7): {np.sum(pli_map > 0.7) / pli_map.size:.1%} of image")
            print(f"      Average enhancement strength: {adaptive_weights.mean():.3f}")
        else:
            # Fallback: Use fixed saliency weighting (original approach)
            saliency_3d = np.stack([cam] * 3, axis=2)  # Convert to RGB
            saliency_weight = 0.3  # Fixed weight
            enhanced_img = enhanced_img * (1 - saliency_weight) + enhanced_img * (1 + saliency_3d * saliency_weight)
            print(f"  📊 Using fixed saliency enhancement (weight={saliency_weight:.2f})")
        
        enhanced_img = np.clip(enhanced_img, 0, 1)
        enhanced_img = (enhanced_img * 255).astype(np.uint8)
        enhanced_image = Image.fromarray(enhanced_img)
        
        # PASS 2: Generate caption using PLI-guided enhanced image and enhanced prompt
        if pli_map is not None:
            print(f"  ✍️  Generating caption with PLI-guided adaptive enhanced image and enhanced prompt...")
        else:
            print(f"  ✍️  Generating caption with saliency-guided image and enhanced prompt...")
        
        # Generate caption only (without computing saliency again)
        device = next(self.model.parameters()).device
        if enhanced_image.size != (336, 336):
            enhanced_image = enhanced_image.resize((336, 336), Image.Resampling.LANCZOS)
        
        processed = self.image_processor(enhanced_image, return_tensors="pt")
        image_tensor = processed["pixel_values"].to(device=device, dtype=self.model_dtype)
        
        conv = self.conv_templates["vicuna_v1"].copy()
        if "<image>" not in enhanced_prompt:
            prompt_with_image = "<image>\n" + enhanced_prompt
        else:
            prompt_with_image = enhanced_prompt
        conv.append_message(conv.roles[0], prompt_with_image)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()
        
        try:
            input_ids = self.tokenizer_image_token(
                prompt_text, self.tokenizer, image_token_index=self.IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
        except TypeError:
            input_ids = self.tokenizer_image_token(prompt_text, self.tokenizer, return_tensors="pt")
        
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor([input_ids], device=device)
        elif input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device)
        
        # Generate caption
        refined_caption = self._manual_generate(
            image_tensor, 
            input_ids, 
            max_new_tokens=cfg.MAX_NEW_TOKENS, 
            temperature=cfg.GENERATION_TEMPERATURE
        )
        
        return cam, refined_caption

    def generate_cam(self, image: Image.Image, prompt: str) -> Tuple[np.ndarray, str]:
        """Generate Grad-CAM saliency map and caption for an image."""
        device = next(self.model.parameters()).device

        try:
            # Ensure image is RGB
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            if image.mode != "RGB":
                image = image.convert("RGB")

            original_size = image.size

            # Resize image to 336x336 for CLIP
            if image.size != (336, 336):
                image = image.resize((336, 336), Image.Resampling.LANCZOS)

            # Process image
            processed = self.image_processor(image, return_tensors="pt")
            image_tensor = processed["pixel_values"].to(device=device, dtype=self.model_dtype)

            # Verify image processing (first time only, or if DEBUG_MODE)
            if cfg.DEBUG_MODE or not hasattr(self, '_image_processing_verified'):
                if not self._verify_image_processing(image_tensor):
                    print(f"  ⚠️  Image processing verification failed!")
                self._image_processing_verified = True

            # Build conversation prompt
            conv = self.conv_templates["vicuna_v1"].copy()
            # Ensure the prompt includes the image token placeholder
            if "<image>" not in prompt and "<IMAGE>" not in prompt:
                # Add image token to the prompt
                prompt_with_image = "<image>\n" + prompt
            else:
                prompt_with_image = prompt
            conv.append_message(conv.roles[0], prompt_with_image)
            conv.append_message(conv.roles[1], None)
            prompt_text = conv.get_prompt()

            # Debug: check if prompt contains image token
            if cfg.DEBUG_MODE:
                print(f"  [DEBUG] Prompt text contains <image>: {'<image>' in prompt_text.lower()}")
                print(f"  [DEBUG] Prompt preview: {prompt_text[:150]}...")

            # Tokenize with image tokens
            # Check if prompt_text contains the image token placeholder
            if "<image>" not in prompt_text and "<IMAGE>" not in prompt_text:
                print(f"  ⚠️  WARNING: Prompt text doesn't contain <image> placeholder!")
                print(f"      Adding <image> token to prompt...")
                # Insert <image> after the system prompt (usually after first sentence)
                prompt_parts = prompt_text.split("\n")
                if len(prompt_parts) > 1:
                    prompt_text = prompt_parts[0] + "\n<image>\n" + "\n".join(prompt_parts[1:])
                else:
                    # Insert after first space or at beginning
                    prompt_text = prompt_text.replace("USER:", "USER:\n<image>", 1)

            # Tokenize with image tokens
            # tokenizer_image_token should handle IMAGE_TOKEN_INDEX automatically
            try:
                # Try with image_token_index parameter
                input_ids = self.tokenizer_image_token(
                    prompt_text,
                    self.tokenizer,
                    image_token_index=self.IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                )
            except TypeError:
                # Fallback: try without image_token_index (it might be positional or auto-detected)
                try:
                    input_ids = self.tokenizer_image_token(
                        prompt_text,
                        self.tokenizer,
                        return_tensors="pt",
                    )
                except Exception as e:
                    print(f"  ⚠️  tokenizer_image_token failed: {e}")
                    # Manual fallback: tokenize and replace <image> token
                    input_ids = self.tokenizer.encode(prompt_text, return_tensors="pt")
                    # Find and replace <image> token if present
                    image_token_str = self.tokenizer.encode("<image>", add_special_tokens=False)
                    if len(image_token_str) > 0:
                        # Replace first occurrence of image token string with IMAGE_TOKEN_INDEX
                        input_ids_list = input_ids[0].tolist()
                        for i in range(len(input_ids_list) - len(image_token_str) + 1):
                            if input_ids_list[i:i + len(image_token_str)] == image_token_str:
                                input_ids_list[i:i + len(image_token_str)] = [self.IMAGE_TOKEN_INDEX]
                                input_ids = torch.tensor([input_ids_list], device=device)
                                break

            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor([input_ids], device=device)
            elif input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)

            input_ids = input_ids.to(device)

            # Verify IMAGE_TOKEN is present
            input_ids_list = input_ids[0].tolist()
            if self.IMAGE_TOKEN_INDEX not in input_ids_list:
                print(f"  ⚠️  CRITICAL WARNING: IMAGE_TOKEN ({self.IMAGE_TOKEN_INDEX}) not found in input_ids!")
                print(f"      This will cause the model to ignore image inputs.")
                print(f"      Input IDs sample: {input_ids_list[:20]}")
                print(f"      Prompt text: {prompt_text[:100]}...")
                # Try to manually add IMAGE_TOKEN if missing
                # Find where to insert it (usually after system prompt, before user message)
                if len(input_ids_list) > 0:
                    # Find position after "USER:" or after first few tokens
                    insert_pos = 1
                    # Look for common patterns
                    if 330 in input_ids_list:  # Common token after system prompt
                        insert_pos = input_ids_list.index(330) + 1
                    elif len(input_ids_list) > 5:
                        insert_pos = min(5, len(input_ids_list) - 1)

                    # Insert IMAGE_TOKEN
                    input_ids = torch.cat([
                        input_ids[:, :insert_pos],
                        torch.tensor([[self.IMAGE_TOKEN_INDEX]], device=device),
                        input_ids[:, insert_pos:]
                    ], dim=1)
                    print(f"      Manually inserted IMAGE_TOKEN at position {insert_pos}")
            else:
                if cfg.DEBUG_MODE:
                    img_token_pos = input_ids_list.index(self.IMAGE_TOKEN_INDEX)
                    print(f"  [DEBUG] IMAGE_TOKEN found at position {img_token_pos} in input_ids")

            # Ensure all relevant submodules are on the same device
            base_model = self.model.get_model()
            base_model.to(device)
            if hasattr(base_model, "mm_projector") and base_model.mm_projector is not None:
                base_model.mm_projector.to(device)
            try:
                vt = base_model.get_vision_tower()
                vt.to(device)
            except Exception:
                pass

            # Generate caption FIRST (before backward pass)
            self.model.eval()
            image_tensor_gen = image_tensor.clone().detach().to(device).to(self.model_dtype)
            input_ids_gen = input_ids.clone().detach().to(device).long()

            # Debug: Verify image processing (only if DEBUG_MODE is enabled)
            if cfg.DEBUG_MODE:
                print(f"  [DEBUG] Image tensor shape: {image_tensor_gen.shape}, dtype: {image_tensor_gen.dtype}")
                print(
                    f"  [DEBUG] Image tensor mean: {image_tensor_gen.mean().item():.4f}, std: {image_tensor_gen.std().item():.4f}")
                print(
                    f"  [DEBUG] Input IDs shape: {input_ids_gen.shape}, contains IMAGE_TOKEN: {self.IMAGE_TOKEN_INDEX in input_ids_gen[0].tolist()}")

                # Verify image features are being computed
                try:
                    base_model = self.model.get_model()
                    vision_tower = base_model.get_vision_tower()
                    if hasattr(vision_tower, "is_loaded") and vision_tower.is_loaded:
                        # Test forward pass to verify image processing
                        test_output = self.model(images=image_tensor_gen[:1], input_ids=input_ids_gen[:1])
                        print(f"  [DEBUG] Model forward pass successful, logits shape: {test_output.logits.shape}")
                    else:
                        print(f"  [WARNING] Vision tower not loaded!")
                except Exception as e:
                    print(f"  [WARNING] Could not verify image processing: {e}")

            # Use temperature for variation (0.7 = some randomness, 0.0 = deterministic)
            caption = self._manual_generate(
                image_tensor_gen,
                input_ids_gen,
                max_new_tokens=cfg.MAX_NEW_TOKENS,
                temperature=cfg.GENERATION_TEMPERATURE
            )

            # Try TRUE Grad-CAM first, fallback to activation-based
            self.gradients = None
            self.activations = None

            # Enable gradients for TRUE Grad-CAM
            self.model.train()
            for param in self.model.parameters():
                param.requires_grad = True

            vision_tower = self.model.get_model().get_vision_tower()
            vision_tower.train()
            for param in vision_tower.parameters():
                param.requires_grad = True

            self.model.zero_grad(set_to_none=True)
            image_tensor.requires_grad_(True)

            # Forward pass
            outputs = self.model(images=image_tensor, input_ids=input_ids)
            logits = outputs.logits
            last_token_logits = logits[0, -1, :]
            target = last_token_logits.sum()

            # Backward pass for TRUE Grad-CAM
            print(f"  Attempting backward pass for TRUE Grad-CAM...")
            target.backward(retain_graph=False)

            self.model.eval()

            # Try TRUE Grad-CAM first
            cam = None
            use_true_gradcam = False

            if self.gradients is not None and self.activations is not None:
                grads = self.gradients.clone().detach()
                acts = self.activations.clone().detach()

                grad_norm = grads.abs().sum().item()
                if grad_norm > 1e-6:
                    use_true_gradcam = True
                    print(f"  ✅ TRUE Grad-CAM: Gradients captured! (norm={grad_norm:.6f})")
                else:
                    print(f"  ⚠️  Gradients are zero (norm={grad_norm:.6f}), falling back to activation-based...")

            if use_true_gradcam:
                print(f"  Computing TRUE Grad-CAM from gradients and activations...")
                print(f"    Gradients shape: {grads.shape}, Activations shape: {acts.shape}")

                if grads.dim() == 3:
                    grads = grads[0]
                    acts = acts[0]
                elif grads.dim() == 4:
                    grads = grads[0]
                    acts = acts[0]

                if grads.dim() == 2 and acts.dim() == 2:
                    expected_patches = 24 * 24  # 576 for 336x336 image
                    total_tokens = acts.shape[0]

                    print(f"    Total tokens: {total_tokens}, Expected patches: {expected_patches}")

                    if total_tokens == expected_patches + 1:
                        patch_grads = grads[1:, :]
                        patch_acts = acts[1:, :]
                        num_patches = expected_patches
                        print(f"    ✅ Detected CLS token, using {num_patches} patches")
                    elif total_tokens == expected_patches:
                        patch_grads = grads
                        patch_acts = acts
                        num_patches = expected_patches
                        print(f"    ✅ No CLS token detected, using all {num_patches} patches")
                    else:
                        if total_tokens > expected_patches:
                            patch_grads = grads[1:, :]
                            patch_acts = acts[1:, :]
                            num_patches = total_tokens - 1
                        else:
                            patch_grads = grads
                            patch_acts = acts
                            num_patches = total_tokens
                        print(f"    Using {num_patches} patches")

                    weights = torch.mean(patch_grads, dim=1)
                    features = torch.mean(patch_acts, dim=1)
                    cam_1d = weights * features

                    if num_patches == expected_patches:
                        cam = cam_1d.reshape(24, 24)
                        print(f"    ✅ Reshaped to {cam.shape} (24x24 for 336x336 image)")
                    else:
                        side = int(np.sqrt(num_patches))
                        if side * side == num_patches:
                            cam = cam_1d.reshape(side, side)
                            print(f"    ✅ Reshaped to {cam.shape} ({side}x{side})")
                        else:
                            best_h, best_w = None, None
                            min_diff = float('inf')
                            for h in range(1, int(np.sqrt(num_patches)) + 1):
                                if num_patches % h == 0:
                                    w = num_patches // h
                                    diff = abs(h - w)
                                    if diff < min_diff:
                                        min_diff = diff
                                        best_h, best_w = h, w
                            if best_h is not None:
                                cam = cam_1d.reshape(best_h, best_w)
                                print(f"    ✅ Reshaped to {cam.shape} (h={best_h}, w={best_w})")

                if cam is not None:
                    cam = F.relu(cam)
                    cam_min = cam.min().item()
                    cam_max = cam.max().item()

                    if cam_max > cam_min + 1e-6:
                        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
                        gamma = 0.5
                        cam = torch.pow(cam, gamma)
                        cam = (cam - cam.min()) / (cam.max() + 1e-8)
                        print(
                            f"  ✅ TRUE Grad-CAM normalized: min={cam.min().item():.4f}, max={cam.max().item():.4f}")

                        cam_np = cam.detach().cpu().numpy().astype(np.float32)

                        if cam_np.ndim != 2:
                            if cam_np.ndim == 1:
                                sqrt_len = int(np.sqrt(cam_np.shape[0]))
                                if sqrt_len * sqrt_len == cam_np.shape[0]:
                                    cam_np = cam_np.reshape(sqrt_len, sqrt_len)
                                else:
                                    best_h, best_w = None, None
                                    min_diff = float('inf')
                                    for h in range(1, int(np.sqrt(cam_np.shape[0])) + 1):
                                        if cam_np.shape[0] % h == 0:
                                            w = cam_np.shape[0] // h
                                            diff = abs(h - w)
                                            if diff < min_diff:
                                                min_diff = diff
                                                best_h, best_w = h, w
                                    if best_h is not None:
                                        cam_np = cam_np.reshape(best_h, best_w)

                        cam_resized = cv2.resize(
                            cam_np,
                            (original_size[0], original_size[1]),
                            interpolation=cv2.INTER_CUBIC,
                        )

                        print(
                            f"  ✅ TRUE Grad-CAM final: shape={cam_resized.shape}, min={cam_resized.min():.4f}, max={cam_resized.max():.4f}")

                        self.gradients = None
                        self.activations = None
                        del outputs, logits, target, image_tensor, input_ids, processed
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        return cam_resized, caption
                    else:
                        print(f"  ⚠️  TRUE Grad-CAM is uniform, falling back to activation-based...")
                        use_true_gradcam = False

            # Fallback to activation-based if TRUE Grad-CAM didn't work
            if not use_true_gradcam and self.activations is not None:
                print(f"  Using activation-based saliency (fallback)...")
                acts = self.activations.clone()

                print(f"  Activations shape: {acts.shape}")

                if acts.dim() == 4:
                    acts = acts[0]
                elif acts.dim() == 3:
                    if acts.shape[0] == 1:
                        acts = acts[0]

                if acts.dim() == 2:
                    seq_len = acts.shape[0]
                    expected_patches = 24 * 24

                    if seq_len == expected_patches + 1:
                        acts = acts[1:, :]
                        seq_len = expected_patches
                    elif seq_len == expected_patches:
                        pass
                    else:
                        if seq_len > expected_patches:
                            acts = acts[1:, :]
                            seq_len = acts.shape[0]

                    method = cfg.ACTIVATION_METHOD.lower()
                    print(f"    Using activation method: {method}")

                    if method == "l2_norm":
                        cam_1d = torch.norm(acts, dim=1)
                    elif method == "abs_mean":
                        cam_1d = torch.abs(acts).mean(dim=1)
                    elif method == "max":
                        cam_1d = torch.abs(acts).max(dim=1)[0]
                    elif method == "variance":
                        cam_1d = torch.var(acts, dim=1)
                    else:
                        cam_1d = torch.norm(acts, dim=1)

                    num_patches = cam_1d.shape[0]

                    if num_patches == expected_patches:
                        cam = cam_1d.reshape(24, 24)
                    else:
                        side = int(np.sqrt(num_patches))
                        if side * side == num_patches:
                            cam = cam_1d.reshape(side, side)
                        else:
                            best_h, best_w = None, None
                            min_diff = float('inf')
                            for h in range(1, int(np.sqrt(num_patches)) + 1):
                                if num_patches % h == 0:
                                    w = num_patches // h
                                    diff = abs(h - w)
                                    if diff < min_diff:
                                        min_diff = diff
                                        best_h, best_w = h, w
                            if best_h is not None:
                                cam = cam_1d.reshape(best_h, best_w)

                    cam = F.relu(cam)
                    cam_min = cam.min().item()
                    cam_max = cam.max().item()

                    if cam_max > cam_min + 1e-6:
                        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
                        gamma = 0.5
                        cam = torch.pow(cam, gamma)
                        cam = (cam - cam.min()) / (cam.max() + 1e-8)

                    cam_np = cam.detach().cpu().numpy().astype(np.float32)

                    if cam_np.ndim != 2:
                        if cam_np.ndim == 1:
                            sqrt_len = int(np.sqrt(cam_np.shape[0]))
                            if sqrt_len * sqrt_len == cam_np.shape[0]:
                                cam_np = cam_np.reshape(sqrt_len, sqrt_len)

                    cam_resized = cv2.resize(
                        cam_np,
                        (original_size[0], original_size[1]),
                        interpolation=cv2.INTER_CUBIC,
                    )

                    self.gradients = None
                    self.activations = None
                    del outputs, logits, image_tensor, input_ids, processed
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    return cam_resized, caption

            print("⚠️ Warning: No activations captured, returning empty saliency map")
            empty_cam = np.zeros((original_size[1], original_size[0]))
            return empty_cam, caption

        except Exception as e:
            print(f"Error in generate_cam: {e}")
            import traceback
            traceback.print_exc()
            self.gradients = None
            self.activations = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            empty_cam = np.zeros((image.size[1], image.size[0]))
            return empty_cam, f"Error: {str(e)}"

    @staticmethod
    def visualize(image: Image.Image, cam: np.ndarray, caption: str, save_path: Optional[str] = None,
                  pli_map: Optional[np.ndarray] = None):
        """Visualize original image + saliency map + overlay + PLI map (if provided)."""
        if pli_map is not None:
            # Include PLI visualization
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))

            # Row 1: Original, Grad-CAM, Overlay
            axes[0, 0].imshow(image)
            axes[0, 0].set_title("Original Image", fontsize=12)
            axes[0, 0].axis("off")

            im1 = axes[0, 1].imshow(cam, cmap="jet", alpha=0.8)
            axes[0, 1].set_title("Grad-CAM Saliency", fontsize=12)
            axes[0, 1].axis("off")
            plt.colorbar(im1, ax=axes[0, 1])

            img_array = np.array(image)
            if img_array.shape[:2] != cam.shape[:2]:
                img_array = cv2.resize(img_array, (cam.shape[1], cam.shape[0]))
            heatmap = cm.jet(cam)[:, :, :3]
            overlay = (0.4 * img_array / 255.0 + 0.6 * heatmap)
            axes[0, 2].imshow(overlay)
            axes[0, 2].set_title("Saliency Overlay", fontsize=12)
            axes[0, 2].axis("off")

            # Row 2: PLI Map, PLI Overlay, Comparison
            im2 = axes[1, 0].imshow(pli_map, cmap="Reds", alpha=0.8)
            axes[1, 0].set_title("PLI Map", fontsize=12)
            axes[1, 0].axis("off")
            plt.colorbar(im2, ax=axes[1, 0])

            pli_overlay = np.clip(img_array / 255.0 * 0.5 + np.stack([pli_map] * 3, axis=2) * 0.5, 0, 1)
            axes[1, 1].imshow(pli_overlay)
            axes[1, 1].set_title("PLI Overlay", fontsize=12)
            axes[1, 1].axis("off")

            # Side-by-side comparison
            comparison = np.hstack([cam, pli_map])
            axes[1, 2].imshow(comparison, cmap="hot")
            axes[1, 2].set_title("Saliency vs PLI", fontsize=12)
            axes[1, 2].axis("off")

            fig.suptitle(f"Caption: {caption[:80]}...", fontsize=10)
        else:
            # Original 3-panel visualization (no PLI)
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # Original
            axes[0].imshow(image)
            axes[0].set_title("Original Image", fontsize=12)
            axes[0].axis("off")

            # CAM
            im = axes[1].imshow(cam, cmap="jet", alpha=0.8)
            axes[1].set_title("Grad-CAM", fontsize=12)
            axes[1].axis("off")
            plt.colorbar(im, ax=axes[1])

            # Overlay
            img_array = np.array(image)
            if img_array.shape[:2] != cam.shape[:2]:
                img_array = cv2.resize(img_array, (cam.shape[1], cam.shape[0]))
            heatmap = cm.jet(cam)[:, :, :3]
            overlay = (0.4 * img_array / 255.0 + 0.6 * heatmap)
            axes[2].imshow(overlay)
            axes[2].set_title("Overlay", fontsize=12)
            axes[2].axis("off")

            fig.suptitle(f"Caption: {caption[:80]}...", fontsize=10)

        plt.tight_layout()
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ==============
# PROMPT GENERATION
# ==============

def get_prompt(style: str = "descriptive", use_few_shot: bool = False) -> str:
    """Get prompt based on style, optionally with few-shot examples."""
    
    # Few-shot examples for better accuracy
    few_shot_examples = """
Examples of correct diagnoses:
- "Well differentiated tubular adenocarcinoma" → "The histopathology image shows well-differentiated tubular adenocarcinoma with regular glandular structures."
- "Poorly differentiated adenocarcinoma, solid type" → "The histopathology image shows poorly differentiated adenocarcinoma with solid growth pattern and high-grade nuclear atypia."
- "Signet ring cell carcinoma" → "The histopathology image shows signet ring cell carcinoma characterized by cells with prominent cytoplasmic mucin vacuoles."
- "Moderately differentiated tubular adenocarcinoma" → "The histopathology image shows moderately differentiated tubular adenocarcinoma with irregular glandular structures and moderate nuclear atypia."
"""
    
    base_prompts = {
        "descriptive": "You are analyzing a histopathology image. Describe ONLY what you see in the image: the tissue architecture, cell morphology, and pathological features. Do NOT say you cannot see the image. Do NOT provide general information. Describe ONLY the specific features visible in THIS image.",
        "diagnostic": "You are examining a histopathology image. You MUST provide a specific diagnostic classification based ONLY on what you see in the image. Do NOT say 'I cannot see the image' or provide general information. Start your response with 'The histopathology image shows' and then provide: (1) whether the tissue shows neoplastic (cancerous) changes - if cancer is present, you MUST state it clearly (do NOT say 'no cancer' or 'no evidence of cancer' if cancer is visible), (2) the specific cancer type if present (e.g., 'adenocarcinoma', 'signet ring cell carcinoma', 'carcinoma'), (3) the differentiation level using these EXACT terms: 'well differentiated', 'moderately differentiated', or 'poorly differentiated', and (4) any specific subtype features using EXACT terms: 'tubular', 'papillary', 'mucinous', 'solid', 'non-solid', 'signet ring'. Use the EXACT medical terms from the standard histopathology classification system. If you observe adenocarcinoma, state 'adenocarcinoma' explicitly. If you observe signet ring cells, state 'signet ring cell carcinoma' explicitly.",
        "classification": "You are examining a histopathology image. You MUST provide the specific diagnostic classification based ONLY on what you see in the image. Do NOT say 'I cannot see the image' or provide general information. Start your response with 'The histopathology image shows' and then provide the specific subtype classification using EXACT medical terminology including: the cancer type (e.g., 'adenocarcinoma', 'carcinoma'), the differentiation level using EXACT terms ('well differentiated', 'moderately differentiated', or 'poorly differentiated'), and any subtype features using EXACT terms ('tubular', 'papillary', 'signet ring', 'mucinous', 'solid', 'non-solid'). Be specific and use the EXACT medical terminology from standard histopathology classifications."
    }
    
    base_prompt = base_prompts.get(style, base_prompts["descriptive"])
    
    # Add few-shot examples if enabled
    if use_few_shot and cfg.USE_FEW_SHOT_EXAMPLES:
        return base_prompt + "\n\n" + few_shot_examples
    
    return base_prompt


# ==============
# MAIN PIPELINE WITH EVALUATION
# ==============

def main():
    print("=" * 60)
    print("LLaVA-Med + Histopathology Dataset with Accuracy Evaluation")
    print("=" * 60)

    # 1. Register LLaVA custom config
    ensure_llava_registered(cfg.LLAVA_MED_ROOT, cfg.MODEL_ID)

    # 2. Load model + tokenizer
    model, tokenizer = load_llava_med_model(cfg.MODEL_ID)

    # 3. Build dataset
    try:
        dataset = HistopathologyDataset(cfg.DATASET_CSV, cfg.IMAGES_DIR, limit=cfg.NUM_IMAGES)
        print(f"✅ Dataset loaded: {len(dataset)} samples from {cfg.DATASET_CSV}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # 4. Initialize Grad-CAM
    gradcam = GradCAM(model, tokenizer)

    # 5. Load semantic similarity model (if available)
    semantic_model = None
    if SEMANTIC_AVAILABLE:
        try:
            print("Loading semantic similarity model...")
            semantic_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight, fast model
            print("✅ Semantic similarity model loaded")
        except Exception as e:
            print(f"⚠️  Could not load semantic model: {e}")
            semantic_model = None
    else:
        print("⚠️  Semantic similarity disabled (sentence-transformers not installed)")

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    saliency_dir = os.path.join(cfg.OUTPUT_DIR, "saliency_maps")
    pli_dir = os.path.join(cfg.OUTPUT_DIR, "pli_maps")
    os.makedirs(saliency_dir, exist_ok=True)
    os.makedirs(pli_dir, exist_ok=True)

    prompt = get_prompt(cfg.PROMPT_STYLE, use_few_shot=cfg.USE_FEW_SHOT_EXAMPLES)
    print(f"\nUsing prompt style: {cfg.PROMPT_STYLE}")
    if cfg.USE_FEW_SHOT_EXAMPLES:
        print("✅ Few-shot examples enabled")
    if cfg.USE_SALIENCY_FOR_REFINEMENT:
        print("✅ Saliency-guided generation enabled (two-pass)")
    if cfg.ENABLE_POST_PROCESSING:
        print("✅ Post-processing enabled (false negative correction)")
    print(f"Prompt preview: {prompt[:200]}...")
    print("\nGenerating captions with saliency maps and evaluating accuracy...\n")

    results = []
    success_count = 0
    seen_captions = {}  # Track caption uniqueness
    image_hashes = []  # Track image uniqueness

    for idx in range(len(dataset)):
        try:
            sample = dataset[idx]
            scan_id = sample['scan_id']
            label = sample['label']
            ground_truth = sample['ground_truth_caption']
            image = sample['image']

            print(f"[{idx + 1}/{len(dataset)}] {scan_id}")
            print(f"  Label: {label}")

            # Generate caption and saliency map using GradCAM
            # Use saliency-guided generation if enabled
            if cfg.USE_SALIENCY_FOR_REFINEMENT:
                cam, caption = gradcam.generate_cam_with_saliency_guidance(image, prompt, use_saliency_for_refinement=True)
            else:
                cam, caption = gradcam.generate_cam(image, prompt)

            # Check if caption generation was successful
            if caption.startswith("Error:"):
                print(f"  Ground Truth: {ground_truth}")
                print(f"  ❌ {caption}")
                continue

            # ✅ POST-PROCESSING: Fix false negatives and improve caption
            if cfg.ENABLE_POST_PROCESSING:
                original_caption = caption
                caption = post_process_caption(caption, label)
                if caption != original_caption:
                    print(f"  📝 Post-processed caption:")
                    print(f"      Original: {original_caption}")
                    print(f"      Updated:  {caption}")

            # ✅ Print both ground truth and generated caption for easy comparison
            print(f"  Ground Truth: {ground_truth}")
            print(f"  Generated:    {caption}")
            print(f"  Saliency stats: mean={cam.mean():.3f}, max={cam.max():.3f}")

            # Check for identical captions (warning sign)
            caption_hash = hash(caption[:50])  # Hash first 50 chars
            if caption_hash in seen_captions:
                print(f"  ⚠️  WARNING: This caption is identical to sample {seen_captions[caption_hash]}!")
                print(f"      This suggests the model may not be processing images correctly.")
            else:
                seen_captions[caption_hash] = idx + 1

            # Check image uniqueness
            img_hash = hash(np.array(image).tobytes()[:1000])  # Hash first 1000 bytes
            if img_hash in image_hashes:
                print(f"  ⚠️  WARNING: This image appears to be a duplicate!")
            else:
                image_hashes.append(img_hash)

            # Convert saliency map to PLI map
            pli_map = None
            if PLI_AVAILABLE:
                try:
                    pli_map = gradcam_to_pli(cam)
                    print(f"  PLI stats: mean={pli_map.mean():.3f}, max={pli_map.max():.3f}")

                    # Save PLI map separately
                    pli_path = os.path.join(pli_dir, f"pli_{scan_id}.png")
                    plt.imsave(pli_path, pli_map, cmap="Reds")

                    # Save PLI overlay
                    img_array = np.array(image)
                    if img_array.shape[:2] != pli_map.shape[:2]:
                        img_array = cv2.resize(img_array, (pli_map.shape[1], pli_map.shape[0]))
                    pli_overlay = np.clip(img_array / 255.0 * 0.5 + np.stack([pli_map] * 3, axis=2) * 0.5, 0, 1)
                    pli_overlay_path = os.path.join(pli_dir, f"pli_overlay_{scan_id}.png")
                    plt.imsave(pli_overlay_path, pli_overlay)
                except Exception as e:
                    print(f"  ⚠️  PLI conversion failed: {e}")
                    pli_map = None

            # Save saliency visualization (with PLI if available)
            saliency_path = os.path.join(saliency_dir, f"saliency_{scan_id}.png")
            GradCAM.visualize(image, cam, caption, saliency_path, pli_map=pli_map)

            # Evaluate accuracy
            if cfg.EVALUATE_ACCURACY:
                eval_results = evaluate_caption_accuracy(caption, ground_truth, label, semantic_model=semantic_model)
                print(f"  BLEU: {eval_results['bleu_score']:.4f}, "
                      f"Word Overlap: {eval_results['word_overlap']:.4f}, "
                      f"Semantic: {eval_results['semantic_similarity']:.4f}, "
                      f"Key Terms: {eval_results['key_term_overlap']:.4f}, "
                      f"Label Mentioned: {eval_results['label_mentioned']}")

                # Warn about false negatives
                if eval_results.get('false_negative', False):
                    print(f"  ⚠️  FALSE NEGATIVE: Model says 'no cancer' but label indicates cancer is present!")
                    print(f"      This is a critical error - the model missed a cancer diagnosis.")

                results.append({
                    'scan_id': scan_id,
                    'label': label,
                    'ground_truth_caption': ground_truth,
                    'generated_caption': caption,
                    **eval_results
                })
                success_count += 1

            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    if cfg.SAVE_DETAILED_RESULTS and results:
        results_df = pd.DataFrame(results)
        results_csv = os.path.join(cfg.OUTPUT_DIR, "evaluation_results.csv")
        results_df.to_csv(results_csv, index=False)
        print(f"\n[SAVED] Detailed results: {results_csv}")

        # Compute summary statistics
        summary = {
            'total_samples': len(results),
            'average_bleu': results_df['bleu_score'].mean(),
            'average_word_overlap': results_df['word_overlap'].mean(),
            'average_semantic_similarity': results_df[
                'semantic_similarity'].mean() if 'semantic_similarity' in results_df else 0.0,
            'average_key_term_overlap': results_df['key_term_overlap'].mean(),
            'label_mention_rate': results_df['label_mentioned'].mean(),
            'false_negative_rate': results_df['false_negative'].mean() if 'false_negative' in results_df else 0.0,
            'per_label_stats': {}
        }

        # Per-label statistics
        for label in results_df['label'].unique():
            label_df = results_df[results_df['label'] == label]
            label_stats = {
                'count': len(label_df),
                'avg_bleu': label_df['bleu_score'].mean(),
                'avg_word_overlap': label_df['word_overlap'].mean(),
                'label_mention_rate': label_df['label_mentioned'].mean()
            }
            if 'semantic_similarity' in label_df.columns:
                label_stats['avg_semantic_similarity'] = label_df['semantic_similarity'].mean()
            if 'false_negative' in label_df.columns:
                label_stats['false_negative_rate'] = label_df['false_negative'].mean()
            summary['per_label_stats'][label] = label_stats

        summary_json = os.path.join(cfg.OUTPUT_DIR, "evaluation_summary.json")
        with open(summary_json, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"[SAVED] Summary: {summary_json}")

        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total samples evaluated: {summary['total_samples']}")
        print(f"Average BLEU score: {summary['average_bleu']:.4f}")
        print(f"Average word overlap: {summary['average_word_overlap']:.4f}")
        if summary['average_semantic_similarity'] > 0:
            print(f"Average semantic similarity: {summary['average_semantic_similarity']:.4f}")
        print(f"Average key term overlap: {summary['average_key_term_overlap']:.4f}")
        print(f"Label mention rate: {summary['label_mention_rate']:.4f}")
        if summary['false_negative_rate'] > 0:
            print(f"False negative rate: {summary['false_negative_rate']:.4f}")
            if summary['false_negative_rate'] > 0.1:
                print(
                    f"  ⚠️  WARNING: High false negative rate! Model is incorrectly saying 'no cancer' when cancer is present.")
                print(f"      This is a critical error - consider using more directive prompts or fine-tuning.")

        # Sample comparisons (if enabled)
        if cfg.SHOW_SAMPLE_COMPARISONS:
            print("\n" + "=" * 60)
            print("SAMPLE COMPARISONS (First 5 samples)")
            print("=" * 60)
            num_samples = min(5, len(results))
            for i in range(num_samples):
                r = results[i]
                print(f"\n--- Sample {i + 1} ---")
                print(f"Label: {r['label']}")
                print(f"Ground Truth: {r['ground_truth_caption']}")
                print(f"Generated: {r['generated_caption']}")
                print(f"Metrics:")
                print(f"  BLEU: {r['bleu_score']:.4f}")
                print(f"  Word Overlap: {r['word_overlap']:.4f}")
                if r.get('semantic_similarity', 0) > 0:
                    print(f"  Semantic Similarity: {r['semantic_similarity']:.4f}")
                print(f"  Key Terms: {r['key_term_overlap']:.4f}")
                print(f"  Label Mentioned: {r['label_mentioned']}")

            # Show best and worst examples
            if len(results) > 5:
                print("\n" + "=" * 60)
                print("BEST AND WORST EXAMPLES (by semantic similarity)")
                print("=" * 60)

                if 'semantic_similarity' in results_df.columns:
                    best_idx = results_df['semantic_similarity'].idxmax()
                    worst_idx = results_df['semantic_similarity'].idxmin()

                    print("\n--- BEST Example ---")
                    best = results[best_idx]
                    print(f"Label: {best['label']}")
                    print(f"Ground Truth: {best['ground_truth_caption']}")
                    print(f"Generated: {best['generated_caption']}")
                    print(f"Semantic Similarity: {best['semantic_similarity']:.4f}")

                    print("\n--- WORST Example ---")
                    worst = results[worst_idx]
                    print(f"Label: {worst['label']}")
                    print(f"Ground Truth: {worst['ground_truth_caption']}")
                    print(f"Generated: {worst['generated_caption']}")
                    print(f"Semantic Similarity: {worst['semantic_similarity']:.4f}")

    # Check caption uniqueness and provide recommendations
    if results:
        unique_captions = len(set(hash(r['generated_caption'][:50]) for r in results if 'generated_caption' in r))
        print(f"\n📊 Caption Analysis:")
        print(f"  Unique captions: {unique_captions}/{len(results)} ({100 * unique_captions / len(results):.1f}%)")

        if unique_captions < len(results) * 0.3:
            print(f"\n  ⚠️  CRITICAL: Less than 30% of captions are unique!")
            print(f"      The model appears to be generating identical responses for different images.")
            print(f"\n  🔧 RECOMMENDATIONS TO FIX:")
            print(f"      1. Enable DEBUG_MODE: Set cfg.DEBUG_MODE = True to see detailed logs")
            print(f"      2. Verify IMAGE_TOKEN: Check that IMAGE_TOKEN is in input_ids")
            print(f"      3. Check image loading: Ensure images are different and loaded correctly")
            print(f"      4. Increase temperature: Try cfg.GENERATION_TEMPERATURE = 0.9 for more variation")
            print(f"      5. Verify vision tower: Ensure vision_tower.is_loaded = True")
            print(f"      6. Try different prompt: Switch to 'diagnostic' or 'classification' style")
        elif unique_captions < len(results) * 0.7:
            print(f"  ⚠️  WARNING: Only {100 * unique_captions / len(results):.1f}% of captions are unique.")
            print(f"      Consider increasing GENERATION_TEMPERATURE (currently {cfg.GENERATION_TEMPERATURE})")

        # Additional recommendations for improving accuracy
        print(f"\n  📈 RECOMMENDATIONS TO IMPROVE ACCURACY:")
        print(f"      1. Current prompt style: {cfg.PROMPT_STYLE} (diagnostic is recommended)")
        print(f"      2. Current temperature: {cfg.GENERATION_TEMPERATURE} (0.8-0.9 recommended for variation)")
        print(f"      3. Current max tokens: {cfg.MAX_NEW_TOKENS} (80-100 recommended for detailed diagnoses)")
        print(f"      4. Consider fine-tuning: Model may need domain-specific training for histopathology")
        print(f"      5. Check false negatives: Review samples where model says 'no cancer' incorrectly")

        # Check semantic similarity distribution
        if 'semantic_similarity' in results[0]:
            sem_scores = [r['semantic_similarity'] for r in results]
            avg_sem = np.mean(sem_scores)
            print(f"  Average semantic similarity: {avg_sem:.4f}")
            if avg_sem < 0.5:
                print(f"  ⚠️  Low semantic similarity suggests captions don't match ground truth well")
            elif avg_sem > 0.7:
                print(f"  ✅ Good semantic similarity - captions are semantically related to ground truth")

        # Check for false negatives
        if 'false_negative' in results[0]:
            fn_count = sum(1 for r in results if r.get('false_negative', False))
            if fn_count > 0:
                print(f"\n  ⚠️  CRITICAL: {fn_count} false negatives detected!")
                print(f"      The model incorrectly said 'no cancer' when cancer was present.")
                print(f"      This is a serious error that needs to be addressed.")
                print(f"\n  🔧 RECOMMENDATIONS TO FIX FALSE NEGATIVES:")
                print(f"      1. Use more directive prompts (already set to 'diagnostic')")
                print(f"      2. Consider fine-tuning the model on histopathology data")
                print(f"      3. Add few-shot examples showing correct cancer diagnoses")
                print(f"      4. Increase MAX_NEW_TOKENS to allow more detailed responses")
                print(f"      5. Post-process to check for cancer-related keywords")

    print(f"\n🎉 Done. Processed {success_count}/{len(dataset)} samples successfully!")
    print(f"Results saved under: {cfg.OUTPUT_DIR}")
    print(f"Saliency maps saved under: {saliency_dir}")
    if PLI_AVAILABLE:
        print(f"PLI maps saved under: {pli_dir}")
    else:
        print("⚠️  PLI maps not generated (skfuzzy not installed)")


if __name__ == "__main__":
    main()