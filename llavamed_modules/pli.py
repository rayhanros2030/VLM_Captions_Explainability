"""
PLI (Probabilistic Localization Index) conversion module.
Converts Grad-CAM saliency maps to PLI maps using fuzzy logic.
Also contains baseline methods for comparison.
"""

import numpy as np
import torch
import cv2
from PIL import Image
from typing import Dict

# Import config flags
from .config import PLI_AVAILABLE

# Import skfuzzy if available
if PLI_AVAILABLE:
    import skfuzzy as fuzz


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

    # Validate input saliency map
    if saliency_map is None or saliency_map.size == 0:
        print(f"  ⚠️  PLI: Invalid saliency map (None or empty)")
        return np.zeros((100, 100), dtype=np.float32)
    
    # Ensure saliency map is valid and has variation
    saliency_map = np.clip(saliency_map, 0, 1)
    saliency_range = saliency_map.max() - saliency_map.min()
    
    if saliency_range < 1e-6:
        print(f"  ⚠️  PLI: Saliency map is uniform (range={saliency_range:.6f}), using fallback")
        pli_map = saliency_map.copy()
        if pli_map.max() > 0:
            pli_map = pli_map / pli_map.max()
        return pli_map

    try:
        # Universe of discourse for fuzzy membership
        x = np.linspace(0, 1, 100)

        # Define fuzzy membership functions
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

        # Fuzzy inference rules - weighted aggregation
        pli_flat = (
            0.1 * mu_not_involved +
            0.3 * mu_very_low +
            0.5 * mu_moderate +
            0.7 * mu_high +
            1.0 * mu_complete
        )

        # Validate PLI computation
        pli_min = pli_flat.min()
        pli_max = pli_flat.max()
        pli_range = pli_max - pli_min
        
        if pli_range < 1e-6:
            print(f"  ⚠️  PLI: Computed PLI map is uniform (range={pli_range:.6f}), using saliency as fallback")
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
        pli_map = saliency_map.copy()
        if pli_map.max() > 0:
            pli_map = pli_map / pli_map.max()
        return pli_map


def generate_raw_gradcam(saliency_map: np.ndarray) -> np.ndarray:
    """
    Raw Grad-CAM baseline (no PLI conversion).
    Just returns the normalized saliency map.
    """
    saliency_map = np.clip(saliency_map, 0, 1)
    if saliency_map.max() > saliency_map.min():
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-10)
    return saliency_map


def generate_attention_saliency(model, tokenizer, image: Image.Image, prompt: str, 
                                image_processor, model_dtype, conv_templates, 
                                tokenizer_image_token, IMAGE_TOKEN_INDEX) -> np.ndarray:
    """
    Attention-based saliency baseline.
    Uses attention weights from the model to generate saliency maps.
    """
    device = next(model.parameters()).device
    
    # Process image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    original_size = image.size
    if image.size != (336, 336):
        image = image.resize((336, 336), Image.Resampling.LANCZOS)
    
    processed = image_processor(image, return_tensors="pt")
    image_tensor = processed["pixel_values"].to(device=device, dtype=model_dtype)
    
    # Build prompt
    conv = conv_templates["vicuna_v1"].copy()
    if "<image>" not in prompt:
        prompt_with_image = "<image>\n" + prompt
    else:
        prompt_with_image = prompt
    conv.append_message(conv.roles[0], prompt_with_image)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()
    
    try:
        input_ids = tokenizer_image_token(
            prompt_text, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
    except TypeError:
        input_ids = tokenizer_image_token(prompt_text, tokenizer, return_tensors="pt")
    
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor([input_ids], device=device)
    elif input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device)
    
    # Forward pass with attention
    model.eval()
    with torch.no_grad():
        outputs = model(images=image_tensor, input_ids=input_ids, output_attentions=True)
    
    # Extract attention weights (if available)
    if hasattr(outputs, 'attentions') and outputs.attentions is not None:
        attentions = outputs.attentions
        if len(attentions) > 0:
            # Get last layer attention
            last_attention = attentions[-1]  # [batch, heads, seq_len, seq_len]
            # Average over heads
            attention_weights = last_attention.mean(dim=1)  # [batch, seq_len, seq_len]
            
            batch_size, seq_len, _ = attention_weights.shape
            num_image_patches = 24 * 24  # Assume 24x24 = 576 patches
            if seq_len > num_image_patches:
                # Extract attention from image patches
                image_attention = attention_weights[0, :num_image_patches, num_image_patches:].mean(dim=1)
                # Reshape to spatial dimensions
                if num_image_patches == 576:
                    attention_map = image_attention[:576].reshape(24, 24).cpu().numpy()
                else:
                    side = int(np.sqrt(num_image_patches))
                    attention_map = image_attention[:num_image_patches].reshape(side, side).cpu().numpy()
                
                # Normalize
                if attention_map.max() > attention_map.min():
                    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-10)
                
                # Resize to original image size
                attention_map = cv2.resize(attention_map, (original_size[0], original_size[1]), interpolation=cv2.INTER_CUBIC)
                return attention_map.astype(np.float32)
    
    # Fallback: return uniform map
    return np.ones((original_size[1], original_size[0]), dtype=np.float32) * 0.5

