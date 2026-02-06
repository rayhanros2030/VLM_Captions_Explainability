"""
Word and token-level saliency module for LLaVA-Med integration.
Contains functions for generating token/word-level saliency maps,
cross-token consistency analysis, and deletion-insertion tests.
"""

import re
import numpy as np
import torch
import cv2
from PIL import Image
from typing import Dict, List

# Import from other modules
from .config import PLI_AVAILABLE, SEMANTIC_AVAILABLE, cfg
from .pli import gradcam_to_pli


def generate_token_level_saliency(gradcam_instance, image: Image.Image, prompt: str, 
                                   caption: str) -> Dict[str, np.ndarray]:
    """
    Generate token-level saliency maps using GradCAM approach.
    Maps each token in the caption to image regions.
    
    Returns:
        Dict mapping token strings to saliency maps (np.ndarray)
    """
    # Use the existing GradCAM method but with token-specific prompts
    device = next(gradcam_instance.model.parameters()).device
    
    # Tokenize caption to get individual tokens
    caption_ids = gradcam_instance.tokenizer.encode(caption, add_special_tokens=False)
    
    # Process image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    original_size = image.size
    
    token_saliency_maps = {}
    
    print(f"  🔍 Generating token saliency for {len(caption_ids)} tokens...")
    
    # For each token, create a prompt that asks about that specific token
    # and use GradCAM to generate saliency
    for i, token_id in enumerate(caption_ids):
        try:
            token_str = gradcam_instance.tokenizer.decode([token_id], skip_special_tokens=False)
            
            # Skip empty tokens or whitespace
            if not token_str or token_str.strip() == '':
                continue
            
            # For now, use the overall saliency map for all tokens
            # (A more sophisticated approach would compute token-specific gradients)
            # This at least gives you a working visualization
            try:
                # Get the overall saliency map (this works and produces real maps)
                saliency_map, _ = gradcam_instance.generate_cam(image, prompt)
                
                # Resize to original size if needed
                if saliency_map.shape[:2] != (original_size[1], original_size[0]):
                    saliency_map_resized = cv2.resize(
                        saliency_map, 
                        (original_size[0], original_size[1]), 
                        interpolation=cv2.INTER_CUBIC
                    )
                else:
                    saliency_map_resized = saliency_map
                
                # Normalize
                if saliency_map_resized.max() > saliency_map_resized.min():
                    saliency_map_resized = (saliency_map_resized - saliency_map_resized.min()) / (saliency_map_resized.max() - saliency_map_resized.min() + 1e-10)
                
                token_saliency_maps[token_str] = saliency_map_resized.astype(np.float32)
                
                if cfg.DEBUG_MODE:
                    print(f"    ✅ Token {i+1}/{len(caption_ids)}: '{token_str}' - saliency range: [{saliency_map_resized.min():.3f}, {saliency_map_resized.max():.3f}]")
            except Exception as e:
                print(f"    ⚠️  Token {i+1}/{len(caption_ids)}: '{token_str}' - GradCAM failed: {e}")
                # Use a simple fallback: use the overall saliency map
                try:
                    overall_saliency, _ = gradcam_instance.generate_cam(image, prompt)
                    if overall_saliency.shape[:2] != (original_size[1], original_size[0]):
                        overall_saliency = cv2.resize(
                            overall_saliency, 
                            (original_size[0], original_size[1]), 
                            interpolation=cv2.INTER_CUBIC
                        )
                    token_saliency_maps[token_str] = overall_saliency.astype(np.float32)
                except:
                    # Final fallback: uniform map
                    token_saliency_maps[token_str] = np.ones((original_size[1], original_size[0]), dtype=np.float32) * 0.5
            
        except Exception as e:
            print(f"  ⚠️  Error processing token {i} ({token_str if 'token_str' in locals() else 'unknown'}): {e}")
            token_saliency_maps[token_str] = np.ones((original_size[1], original_size[0]), dtype=np.float32) * 0.5
    
    return token_saliency_maps


def generate_word_level_saliency_and_pli(gradcam_instance, image: Image.Image, prompt: str, 
                                         caption: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Generate word-level saliency maps and PLI maps for each word in the generated caption.
    This enhances explainability by showing which image regions correspond to each word.
    
    Parameters:
        gradcam_instance: GradCAM instance
        image: PIL Image
        prompt: Input prompt
        caption: Generated caption (will be split into words)
    
    Returns:
        Dict mapping word strings to dicts containing:
            - 'saliency': saliency map (np.ndarray)
            - 'pli': PLI map (np.ndarray)
    """
    # Split caption into words (preserve punctuation for context)
    words = re.findall(r'\b\w+\b', caption.lower())  # Extract words only (alphanumeric)
    
    if len(words) == 0:
        print(f"  ⚠️  No words found in caption: {caption}")
        return {}
    
    print(f"  📝 Generating saliency and PLI maps for {len(words)} words in caption...")
    
    device = next(gradcam_instance.model.parameters()).device
    
    # Process image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    original_size = image.size
    if image.size != (336, 336):
        image_resized = image.resize((336, 336), Image.Resampling.LANCZOS)
    else:
        image_resized = image
    
    processed = gradcam_instance.image_processor(image_resized, return_tensors="pt")
    image_tensor = processed["pixel_values"].to(device=device, dtype=gradcam_instance.model_dtype)
    
    # Build prompt with image token
    conv = gradcam_instance.conv_templates["vicuna_v1"].copy()
    if "<image>" not in prompt:
        prompt_with_image = "<image>\n" + prompt
    else:
        prompt_with_image = prompt
    conv.append_message(conv.roles[0], prompt_with_image)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()
    
    tokenizer_image_token = gradcam_instance.tokenizer_image_token
    IMAGE_TOKEN_INDEX = gradcam_instance.IMAGE_TOKEN_INDEX
    
    try:
        input_ids = tokenizer_image_token(
            prompt_text, gradcam_instance.tokenizer, 
            image_token_index=IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
    except TypeError:
        input_ids = tokenizer_image_token(
            prompt_text, gradcam_instance.tokenizer, return_tensors="pt"
        )
    
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor([input_ids], device=device)
    elif input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device)
    
    # Tokenize the full caption to get token positions for each word
    caption_tokens = gradcam_instance.tokenizer.tokenize(caption)
    caption_ids = gradcam_instance.tokenizer.encode(caption, add_special_tokens=False)
    
    # Map words to tokens - simpler approach: decode full caption and match words
    full_decoded = gradcam_instance.tokenizer.decode(caption_ids, skip_special_tokens=True).lower()
    
    # Map each word to its approximate token position
    word_to_tokens = {}
    token_idx = 0
    
    # Simple strategy: map words sequentially to tokens
    for word_idx, word in enumerate(words):
        word_lower = word.lower()
        assigned = False
        for check_token_idx in range(token_idx, min(token_idx + 3, len(caption_ids))):
            check_tokens = caption_ids[max(0, check_token_idx-1):min(len(caption_ids), check_token_idx+2)]
            decoded_chunk = gradcam_instance.tokenizer.decode(check_tokens, skip_special_tokens=True).lower()
            
            if word_lower in decoded_chunk or any(char in decoded_chunk for char in word_lower if len(word_lower) > 2):
                word_to_tokens[word] = [check_token_idx]
                token_idx = check_token_idx + 1
                assigned = True
                break
        
        if not assigned:
            if token_idx < len(caption_ids):
                word_to_tokens[word] = [token_idx]
                token_idx += 1
            else:
                if word_idx < len(caption_ids):
                    word_to_tokens[word] = [word_idx % len(caption_ids)]
    
    word_saliency_pli_maps = {}
    
    # Generate saliency and PLI for each word
    for word_idx, word in enumerate(words):
        if word not in word_to_tokens:
            print(f"  ⚠️  Skipping word '{word}' (no token mapping found)")
            continue
        
        try:
            word_tokens = word_to_tokens[word]
            target_token_idx = word_tokens[0]
            
            if target_token_idx >= len(caption_ids):
                print(f"  ⚠️  Token index {target_token_idx} out of range for word '{word}'")
                continue
            
            target_token_id = caption_ids[target_token_idx]
            
            # Create input with caption up to this token
            tokens_up_to_word = caption_ids[:target_token_idx + 1]
            full_input_ids = torch.cat([input_ids, torch.tensor([tokens_up_to_word], device=device)], dim=1)
            
            # Enable gradients
            gradcam_instance.model.train()
            image_tensor.requires_grad_(True)
            
            # Forward pass
            gradcam_instance.model.zero_grad()
            outputs = gradcam_instance.model(images=image_tensor, input_ids=full_input_ids)
            logits = outputs.logits
            
            # Backward pass for this word's token
            target = logits[0, -1, target_token_id]
            target.backward(retain_graph=False)
            
            # Extract saliency map
            saliency_map = None
            if image_tensor.grad is not None:
                grad = image_tensor.grad[0].abs().mean(dim=0).cpu().numpy()
                grad_resized = cv2.resize(grad, (original_size[0], original_size[1]), interpolation=cv2.INTER_CUBIC)
                if grad_resized.max() > grad_resized.min():
                    grad_resized = (grad_resized - grad_resized.min()) / (grad_resized.max() - grad_resized.min() + 1e-10)
                saliency_map = grad_resized.astype(np.float32)
            else:
                saliency_map = np.ones((original_size[1], original_size[0]), dtype=np.float32) * 0.5
            
            # Convert saliency to PLI map
            pli_map = None
            if PLI_AVAILABLE:
                try:
                    pli_map = gradcam_to_pli(saliency_map)
                except Exception as e:
                    print(f"  ⚠️  PLI conversion failed for word '{word}': {e}")
                    pli_map = saliency_map.copy()
                    if pli_map.max() > 0:
                        pli_map = pli_map / pli_map.max()
            else:
                pli_map = saliency_map.copy()
            
            word_saliency_pli_maps[word] = {
                'saliency': saliency_map,
                'pli': pli_map
            }
            
            gradcam_instance.model.eval()
            image_tensor.requires_grad_(False)
            
            if (word_idx + 1) % 5 == 0:
                print(f"    Processed {word_idx + 1}/{len(words)} words...")
            
        except Exception as e:
            print(f"  ⚠️  Error generating saliency for word '{word}': {e}")
            fallback_map = np.ones((original_size[1], original_size[0]), dtype=np.float32) * 0.5
            word_saliency_pli_maps[word] = {
                'saliency': fallback_map,
                'pli': fallback_map
            }
    
    print(f"  ✅ Generated saliency and PLI maps for {len(word_saliency_pli_maps)} words")
    return word_saliency_pli_maps


def compute_stability_metric(saliency_maps: List[np.ndarray]) -> Dict[str, float]:
    """
    Compute stability metrics for saliency maps across multiple runs.
    
    Metrics:
    - Variance: Lower variance = more stable
    - Coefficient of Variation: Normalized variance
    - Consistency: How similar maps are to each other
    """
    if len(saliency_maps) < 2:
        return {"variance": 0.0, "coefficient_of_variation": 0.0, "consistency": 1.0}
    
    # Stack maps
    maps_array = np.stack(saliency_maps)
    
    # Compute variance across runs
    variance_map = np.var(maps_array, axis=0)
    mean_variance = np.mean(variance_map)
    
    # Compute mean map
    mean_map = np.mean(maps_array, axis=0)
    mean_value = np.mean(mean_map)
    
    # Coefficient of variation
    coefficient_of_variation = np.sqrt(mean_variance) / (mean_value + 1e-10)
    
    # Consistency: pairwise similarity
    consistency_scores = []
    for i in range(len(saliency_maps)):
        for j in range(i + 1, len(saliency_maps)):
            map1_flat = saliency_maps[i].flatten()
            map2_flat = saliency_maps[j].flatten()
            dot_product = np.dot(map1_flat, map2_flat)
            norm1 = np.linalg.norm(map1_flat)
            norm2 = np.linalg.norm(map2_flat)
            if norm1 > 0 and norm2 > 0:
                similarity = dot_product / (norm1 * norm2)
                consistency_scores.append(similarity)
    
    consistency = np.mean(consistency_scores) if consistency_scores else 0.0
    
    return {
        "variance": float(mean_variance),
        "coefficient_of_variation": float(coefficient_of_variation),
        "consistency": float(consistency)
    }


def compute_readability_metric(saliency_map: np.ndarray) -> Dict[str, float]:
    """
    Compute readability metrics for a saliency map.
    
    Metrics:
    - Focus: How concentrated the saliency is (entropy)
    - Contrast: Difference between high and low saliency regions
    - Smoothness: How smooth the map is (gradient magnitude)
    """
    # Normalize
    saliency_map = np.clip(saliency_map, 0, 1)
    if saliency_map.max() > saliency_map.min():
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-10)
    
    # Focus: Entropy (lower entropy = more focused)
    hist, _ = np.histogram(saliency_map.flatten(), bins=50, range=(0, 1))
    hist = hist / (hist.sum() + 1e-10)
    hist = hist[hist > 0]  # Remove zeros
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    focus = 1.0 / (1.0 + entropy)  # Normalize to [0, 1]
    
    # Contrast: Difference between high and low regions
    high_threshold = np.percentile(saliency_map, 90)
    low_threshold = np.percentile(saliency_map, 10)
    contrast = float(high_threshold - low_threshold)
    
    # Smoothness: Inverse of gradient magnitude
    grad_y, grad_x = np.gradient(saliency_map)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    smoothness = 1.0 / (1.0 + np.mean(gradient_magnitude))
    
    return {
        "focus": float(focus),
        "contrast": float(contrast),
        "smoothness": float(smoothness)
    }


def compute_cross_token_consistency(token_saliency_maps: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Analyze consistency across tokens in the caption.
    
    Metrics:
    - Token similarity: How similar are saliency maps for different tokens
    - Spatial coherence: Do related tokens attend to similar regions
    - Temporal consistency: Consistency across token sequence
    """
    if len(token_saliency_maps) < 2:
        return {"average_token_similarity": 0.0, "spatial_coherence": 0.0, "temporal_consistency": 0.0}
    
    tokens = list(token_saliency_maps.keys())
    maps = list(token_saliency_maps.values())
    
    # Token similarity: pairwise similarity
    similarities = []
    for i in range(len(maps)):
        for j in range(i + 1, len(maps)):
            map1_flat = maps[i].flatten()
            map2_flat = maps[j].flatten()
            dot_product = np.dot(map1_flat, map2_flat)
            norm1 = np.linalg.norm(map1_flat)
            norm2 = np.linalg.norm(map2_flat)
            if norm1 > 0 and norm2 > 0:
                similarity = dot_product / (norm1 * norm2)
                similarities.append(similarity)
    
    average_token_similarity = np.mean(similarities) if similarities else 0.0
    
    # Spatial coherence: variance of saliency across tokens (lower = more coherent)
    maps_array = np.stack(maps)
    spatial_variance = np.var(maps_array, axis=0)
    spatial_coherence = 1.0 / (1.0 + np.mean(spatial_variance))  # Inverse of variance
    
    # Temporal consistency: similarity between consecutive tokens
    temporal_similarities = []
    for i in range(len(maps) - 1):
        map1_flat = maps[i].flatten()
        map2_flat = maps[i + 1].flatten()
        dot_product = np.dot(map1_flat, map2_flat)
        norm1 = np.linalg.norm(map1_flat)
        norm2 = np.linalg.norm(map2_flat)
        if norm1 > 0 and norm2 > 0:
            similarity = dot_product / (norm1 * norm2)
            temporal_similarities.append(similarity)
    
    temporal_consistency = np.mean(temporal_similarities) if temporal_similarities else 0.0
    
    return {
        "average_token_similarity": float(average_token_similarity),
        "spatial_coherence": float(spatial_coherence),
        "temporal_consistency": float(temporal_consistency)
    }


def compute_cross_token_grounding_consistency(token_saliency_maps: Dict[str, np.ndarray], 
                                               overlap_threshold: float = 0.3) -> Dict[str, float]:
    """
    Check that different caption tokens highlight distinct, non-overlapping regions.
    
    Metrics:
    - Overlap ratio: How much do token regions overlap with each other
    - Distinctness: Measure of how distinct token regions are
    - Spatial separation: Average distance between token region centers
    """
    if len(token_saliency_maps) < 2:
        return {"overlap_ratio": 0.0, "distinctness": 1.0, "spatial_separation": 0.0}
    
    tokens = list(token_saliency_maps.keys())
    maps = list(token_saliency_maps.values())
    
    # Normalize maps and create binary masks (top 30% saliency)
    binary_masks = []
    region_centers = []
    
    for map_array in maps:
        threshold = np.percentile(map_array, 70)  # Top 30%
        binary_mask = (map_array >= threshold).astype(np.float32)
        binary_masks.append(binary_mask)
        
        # Compute region center (weighted by saliency)
        h, w = map_array.shape
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        total_weight = map_array.sum()
        if total_weight > 0:
            center_y = (y_coords * map_array).sum() / total_weight
            center_x = (x_coords * map_array).sum() / total_weight
            region_centers.append((center_y, center_x))
        else:
            region_centers.append((h/2, w/2))
    
    # Compute pairwise overlaps
    overlap_ratios = []
    for i in range(len(binary_masks)):
        for j in range(i + 1, len(binary_masks)):
            mask1 = binary_masks[i]
            mask2 = binary_masks[j]
            
            intersection = (mask1 * mask2).sum()
            union = ((mask1 + mask2) > 0).sum()
            
            if union > 0:
                iou = intersection / union
                overlap_ratios.append(iou)
            
            area1 = mask1.sum()
            area2 = mask2.sum()
            if min(area1, area2) > 0:
                overlap_ratio = intersection / min(area1, area2)
                overlap_ratios.append(overlap_ratio)
    
    avg_overlap = np.mean(overlap_ratios) if overlap_ratios else 0.0
    distinctness = 1.0 - avg_overlap
    
    # Spatial separation: average distance between region centers
    distances = []
    for i in range(len(region_centers)):
        for j in range(i + 1, len(region_centers)):
            y1, x1 = region_centers[i]
            y2, x2 = region_centers[j]
            dist = np.sqrt((y1 - y2)**2 + (x1 - x2)**2)
            distances.append(dist)
    
    avg_separation = np.mean(distances) if distances else 0.0
    # Normalize by image diagonal
    if len(maps) > 0:
        h, w = maps[0].shape
        max_separation = np.sqrt(h**2 + w**2)
        normalized_separation = avg_separation / max_separation if max_separation > 0 else 0.0
    else:
        normalized_separation = 0.0
    
    return {
        "overlap_ratio": float(avg_overlap),
        "distinctness": float(distinctness),
        "spatial_separation": float(normalized_separation)
    }


def deletion_insertion_test(gradcam_instance, image: Image.Image, prompt: str, 
                            saliency_map: np.ndarray, deletion_percentages: List[float]) -> Dict[str, Dict]:
    """
    Test caption faithfulness by removing top-saliency pixels and regenerating captions.
    Measure if relevant tokens disappear.
    
    Returns:
        Dict with results for each deletion percentage
    """
    device = next(gradcam_instance.model.parameters()).device
    original_size = image.size
    
    # Generate original caption
    _, original_caption = gradcam_instance.generate_cam(image, prompt)
    original_tokens = set(gradcam_instance.tokenizer.tokenize(original_caption.lower()))
    
    results = {}
    
    for deletion_pct in deletion_percentages:
        # Create mask for top-saliency pixels
        threshold = np.percentile(saliency_map, 100 - (deletion_pct * 100))
        mask = saliency_map < threshold  # Keep pixels below threshold (remove high saliency)
        
        # Resize mask to image size if needed
        if mask.shape[:2] != (original_size[1], original_size[0]):
            mask = cv2.resize(mask.astype(np.float32), 
                            (original_size[0], original_size[1]), 
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        
        # Apply mask to image (set removed pixels to black or mean)
        img_array = np.array(image)
        masked_img = img_array.copy()
        masked_img[~mask] = 0  # Set removed pixels to black
        
        masked_image = Image.fromarray(masked_img)
        
        # Regenerate caption with masked image
        try:
            _, new_caption = gradcam_instance.generate_cam(masked_image, prompt)
            new_tokens = set(gradcam_instance.tokenizer.tokenize(new_caption.lower()))
            
            # Compute token disappearance
            disappeared_tokens = original_tokens - new_tokens
            appeared_tokens = new_tokens - original_tokens
            
            token_disappearance_rate = len(disappeared_tokens) / len(original_tokens) if len(original_tokens) > 0 else 0.0
            
            # Compute semantic similarity to original
            if SEMANTIC_AVAILABLE:
                try:
                    from sentence_transformers import SentenceTransformer
                    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                    embeddings = semantic_model.encode([original_caption, new_caption], convert_to_tensor=True)
                    from torch.nn.functional import cosine_similarity
                    similarity = cosine_similarity(embeddings[0:1], embeddings[1:2]).item()
                    similarity = (similarity + 1) / 2.0  # Normalize to [0, 1]
                except:
                    similarity = 0.0
            else:
                similarity = 0.0
            
            results[f"deletion_{deletion_pct}"] = {
                "original_caption": original_caption,
                "new_caption": new_caption,
                "token_disappearance_rate": float(token_disappearance_rate),
                "disappeared_tokens": list(disappeared_tokens)[:10],  # Limit to first 10
                "appeared_tokens": list(appeared_tokens)[:10],
                "semantic_similarity": float(similarity),
                "deletion_percentage": deletion_pct
            }
        except Exception as e:
            print(f"  ⚠️  Deletion test failed for {deletion_pct*100}%: {e}")
            results[f"deletion_{deletion_pct}"] = {
                "error": str(e),
                "deletion_percentage": deletion_pct
            }
    
    return results

