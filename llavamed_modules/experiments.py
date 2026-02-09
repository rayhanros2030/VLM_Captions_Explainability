"""
Extended experiments module for LLaVA-Med integration.
Contains robustness tests, multi-scale grounding, fuzzy logic ablation studies,
and pathology-specific augmentation tests.
"""

import numpy as np
import cv2
from PIL import Image
from typing import Dict, List

# Import from other modules
from .config import PLI_AVAILABLE
from .pli import gradcam_to_pli, generate_raw_gradcam
from .saliency import (
    compute_readability_metric, 
    compute_stability_metric,
    generate_token_level_saliency
)


def apply_stain_normalization(image: Image.Image, method: str = "macenko") -> Image.Image:
    """
    Apply stain normalization (important for histopathology).
    Simplified implementation - in practice, use specialized libraries.
    """
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    if method == "macenko":
        # Simplified Macenko normalization
        img_normalized = img_array.copy()
        # Apply channel-wise normalization
        for c in range(3):
            channel = img_normalized[:, :, c]
            mean = channel.mean()
            std = channel.std()
            if std > 0:
                img_normalized[:, :, c] = (channel - mean) / std * 0.1 + 0.5
        img_normalized = np.clip(img_normalized, 0, 1)
    else:
        # Simple histogram equalization per channel
        img_normalized = img_array.copy()
        for c in range(3):
            channel = img_normalized[:, :, c]
            hist, bins = np.histogram(channel.flatten(), bins=256, range=(0, 1))
            cdf = hist.cumsum()
            cdf = cdf / cdf[-1] if cdf[-1] > 0 else cdf
            img_normalized[:, :, c] = np.interp(channel, bins[:-1], cdf)
    
    img_normalized = (img_normalized * 255).astype(np.uint8)
    return Image.fromarray(img_normalized)


def apply_pathology_augmentations(image: Image.Image, augmentation_type: str) -> Image.Image:
    """Apply pathology-specific augmentations."""
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    if augmentation_type == "blur":
        img_array = cv2.GaussianBlur(img_array, (15, 15), 0)
    elif augmentation_type == "brightness":
        factor = np.random.uniform(0.7, 1.3)
        img_array = np.clip(img_array * factor, 0, 1)
    elif augmentation_type == "contrast":
        factor = np.random.uniform(0.8, 1.2)
        mean = img_array.mean()
        img_array = np.clip((img_array - mean) * factor + mean, 0, 1)
    elif augmentation_type == "color_jitter":
        for c in range(3):
            factor = np.random.uniform(0.9, 1.1)
            img_array[:, :, c] = np.clip(img_array[:, :, c] * factor, 0, 1)
    elif augmentation_type == "elastic":
        # Elastic deformation (simplified)
        h, w = img_array.shape[:2]
        dx = np.random.randn(h, w) * 2
        dy = np.random.randn(h, w) * 2
        # Apply deformation (simplified - in practice use scipy.ndimage)
        pass
    
    img_array = np.clip(img_array, 0, 1)
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)


def test_pathology_robustness(gradcam_instance, image: Image.Image, prompt: str,
                              saliency_map: np.ndarray) -> Dict[str, Dict]:
    """Test robustness to stain normalization and pathology-specific augmentations."""
    results = {}
    
    # Test stain normalization
    try:
        normalized_image = apply_stain_normalization(image, method="macenko")
        cam_normalized, caption_normalized = gradcam_instance.generate_cam(normalized_image, prompt)
        
        # Compute similarity to original saliency
        cam_resized = cv2.resize(saliency_map, (cam_normalized.shape[1], cam_normalized.shape[0]), 
                                interpolation=cv2.INTER_CUBIC)
        similarity = np.corrcoef(cam_resized.flatten(), cam_normalized.flatten())[0, 1]
        
        results["stain_normalization"] = {
            "saliency_similarity": float(similarity),
            "caption": caption_normalized,
            "readability": compute_readability_metric(cam_normalized)
        }
    except Exception as e:
        print(f"  ⚠️  Stain normalization test failed: {e}")
        results["stain_normalization"] = {"error": str(e)}
    
    # Test augmentations
    augmentation_types = ["blur", "brightness", "contrast", "color_jitter"]
    for aug_type in augmentation_types:
        try:
            augmented_image = apply_pathology_augmentations(image, aug_type)
            cam_aug, caption_aug = gradcam_instance.generate_cam(augmented_image, prompt)
            
            cam_resized = cv2.resize(saliency_map, (cam_aug.shape[1], cam_aug.shape[0]), 
                                    interpolation=cv2.INTER_CUBIC)
            similarity = np.corrcoef(cam_resized.flatten(), cam_aug.flatten())[0, 1]
            
            results[f"augmentation_{aug_type}"] = {
                "saliency_similarity": float(similarity),
                "caption": caption_aug,
                "readability": compute_readability_metric(cam_aug)
            }
        except Exception as e:
            print(f"  ⚠️  Augmentation test ({aug_type}) failed: {e}")
            results[f"augmentation_{aug_type}"] = {"error": str(e)}
    
    return results


def test_multi_scale_token_grounding(gradcam_instance, image: Image.Image, prompt: str,
                                     scales: List[int], target_tokens: List[str] = None) -> Dict[str, Dict]:
    """
    Evaluate whether tokens (e.g., "nuclei") ground consistently across magnifications.
    BreakHis-style evaluation.
    """
    if target_tokens is None:
        target_tokens = ["nuclei", "nucleus", "cell", "tissue", "gland", "carcinoma", "adenocarcinoma"]
    
    results = {}
    original_size = image.size
    
    # Generate token saliency maps at each scale
    scale_token_maps = {}
    
    for scale in scales:
        scaled_image = image.resize((scale, scale), Image.Resampling.LANCZOS)
        cam, caption = gradcam_instance.generate_cam(scaled_image, prompt)
        
        # Generate token-level saliency
        try:
            token_maps = generate_token_level_saliency(gradcam_instance, scaled_image, prompt, caption)
            
            # Resize token maps back to original size for comparison
            resized_token_maps = {}
            for token, token_map in token_maps.items():
                resized_map = cv2.resize(token_map, (original_size[0], original_size[1]), 
                                        interpolation=cv2.INTER_CUBIC)
                resized_token_maps[token] = resized_map
            
            scale_token_maps[scale] = {
                "token_maps": resized_token_maps,
                "caption": caption,
                "saliency_map": cv2.resize(cam, (original_size[0], original_size[1]), 
                                          interpolation=cv2.INTER_CUBIC)
            }
        except Exception as e:
            print(f"  ⚠️  Token grounding failed for scale {scale}: {e}")
            scale_token_maps[scale] = {"error": str(e)}
    
    # Compute consistency across scales for target tokens
    if len(scale_token_maps) > 1:
        for token in target_tokens:
            token_consistency = []
            token_maps_at_scales = []
            
            for scale, scale_data in scale_token_maps.items():
                if "token_maps" in scale_data and token in scale_data["token_maps"]:
                    token_maps_at_scales.append(scale_data["token_maps"][token])
            
            if len(token_maps_at_scales) >= 2:
                # Compute pairwise consistency
                for i in range(len(token_maps_at_scales)):
                    for j in range(i + 1, len(token_maps_at_scales)):
                        map1 = token_maps_at_scales[i].flatten()
                        map2 = token_maps_at_scales[j].flatten()
                        correlation = np.corrcoef(map1, map2)[0, 1]
                        if not np.isnan(correlation):
                            token_consistency.append(correlation)
                
                avg_consistency = np.mean(token_consistency) if token_consistency else 0.0
                results[f"token_{token}"] = {
                    "cross_scale_consistency": float(avg_consistency),
                    "num_scales": len(token_maps_at_scales)
                }
    
    results["scales_tested"] = list(scales)
    return results


def gradcam_to_pli_membership_only(saliency_map: np.ndarray) -> np.ndarray:
    """Ablation: Fuzzy membership only (without inference rules)."""
    if not PLI_AVAILABLE:
        return saliency_map
    
    saliency_map = np.clip(saliency_map, 0, 1)
    saliency_range = saliency_map.max() - saliency_map.min()
    
    if saliency_range < 1e-6:
        return saliency_map
    
    try:
        import skfuzzy as fuzz
        x = np.linspace(0, 1, 100)
        
        # Define fuzzy membership functions
        not_involved = fuzz.trimf(x, [0.0, 0.0, 0.1])
        very_low = fuzz.trimf(x, [0.1, 0.2, 0.3])
        moderate = fuzz.trimf(x, [0.3, 0.4, 0.5])
        high = fuzz.trimf(x, [0.5, 0.6, 0.7])
        complete = fuzz.trimf(x, [0.7, 0.85, 1.0])
        
        flat_saliency = saliency_map.flatten()
        
        # Calculate membership degrees
        mu_not_involved = fuzz.interp_membership(x, not_involved, flat_saliency)
        mu_very_low = fuzz.interp_membership(x, very_low, flat_saliency)
        mu_moderate = fuzz.interp_membership(x, moderate, flat_saliency)
        mu_high = fuzz.interp_membership(x, high, flat_saliency)
        mu_complete = fuzz.interp_membership(x, complete, flat_saliency)
        
        # Use maximum membership (no weighted aggregation)
        membership_only = np.maximum.reduce([
            mu_not_involved, mu_very_low, mu_moderate, mu_high, mu_complete
        ])
        
        # Normalize
        if membership_only.max() > membership_only.min():
            membership_only = (membership_only - membership_only.min()) / (membership_only.max() - membership_only.min() + 1e-10)
        
        return membership_only.reshape(saliency_map.shape)
    except Exception as e:
        print(f"  ⚠️  Membership-only conversion failed: {e}")
        return saliency_map


def fuzzy_logic_ablation_study(saliency_map: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Ablation study comparing:
    1. Raw saliency (baseline)
    2. Fuzzy membership only
    3. Fuzzy membership + rules (full PLI)
    """
    results = {}
    
    # 1. Raw saliency
    raw_saliency = generate_raw_gradcam(saliency_map)
    results["raw"] = raw_saliency
    
    # 2. Fuzzy membership only
    if PLI_AVAILABLE:
        membership_only = gradcam_to_pli_membership_only(saliency_map)
        results["membership_only"] = membership_only
        
        # 3. Full PLI (membership + rules)
        full_pli = gradcam_to_pli(saliency_map)
        results["full_pli"] = full_pli
    
    return results


def compare_fuzzy_ablation(ablation_results: Dict[str, np.ndarray]) -> Dict[str, Dict]:
    """Compare the different ablation variants."""
    comparison = {}
    
    for method_name, saliency_map in ablation_results.items():
        readability = compute_readability_metric(saliency_map)
        comparison[method_name] = readability
    
    # Compute relative improvements
    if "raw" in comparison and "full_pli" in comparison:
        raw_metrics = comparison["raw"]
        pli_metrics = comparison["full_pli"]
        
        comparison["improvement"] = {
            "focus": pli_metrics["focus"] - raw_metrics["focus"],
            "contrast": pli_metrics["contrast"] - raw_metrics["contrast"],
            "smoothness": pli_metrics["smoothness"] - raw_metrics["smoothness"]
        }
        
        if "membership_only" in comparison:
            membership_metrics = comparison["membership_only"]
            comparison["membership_contribution"] = {
                "focus": membership_metrics["focus"] - raw_metrics["focus"],
                "contrast": membership_metrics["contrast"] - raw_metrics["contrast"],
                "smoothness": membership_metrics["smoothness"] - raw_metrics["smoothness"]
            }
            comparison["rules_contribution"] = {
                "focus": pli_metrics["focus"] - membership_metrics["focus"],
                "contrast": pli_metrics["contrast"] - membership_metrics["contrast"],
                "smoothness": pli_metrics["smoothness"] - membership_metrics["smoothness"]
            }
    
    return comparison


def add_noise_to_image(image: Image.Image, noise_level: float) -> Image.Image:
    """Add Gaussian noise to image for robustness testing."""
    img_array = np.array(image, dtype=np.float32) / 255.0
    noise = np.random.normal(0, noise_level, img_array.shape)
    noisy_img = np.clip(img_array + noise, 0, 1)
    noisy_img = (noisy_img * 255).astype(np.uint8)
    return Image.fromarray(noisy_img)


def test_robustness(gradcam_instance, image: Image.Image, prompt: str, 
                    noise_levels: List[float], num_runs: int = 3) -> Dict[str, Dict]:
    """
    Test robustness of saliency maps to noise.
    
    Returns:
        Dict with metrics for each noise level
    """
    results = {}
    
    # Baseline (no noise)
    baseline_cam, _ = gradcam_instance.generate_cam(image, prompt)
    baseline_pli = gradcam_to_pli(baseline_cam) if PLI_AVAILABLE else baseline_cam
    
    for noise_level in noise_levels:
        noise_results = {
            "saliency_maps": [],
            "pli_maps": [],
            "stability": None,
            "readability": None
        }
        
        for run in range(num_runs):
            noisy_image = add_noise_to_image(image, noise_level)
            cam, _ = gradcam_instance.generate_cam(noisy_image, prompt)
            pli = gradcam_to_pli(cam) if PLI_AVAILABLE else cam
            
            noise_results["saliency_maps"].append(cam)
            noise_results["pli_maps"].append(pli)
        
        # Compute stability
        noise_results["stability"] = compute_stability_metric(noise_results["saliency_maps"])
        
        # Compute readability (average across runs)
        readability_scores = [compute_readability_metric(m) for m in noise_results["saliency_maps"]]
        noise_results["readability"] = {
            "focus": np.mean([r["focus"] for r in readability_scores]),
            "contrast": np.mean([r["contrast"] for r in readability_scores]),
            "smoothness": np.mean([r["smoothness"] for r in readability_scores])
        }
        
        results[f"noise_{noise_level}"] = noise_results
    
    return results


def test_multi_scale(gradcam_instance, image: Image.Image, prompt: str,
                     scales: List[int]) -> Dict[str, Dict]:
    """
    Test saliency maps at different image scales.
    
    Returns:
        Dict with metrics for each scale
    """
    results = {}
    original_size = image.size
    
    for scale in scales:
        # Resize image
        scaled_image = image.resize((scale, scale), Image.Resampling.LANCZOS)
        
        # Generate saliency
        cam, _ = gradcam_instance.generate_cam(scaled_image, prompt)
        pli = gradcam_to_pli(cam) if PLI_AVAILABLE else cam
        
        # Resize back to original for comparison
        cam_resized = cv2.resize(cam, (original_size[0], original_size[1]), interpolation=cv2.INTER_CUBIC)
        pli_resized = cv2.resize(pli, (original_size[0], original_size[1]), interpolation=cv2.INTER_CUBIC)
        
        # Compute metrics
        readability = compute_readability_metric(cam_resized)
        
        results[f"scale_{scale}"] = {
            "saliency_map": cam_resized,
            "pli_map": pli_resized,
            "readability": readability,
            "scale": scale
        }
    
    return results

