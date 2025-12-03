"""
eval_metrics.py

Evaluation utilities for Pixel-level Interpretability (PLI) maps.

Functions included:
- perturb_image_{noise, brightness, rotate}
- compute_ssim_continuity
- compute_mae_continuity
- compute_lipschitz_stability
- token_logit (helper to get token logit/prob)
- faithfulness_masking (top-k vs random masks)
- batch_evaluate_continuity (run a list of perturbations and return SSIMS)

Dependencies:
pip install scikit-image numpy torch torchvision

"""
from typing import Tuple, List, Callable
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import random 
import math

EPS=1e-8

# -----------------------
# Image perturbations
# -----------------------

def perturb_image_noise(img: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    """Add Gaussian noise to image. img expected in [0,1] float32 HxWxC."""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    pert = img + noise
    return np.clip(pert, 0.0, 1.0)

def perturb_image_brightness(img: np.ndarray, factor: float = 0.2) -> np.ndarray:
    """Multiply brightness by (1 +/- factor). factor=0.2 -> random +/-20%."""
    change = 1.0 + np.random.uniform(-factor, factor)
    pert = img * change
    return np.clip(pert, 0.0, 1.0)

def perturb_image_rotate(img: np.ndarray, angle_deg: float = 5.0) -> np.ndarray:
    """Rotate image by a small angle (degrees). Input [0,1], returns same size."""
    pil = Image.fromarray((img * 255).astype(np.uint8))
    pil_rot = pil.rotate(angle_deg, resample=Image.BILINEAR, expand=False)
    rot = np.asarray(pil_rot).astype(np.float32) / 255.0
    return rot

# -----------------------
# Continuity metrics
# -----------------------
def compute_ssim_continuity(saliency_a: np.ndarray, saliency_b: np.ndarray) -> float:
    """
    Compute SSIM between two saliency maps.
    saliency_*: 2D arrays normalized to [0,1]
    returns: SSIM score in [-1,1] (1 = identical)
    """
    a = (saliency_a - saliency_a.min()) / (saliency_a.max() - saliency_a.min() + EPS)
    b = (saliency_b - saliency_b.min()) / (saliency_b.max() - saliency_b.min() + EPS)
    # skimage expects float images; use multichannel=False
    s = ssim(a, b, data_range=1.0)
    return float(s)

def compute_mae_continuity(saliency_a: np.ndarray, saliency_b: np.ndarray) -> float:
    """Mean absolute error between two saliency maps (normalized)."""
    a = (saliency_a - saliency_a.min()) / (saliency_a.max() - saliency_a.min() + EPS)
    b = (saliency_b - saliency_b.min()) / (saliency_b.max() - saliency_b.min() + EPS)
    return float(np.mean(np.abs(a - b)))

# -----------------------
# Lipschitz stability
# -----------------------
def compute_lipschitz_stability(saliency_a: np.ndarray, saliency_b: np.ndarray,
                                image_a: np.ndarray, image_b: np.ndarray) -> float:
    """
    Compute empirical Lipschitz ratio:
      L = ||s_a - s_b||_2 / (||img_a - img_b||_2 + eps)
    Larger values => saliency changes a lot relative to image change.
    """
    sdiff = np.linalg.norm((saliency_a - saliency_b).ravel())
    idiff = np.linalg.norm((image_a - image_b).ravel())
    return float(sdiff / (idiff + EPS))

# -----------------------
# Token logit helper (model-specific)
# -----------------------
def token_logit(model: torch.nn.Module, image_tensor: torch.Tensor, target_token_id: int,
                device: torch.device, decoder_start: int = None) -> torch.Tensor:
    """
    Compute the raw logit (and probability) for target_token_id given image_tensor.
    image_tensor: torch tensor shape [1, C, H, W] on device, requires_grad may be True/False.
    decoder_start: if None, try to infer from model.config (decoder_start_token_id/bos_token_id/eos)
    Returns: tuple (logit, prob) as torch scalars (on CPU)
    """
    # Prepare decoder start
    if decoder_start is None:
        decoder_start = getattr(model.config, "decoder_start_token_id", None)
    if decoder_start is None:
        decoder_start = getattr(model.config, "bos_token_id", None)
    if decoder_start is None:
        decoder_start = getattr(model.config, "eos_token_id", None)
    if decoder_start is None:
        # safe fallback to first generated token id if model has generation history
        decoder_start = 0

    # Run encoder (BLIP uses vision_model)
    if hasattr(model, "vision_model"):
        enc = model.vision_model(image_tensor)
    elif hasattr(model, "get_encoder"):
        enc = model.get_encoder()(image_tensor)
    elif hasattr(model, "encoder"):
        enc = model.encoder(image_tensor)
    else:
        raise RuntimeError("No encoder found on model for token_logit.")

    encoder_hidden = enc.last_hidden_state if hasattr(enc, "last_hidden_state") else enc[0]

    decoder_input_ids = torch.tensor([[int(decoder_start)]], device=device)
    decoder_outputs = model.text_decoder(
        input_ids=decoder_input_ids,
        encoder_hidden_states=encoder_hidden,
    )
    logits = decoder_outputs.logits  # [batch, seq_len=1, vocab_size]
    # get logit for target token
    logit = logits[0, 0, int(target_token_id)]
    prob = F.softmax(logits[0, 0, :], dim=0)[int(target_token_id)]
    return logit.detach().cpu(), prob.detach().cpu()

# -----------------------
# Faithfulness via masking
# -----------------------
def faithfulness_masking(model: torch.nn.Module,
                         processor: Callable,
                         raw_image: Image.Image,
                         target_token_id: int,
                         saliency_map: np.ndarray,
                         mask_fraction: float = 0.05,
                         mode: str = "top",
                         n_random: int = 5,
                         device: torch.device = torch.device("cpu"),
                         baseline: str = "mean") -> dict:
    """
    Test faithfulness by masking pixels and measuring drop in target token probability.

    Parameters:
      model, processor: same objects used in pipeline (HuggingFace)
      raw_image: PIL Image (RGB)
      target_token_id: token ID (int) of the token to inspect
      saliency_map: 2D numpy array normalized [0,1] (shape matches model input spatially)
      mask_fraction: fraction of pixels to mask (e.g., 0.05 => 5% top pixels)
      mode: 'top' (mask top-saliency) or 'random'
      n_random: number random trials to average for 'random' baseline
      device: torch device
      baseline: 'mean' or 'zero' replacement value when masking

    Returns:
      dict with keys:
        prob_orig, prob_masked, delta, prob_random_mean, delta_over_random (ratio)
    """
    # Prepare original tensor and compute original prob
    inputs = processor(raw_image, return_tensors="pt").to(device)
    image_tensor = inputs["pixel_values"].clone().detach().to(device)

    # original prob
    _, prob_orig = token_logit(model, image_tensor, target_token_id, device)

    # compute mask indices
    h, w = saliency_map.shape
    flat = saliency_map.flatten()
    n_pixels = flat.size
    k = max(1, int(mask_fraction * n_pixels))

    # baseline image array
    orig_arr = np.array(raw_image.resize((w, h))).astype(np.float32) / 255.0

    if mode == "top":
        # mask top-k saliency pixels
        idxs = np.argsort(-flat)[:k]
        mask = np.zeros(n_pixels, dtype=bool)
        mask[idxs] = True
        mask = mask.reshape(h, w)
        perturbed = orig_arr.copy()
        if baseline == "mean":
            fill = orig_arr.mean(axis=(0,1))
        else:
            fill = 0.0
        perturbed[mask] = fill
        # compose back to PIL then to tensor via processor
        pert_pil = Image.fromarray((np.clip(perturbed,0,1)*255).astype(np.uint8))
        inputs_masked = processor(pert_pil, return_tensors="pt").to(device)
        logit_masked, prob_masked = token_logit(model, inputs_masked["pixel_values"], target_token_id, device) \
            if False else token_logit(model, inputs_masked["pixel_values"], target_token_id, device)
        # token_logit accepts image tensor; adapt:
        # token_logit expects image_tensor shape [1,C,H,W] (we pass inputs_masked["pixel_values"])
        # but we passed raw model and processor; safe to call:
        # Convert inputs_masked to image tensor variable for token_logit
        image_tensor_masked = inputs_masked["pixel_values"].clone().detach().to(device)
        _, prob_masked = token_logit(model, image_tensor_masked, target_token_id, device)
        prob_masked = float(prob_masked)
    else:
        # random masking: do n_random trials and average
        deltas = []
        prob_randoms = []
        for _ in range(n_random):
            idxs = np.random.choice(n_pixels, size=k, replace=False)
            mask = np.zeros(n_pixels, dtype=bool); mask[idxs] = True; mask = mask.reshape(h, w)
            perturbed = orig_arr.copy()
            if baseline == "mean":
                fill = orig_arr.mean(axis=(0,1))
            else:
                fill = 0.0
            perturbed[mask] = fill
            pert_pil = Image.fromarray((np.clip(perturbed,0,1)*255).astype(np.uint8))
            inputs_masked = processor(pert_pil, return_tensors="pt").to(device)
            image_tensor_masked = inputs_masked["pixel_values"].clone().detach().to(device)
            _, prob_r = token_logit(model, image_tensor_masked, target_token_id, device)
            prob_randoms.append(float(prob_r))
            deltas.append(float(prob_orig) - float(prob_r))
        prob_masked = float(np.mean(prob_randoms))
        # for 'random' mode we produce delta against this random expectation
    delta = float(prob_orig) - float(prob_masked)
    # To compute delta over random baseline: compute random baseline if mode == 'top'
    if mode == 'top':
        # compute random baseline
        prob_randoms = []
        for _ in range(n_random):
            idxs = np.random.choice(n_pixels, size=k, replace=False)
            mask = np.zeros(n_pixels, dtype=bool); mask[idxs] = True; mask = mask.reshape(h, w)
            perturbed = orig_arr.copy()
            if baseline == "mean":
                fill = orig_arr.mean(axis=(0,1))
            else:
                fill = 0.0
            perturbed[mask] = fill
            pert_pil = Image.fromarray((np.clip(perturbed,0,1)*255).astype(np.uint8))
            inputs_masked = processor(pert_pil, return_tensors="pt").to(device)
            image_tensor_masked = inputs_masked["pixel_values"].clone().detach().to(device)
            _, prob_r = token_logit(model, image_tensor_masked, target_token_id, device)
            prob_randoms.append(float(prob_r))
        prob_random_mean = float(np.mean(prob_randoms))
        delta_random = float(prob_orig) - prob_random_mean
    else:
        prob_random_mean = prob_masked
        delta_random = float(np.mean(deltas))

    score = (delta / (delta_random + EPS)) if (delta_random > 0) else float(delta)
    return {
        "prob_orig": float(prob_orig),
        "prob_masked": float(prob_masked),
        "delta": float(delta),
        "prob_random_mean": float(prob_random_mean),
        "delta_random": float(delta_random),
        "ratio_over_random": float(score)
    }

# -----------------------
# Batch evaluation helpers
# -----------------------
def batch_evaluate_continuity(model, processor, raw_image: Image.Image, saliency_map: np.ndarray,
                              perturb_fns: List[Callable[[np.ndarray], np.ndarray]],
                              device: torch.device):
    """
    Apply each image perturbation in perturb_fns to raw_image, recompute saliency map
    (user should provide a function to compute saliency given an image), and compute SSIM/MAE.
    Here we only implement the skeleton: compute perturbed image arrays.

    Returns list of tuples (perturb_name, ssim_score, mae_score)
    """
    results = []
    # NOTE: user MUST supply a function to recompute saliency from raw image (not included here)
    # For simplicity we just return perturbed images so caller can compute saliency and call compute_ssim_continuity.
    perturbed_images = []
    for fn in perturb_fns:
        pert = fn(np.array(raw_image).astype(np.float32) / 255.0)
        perturbed_images.append(pert)
    return perturbed_images  # caller will compute saliency maps for these and then compute SSIMs externally

