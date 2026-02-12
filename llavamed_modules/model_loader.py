"""
Model loading utilities for LLaVA-Med integration.
Handles model and tokenizer loading, and custom config registration.
"""

import os
import sys
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


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

    # Load model without device_map (to avoid requiring accelerate)
    # Explicitly set device_map=None to prevent transformers from auto-detecting it
    # Load on CPU first, then move to device manually
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=None,  # Explicitly disable device_map
    )
    
    # Move model to device manually
    model = model.to(device)
    base_model = model.get_model()

    # Explicitly load and move vision tower
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

