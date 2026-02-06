"""
GradCAM module for LLaVA-Med integration.
Contains the GradCAM class for generating saliency maps and captions.
"""

import re
import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from typing import Tuple, Optional

# Import from other modules
from .config import cfg
from .pli import gradcam_to_pli, PLI_AVAILABLE

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
                    # Apply repetition penalty to reduce repetitive outputs
                    if step > 0 and generated_ids.shape[1] > 1:
                        # Get recent tokens (last 20 tokens for stronger penalty)
                        recent_tokens = generated_ids[0, -20:].tolist()
                        # Count token frequencies
                        from collections import Counter
                        token_counts = Counter(recent_tokens)
                        # Apply stronger penalty to frequently repeated tokens
                        repetition_penalty = 1.5  # Increased from 1.2
                        for token_id, count in token_counts.items():
                            if token_id < next_token_logits.shape[-1] and count > 1:
                                # Stronger penalty for more frequent tokens
                                penalty = repetition_penalty ** count
                                next_token_logits[0, token_id] /= penalty
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
        
        # ✅ NEW: Check for medical notation/abbreviations (like "αMB85(+), PD+, ypN0")
        # These patterns have: letters, numbers, parentheses, plus signs, commas
        stripped = caption.strip()
        
        # Check for patterns like "αMB85(+), PD+, ypN0" - short captions with codes/symbols
        if len(stripped) < 100:  # Short captions are more likely to be codes
            # Count special medical notation characters
            special_chars = sum(1 for c in stripped if c in '()+-=,αβγ')
            upper_nums = sum(1 for c in stripped if c.isupper() or c.isdigit())
            lower_alpha = sum(1 for c in stripped if c.islower())
            
            # If it has parentheses, plus signs, or Greek letters, and is mostly codes
            has_medical_symbols = any(c in stripped for c in '()+-αβγ')
            if has_medical_symbols:
                # If more than 60% is uppercase/numbers/special chars, likely medical notation
                if len(stripped) > 0 and (upper_nums + special_chars) / len(stripped) > 0.6:
                    print(f"      ❌ Detected medical notation/abbreviation pattern: '{stripped}'")
                    return False
                
                # Also check for pattern: letter(s) + number + (symbols) + comma + more codes
                if ',' in stripped and re.search(r'[A-Zα-ω]+\d*[\(\)\+\-]*', stripped):
                    print(f"      ❌ Detected medical notation (code pattern with commas): '{stripped}'")
                    return False
        
        # Check for very short captions that are mostly codes (no descriptive words)
        if len(stripped) < 30:
            words = stripped.split()
            # If it has very few words and they're mostly uppercase/short, likely codes
            if len(words) <= 3:
                all_short_uppercase = all(len(w) <= 5 and (w.isupper() or any(c.isdigit() for c in w)) for w in words)
                if all_short_uppercase and any(c in stripped for c in '()+-,'):
                    print(f"      ❌ Detected medical notation (short code-like caption): '{stripped}'")
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
        # Ensure torch is available in local scope (fixes UnboundLocalError)
        import torch as _torch
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
            # Handle NumPy compatibility issue (NumPy 2.x incompatibility)
            try:
                processed = self.image_processor(image, return_tensors="pt")
                image_tensor = processed["pixel_values"].to(device=device, dtype=self.model_dtype)
            except (ValueError, RuntimeError) as e:
                if "Numpy is not available" in str(e) or "numpy" in str(e).lower():
                    # Fallback: process without return_tensors, then convert manually
                    try:
                        processed = self.image_processor(image, return_tensors=None)
                        # Try to use numpy if available, otherwise use torch directly
                        if isinstance(processed["pixel_values"], list):
                            # Convert list to tensor directly (torch is already imported at top of file)
                            pixel_values = _torch.tensor(processed["pixel_values"], dtype=self.model_dtype)
                        else:
                            pixel_values = np.array(processed["pixel_values"])
                            pixel_values = _torch.from_numpy(pixel_values)
                        image_tensor = pixel_values.to(device=device, dtype=self.model_dtype)
                    except Exception as e2:
                        raise RuntimeError(
                            f"NumPy compatibility error. Please downgrade NumPy: "
                            f"pip install 'numpy<2' --force-reinstall --no-cache-dir\n"
                            f"Original error: {e}\nFallback error: {e2}"
                        ) from e
                else:
                    raise

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
                                input_ids = _torch.tensor([input_ids_list], device=device)
                                break

            if not isinstance(input_ids, _torch.Tensor):
                input_ids = _torch.tensor([input_ids], device=device)
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
                    input_ids = _torch.cat([
                        input_ids[:, :insert_pos],
                        _torch.tensor([[self.IMAGE_TOKEN_INDEX]], device=device),
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

                    weights = _torch.mean(patch_grads, dim=1)
                    features = _torch.mean(patch_acts, dim=1)
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
                        cam = _torch.pow(cam, gamma)
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
                        if _torch.cuda.is_available():
                            _torch.cuda.empty_cache()

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
                        cam_1d = _torch.norm(acts, dim=1)
                    elif method == "abs_mean":
                        cam_1d = _torch.abs(acts).mean(dim=1)
                    elif method == "max":
                        cam_1d = _torch.abs(acts).max(dim=1)[0]
                    elif method == "variance":
                        cam_1d = _torch.var(acts, dim=1)
                    else:
                        cam_1d = _torch.norm(acts, dim=1)

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
                        cam = _torch.pow(cam, gamma)
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
                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()

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
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
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
