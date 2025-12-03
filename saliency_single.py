# saliency_loop.py
import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from transformers import BlipProcessor, BlipForConditionalGeneration

# ---- User / path settings ----
img_folder = "data/pcam/sample_images/"  # folder with images
out_dir = "results/saliency_maps"        # folder to save saliency maps
os.makedirs(out_dir, exist_ok=True)

# ---- Device ----
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ---- Load model & processor ----
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
model.eval()

# Disable gradients for model parameters (we only want grads w.r.t. image pixels)
param_requires = [p.requires_grad for p in model.parameters()]
for p in model.parameters():  
    p.requires_grad = False

# ---- Loop through all images ----
for img_file in os.listdir(img_folder):
    if not img_file.endswith(".png"):
        continue  # skip non-PNG files

    image_path = os.path.join(img_folder, img_file)
    raw_image = Image.open(image_path).convert("RGB")

    # Prepare inputs
    inputs = processor(raw_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate caption
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_length=30)

    caption = processor.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Image: {img_file}")
    print(f"Generated caption: {caption}")
    print(f"Generated token ids: {generated_ids}")

    # Pick first "real" token id for saliency
    gen_list = generated_ids[0].tolist()
    decoder_start = getattr(model.config, "decoder_start_token_id", None)
    target_token_id = None
    for tid in gen_list:
        if decoder_start is None or tid != decoder_start:
            target_token_id = int(tid)
            break
    if target_token_id is None:
        target_token_id = int(gen_list[-1])
    print("Target token id chosen:", target_token_id)

    # Prepare image tensor for gradient
    image_tensor = inputs["pixel_values"].clone().detach().to(device)
    image_tensor.requires_grad_(True)

    # Get encoder outputs
    encoder_outputs = None
    try:
        if hasattr(model, "vision_model"):
            encoder_outputs = model.vision_model(image_tensor)
        elif hasattr(model, "get_encoder"):
            encoder_outputs = model.get_encoder()(image_tensor)
        elif hasattr(model, "encoder"):
            encoder_outputs = model.encoder(image_tensor)
        else:
            raise AttributeError("No image encoder found in model.")
    except Exception as e:
        print("ERROR: Could not run encoder:", e)
        continue

    # Extract hidden states
    if hasattr(encoder_outputs, "last_hidden_state"):
        encoder_hidden = encoder_outputs.last_hidden_state
    else:
        encoder_hidden = encoder_outputs[0]

    # Decoder forward for first token
    decoder_input_ids = torch.tensor([[target_token_id]], device=device)
    decoder_outputs = model.text_decoder(
        input_ids=decoder_input_ids,
        encoder_hidden_states=encoder_hidden,
    )
    logits = decoder_outputs.logits
    score = logits[0, 0, target_token_id]

    # Backprop to get gradients w.r.t. image pixels
    if image_tensor.grad is not None:
        image_tensor.grad.zero_()
    model.zero_grad(set_to_none=True)
    score.backward()

    # Build saliency map
    grad = image_tensor.grad.detach()
    saliency = grad.abs().squeeze(0).mean(dim=0)
    saliency = saliency.cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-10)

    # Save saliency heatmap
    save_heatmap = os.path.join(out_dir, img_file.replace(".png", "_saliency.png"))
    plt.imsave(save_heatmap, saliency, cmap="hot")
    print("Saved saliency heatmap to:", save_heatmap)

    # Save overlay on original image
    orig = np.array(raw_image.resize(saliency.shape[::-1])) / 255.0
    overlay = np.clip(orig * 0.5 + np.stack([saliency]*3, axis=2) * 0.5, 0, 1)
    save_overlay = os.path.join(out_dir, img_file.replace(".png", "_overlay.png"))
    plt.imsave(save_overlay, overlay)
    print("Saved overlay to:", save_overlay)
    print("-" * 40)

# Restore model parameter gradients just in case
for p, val in zip(model.parameters(), param_requires):
    p.requires_grad = val

