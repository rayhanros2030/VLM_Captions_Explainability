from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import matplotlib.pyplot as plt

# Device setup
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load model + processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

# Pick one image
img_path = "data/pcam/sample_images/pcam_000000.png"
raw_image = Image.open(img_path).convert("RGB")

# Preprocess
inputs = processor(raw_image, return_tensors="pt").to(device)
image_tensor = inputs["pixel_values"].clone().detach().to(device)
image_tensor.requires_grad_(True)

# Generate a caption
generated_ids = model.generate(**inputs, max_length=30)
caption = processor.decode(generated_ids[0], skip_special_tokens=True)
print("Generated caption:", caption)

# Run encoder
encoder_outputs = model.get_encoder()(image_tensor)

# Decoder: predict the first token
decoder_input_ids = torch.tensor(
    [[model.config.decoder_start_token_id]], device=device
)
decoder_outputs = model.text_decoder(
    input_ids=decoder_input_ids,
    encoder_hidden_states=encoder_outputs.last_hidden_state,
)

# Pick target token
target_token_id = generated_ids[0, 0]
score = decoder_outputs.logits[0, 0, target_token_id]

# Backprop to get gradients
score.backward()

# Compute saliency map
saliency = image_tensor.grad.abs().squeeze().mean(dim=0)
saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-10)

# Show saliency map
plt.imshow(saliency.cpu(), cmap="hot")
plt.axis("off")
plt.show()

import numpy as np
import skfuzzy as fuzz

def gradcam_to_pli(saliency_map):
    """
    Convert a GradCAM-style saliency map into a Pixel-Level Interpretability (PLI) map
    using fuzzy logic membership functions (based on Ennab–Mchieck hybrid method).

    Parameters:
        saliency_map (np.ndarray): Normalized saliency map (values in [0,1]).

    Returns:
        pli_map (np.ndarray): Pixel-level interpretability (PLI) map.
    """

    # Ensure saliency map is valid
    saliency_map = np.clip(saliency_map, 0, 1)

    # Universe of discourse for fuzzy membership
    x = np.linspace(0, 1, 100)

    # Define fuzzy membership functions for each linguistic label
    not_involved = fuzz.trimf(x, [0.0, 0.0, 0.1])
    very_low     = fuzz.trimf(x, [0.1, 0.2, 0.3])
    moderate     = fuzz.trimf(x, [0.3, 0.4, 0.5])
    high         = fuzz.trimf(x, [0.5, 0.6, 0.7])
    complete     = fuzz.trimf(x, [0.7, 0.85, 1.0])

    # Flatten saliency map for easier fuzzy interpolation
    flat_saliency = saliency_map.flatten()

    # Calculate fuzzy membership degrees
    mu_not_involved = fuzz.interp_membership(x, not_involved, flat_saliency)
    mu_very_low     = fuzz.interp_membership(x, very_low, flat_saliency)
    mu_moderate     = fuzz.interp_membership(x, moderate, flat_saliency)
    mu_high         = fuzz.interp_membership(x, high, flat_saliency)
    mu_complete     = fuzz.interp_membership(x, complete, flat_saliency)

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

    # Normalize the PLI map to [0,1] for visualization
    pli_flat = (pli_flat - pli_flat.min()) / (pli_flat.max() - pli_flat.min() + 1e-10)
    pli_map = pli_flat.reshape(saliency_map.shape)

    return pli_map
# Convert to numpy
saliency_np = saliency.detach().cpu().numpy()

# Generate PLI map using fuzzy logic
pli_map = gradcam_to_pli(saliency_np)



# Visualize PLI
plt.imshow(pli_map, cmap="Reds")
plt.title("Pixel-Level Interpretability (PLI) Map")
plt.axis("off")
plt.show()

if __name__ == "__main__":
    # Convert to numpy
    saliency_np = saliency.detach().cpu().numpy()

    # Generate PLI map using fuzzy logic
    pli_map = gradcam_to_pli(saliency_np)

    # Visualize both maps
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(saliency_np, cmap="hot")
    plt.title("GradCAM Saliency Map")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(pli_map, cmap="Reds")
    plt.title("Pixel-Level Interpretability (PLI) Map")
    plt.axis("off")
    plt.tight_layout()
    plt.show()



