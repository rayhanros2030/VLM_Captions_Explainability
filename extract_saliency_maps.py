import os
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import matplotlib.pyplot as plt
import numpy as np

device = "mps" if torch.backends.mps.is_available() else "cpu"

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
model.eval()

image_dir = "data/pcam/sample_images"
output_dir = "results/saliency_maps"
os.makedirs(output_dir, exist_ok=True)

image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

for i, img_name in enumerate(image_files):
    img_path = os.path.join(image_dir, img_name)
    raw_image = Image.open(img_path).convert("RGB")

    inputs = processor(raw_image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=30)

    image_tensor = inputs['pixel_values'].clone().detach().requires_grad_(True)

    outputs = model(pixel_values=image_tensor)
    logits = outputs.logits
    target_token_id = out[0, 0]
    score = logits[0, 0, target_token_id]
    score.backward()

    grad = image_tensor.grad[0].permute(1, 2, 0).cpu().numpy()
    grad = np.mean(np.abs(grad), axis=2)
    grad -= grad.min()
    grad /= (grad.max() + 1e-8)

    plt.imshow(raw_image)
    plt.imshow(grad, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f"saliency_{img_name}"))
    plt.close()

    print(f"[{i+1}/{len(image_files)}] Saved saliency map for {img_name}")

