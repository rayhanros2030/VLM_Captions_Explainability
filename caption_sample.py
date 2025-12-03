# caption_sample.py
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# ---- Device ----
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# ---- Load model & processor ----
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
model.eval()  # put model in evaluation mode

# ---- Load your image ----
image_path = "sample.png"  # make sure this matches your file
image = Image.open(image_path).convert("RGB")

# ---- Prepare image and generate caption ----
inputs = processor(image, return_tensors="pt").to(device)

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_length=30)
caption = processor.decode(generated_ids[0], skip_special_tokens=True)

print("Generated caption:", caption)

