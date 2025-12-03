from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


device = "mps" if torch.backends.mps.is_available() else "cpu"


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)


img_path = "data/pcam/sample_images/pcam_000000.png"
raw_image = Image.open(img_path).convert("RGB")


inputs = processor(raw_image, return_tensors="pt").to(device)


generated_ids = model.generate(**inputs, max_length=30)
caption = processor.decode(generated_ids[0], skip_special_tokens=True)

print("Generated caption:", caption)

