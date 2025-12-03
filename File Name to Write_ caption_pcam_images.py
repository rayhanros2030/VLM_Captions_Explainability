import os
import csv
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)


image_dir = "data/pcam/sample_images"
output_csv = "results/captions/captions.csv"


image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
image_files.sort()


with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image", "caption"])  # header

    for i, img_name in enumerate(image_files):
        img_path = os.path.join(image_dir, img_name)
        raw_image = Image.open(img_path).convert("RGB")

        inputs = processor(raw_image, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_length=30)
        caption = processor.decode(out[0], skip_special_tokens=True)

        writer.writerow([img_name, caption])
        print(f"[{i+1}/{len(image_files)}] {img_name} → {caption}")

print(f"\n✅ Captions saved to {output_csv}")
