from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import csv

# device setup (M1/M2 Macs use "mps")
device = "mps" if torch.backends.mps.is_available() else "cpu"

# load BLIP model + processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# input/output paths
image_dir = "data/pcam/sample_images"
output_csv = "results/captions/pcam_captions.csv"

# make sure output folder exists
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# open CSV file for writing
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "caption"])  # header row

    # loop through images
    for idx, filename in enumerate(sorted(os.listdir(image_dir))):
        if not filename.endswith(".png"):
            continue

        img_path = os.path.join(image_dir, filename)
        raw_image = Image.open(img_path).convert("RGB")

        # process image
        inputs = processor(raw_image, return_tensors="pt").to(device)

        # generate caption
        generated_ids = model.generate(**inputs, max_length=30)
        caption = processor.decode(generated_ids[0], skip_special_tokens=True)

        # write to CSV
        writer.writerow([filename, caption])

        print(f"[{idx+1}] {filename} -> {caption}")

print(f"\n✅ Captions saved to {output_csv}")

