# backend/scripts/generate_metadata.py
import csv, os, random

OUT = os.path.join(os.path.dirname(__file__), "..", "dataset", "metadata.csv")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

num_images = 500
categories = ["Dress", "Shirt", "Pants", "Jacket", "Skirt", "Shorts"]
occasions = ["Casual", "Party", "Formal", "Sporty"]
body_types = ["Athletic", "Hourglass", "Pear", "Inverted Triangle"]

with open(OUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["item_id", "name", "category", "occasion", "body_type", "image_path"])
    for i in range(1, num_images + 1):
        category = random.choice(categories)
        occasion = random.choice(occasions)
        body_type = random.choice(body_types)
        name = f"{occasion} {category} #{i}"
        image_name = f"{str(i).zfill(3)}.jpg"   # matches your serial-only filenames
        image_path = f"images/{image_name}"
        writer.writerow([i, name, category, occasion, body_type, image_path])

print("metadata.csv generated at:", OUT)
