from PIL import Image
import os

img_folder = 'backend/dataset/images/'

for img_name in os.listdir(img_folder):
    if img_name.endswith('.jpg'):
        img_path = os.path.join(img_folder, img_name)
        with Image.open(img_path) as img:
            img = img.resize((224, 224))  # Standard size for ResNet50
            img.save(img_path)
