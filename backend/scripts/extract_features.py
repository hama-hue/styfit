# backend/scripts/extract_features.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import pickle
from tqdm import tqdm

BASE = Path(__file__).resolve().parent.parent
IMG_DIR = BASE / "dataset" / "images"
META_CSV = BASE / "dataset" / "metadata.csv"
OUT_DIR = BASE / "models"
FEATURES_NPY = OUT_DIR / "features.npy"
META_PKL = OUT_DIR / "metadata.pkl"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load metadata (use corrected metadata if exists)
if (BASE / "dataset" / "metadata_corrected.csv").exists():
    meta_path = BASE / "dataset" / "metadata_corrected.csv"
else:
    meta_path = META_CSV
df = pd.read_csv(meta_path)

# Build absolute image paths list and filter missing files
paths = []
rows = []
for idx, row in df.iterrows():
    rel = str(row['image_path'])
    img_path = (IMG_DIR / Path(rel).name).resolve()
    if img_path.exists():
        paths.append(str(img_path))
        rows.append(row.to_dict())
    else:
        print("Missing image, skipping:", img_path)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = ResNet50_Weights.IMAGENET1K_V1
model = resnet50(weights=weights)
model.fc = torch.nn.Identity()
model = model.to(device).eval()

# Use the transforms provided by the weights
transform = weights.transforms()


# batch extraction
batch_size = 16
feats = []
for i in tqdm(range(0, len(paths), batch_size), desc="Extracting"):
    batch_paths = paths[i:i+batch_size]
    imgs = []
    for p in batch_paths:
        img = Image.open(p).convert('RGB')
        imgs.append(transform(img))
    x = torch.stack(imgs).to(device)
    with torch.no_grad():
        out = model(x).cpu().numpy()
    feats.append(out)

features = np.vstack(feats)
print("features shape:", features.shape)

# Save
np.save(FEATURES_NPY, features)
with open(META_PKL, "wb") as f:
    pickle.dump(pd.DataFrame(rows).reset_index(drop=True), f)

print("Saved:", FEATURES_NPY, META_PKL)
