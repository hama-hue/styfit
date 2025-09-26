# backend/scripts/test_model.py
import numpy as np, pickle
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "models"
FEATURES_NPY = MODEL_DIR / "features.npy"
META_PKL = MODEL_DIR / "metadata.pkl"
KNN_PKL = MODEL_DIR / "knn_model.pkl"
IMG_DIR = BASE / "dataset" / "images"

# load
features = np.load(FEATURES_NPY)
meta = pickle.load(open(META_PKL, "rb"))
nn = pickle.load(open(KNN_PKL, "rb"))

# load resnet for single image test
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = ResNet50_Weights.IMAGENET1K_V1
model = resnet50(weights=weights)
model.fc = torch.nn.Identity(); model = model.to(device).eval()
transform = weights.transforms()

# pick a test image from metadata (first one)
test_path = str((IMG_DIR / Path(meta.iloc[0]['image_path']).name).resolve())
img = Image.open(test_path).convert("RGB")
x = transform(img).unsqueeze(0).to(device)
with torch.no_grad():
    feat = model(x).cpu().numpy()

dist, inds = nn.kneighbors(feat)
print("Query:", meta.iloc[0].to_dict())
print("Matches:")
for i, idx in enumerate(inds[0]):
    print(i, "->", meta.iloc[idx].to_dict())
