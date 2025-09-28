import os
import logging
import io
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from sklearn.neighbors import NearestNeighbors
import pickle
import gdown  # to download from Google Drive

logger = logging.getLogger(__name__)

# Force Torch cache to local folder (fixes Windows permission error)
os.environ["TORCH_HOME"] = os.path.join(os.path.dirname(__file__), "torch_cache")

MODELS = {
    "feature_extractor": None,
    "transform": None,
    "features": None,
    "metadata": None,
    "knn": None
}

# Google Drive file IDs (replace with your actual IDs)
GDRIVE_FILES = {
    "features.npy": "1DTAY1JOrduOfZYKSuepwq2Ho3ntNhQ4z",
    "metadata.csv": "1Mf6GD170Tdlh4GqNelu2RserdWmiweYR",
    "knn_model.pkl": "13q1swYdmF8GNow120mvPKHcqC02ONr-f"
}

def download_from_gdrive(models_dir="models"):
    os.makedirs(models_dir, exist_ok=True)
    for filename, file_id in GDRIVE_FILES.items():
        path = os.path.join(models_dir, filename)
        if not os.path.exists(path):
            logger.info(f"⬇️ Downloading {filename} from Google Drive...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, path, quiet=False)
        else:
            logger.info(f"✅ {filename} already exists")

def load_models(models_dir="models"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading ResNet50 on {device}")

    resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    resnet.fc = torch.nn.Identity()
    resnet = resnet.to(device)
    resnet.eval()

    MODELS["feature_extractor"] = (resnet, device)
    MODELS["transform"] = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Ensure models_dir exists and files are downloaded
    download_from_gdrive(models_dir)

    # Load features
    features_path = os.path.join(models_dir, "features.npy")
    if os.path.exists(features_path):
        MODELS["features"] = np.load(features_path)
        logger.info("✅ features.npy loaded")
    else:
        logger.warning("⚠️ features.npy not found")

    # Load metadata
    metadata_path = os.path.join(models_dir, "metadata.csv")
    if os.path.exists(metadata_path):
        MODELS["metadata"] = pd.read_csv(metadata_path)
        logger.info("✅ metadata.csv loaded")
    else:
        logger.warning("⚠️ metadata.csv not found")

    # Load KNN model
    knn_path = os.path.join(models_dir, "knn_model.pkl")
    if os.path.exists(knn_path):
        with open(knn_path, "rb") as f:
            MODELS["knn"] = pickle.load(f)
        logger.info("✅ knn_model.pkl loaded")
    else:
        logger.warning("⚠️ knn_model.pkl not found")

def extract_features_from_bytes(image_bytes):
    if MODELS["feature_extractor"] is None:
        raise RuntimeError("Models not loaded")

    model, device = MODELS["feature_extractor"]
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = MODELS["transform"](img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(x).cpu().numpy()
    return feat
