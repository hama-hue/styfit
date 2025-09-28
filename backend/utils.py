# backend/utils.py
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
import gdown

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Force Torch cache to local folder (fixes some permission errors)
os.environ["TORCH_HOME"] = os.path.join(os.path.dirname(__file__), "torch_cache")

MODELS = {
    "feature_extractor": None,
    "transform": None,
    "features": None,
    "metadata": None,
    "knn": None
}

def _gdrive_id_to_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?id={file_id}"

def download_from_gdrive(models_dir="models"):
    """
    Reads file IDs from env vars:
      GDRIVE_FEATURES_ID, GDRIVE_METADATA_ID, GDRIVE_KNN_ID
    and downloads them to backend/models/ if missing.
    """
    os.makedirs(models_dir, exist_ok=True)

    files = {
        "features.npy": os.getenv("GDRIVE_FEATURES_ID", "").strip(),
        "metadata.csv": os.getenv("GDRIVE_METADATA_ID", "").strip(),
        "knn_model.pkl": os.getenv("GDRIVE_KNN_ID", "").strip()
    }

    for fname, file_id in files.items():
        if not file_id:
            logger.warning(f"No env var set for {fname} (expected GDRIVE_*_ID). Skipping download.")
            continue
        dest = os.path.join(models_dir, fname)
        if os.path.exists(dest):
            logger.info(f"✅ {fname} already exists at {dest}")
            continue
        url = _gdrive_id_to_url(file_id)
        logger.info(f"⬇️ Downloading {fname} from Google Drive: {url}")
        gdown.download(url, dest, quiet=False)
        logger.info(f"Downloaded {fname} -> {dest}")

def load_models(models_dir="models"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading ResNet50 on {device}")

    # load resnet50 backbone
    resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    resnet.fc = torch.nn.Identity()
    resnet = resnet.to(device)
    resnet.eval()

    MODELS["feature_extractor"] = (resnet, device)
    MODELS["transform"] = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Ensure files are present (download if needed)
    download_from_gdrive(models_dir=models_dir)

    # Load features.npy if present
    features_path = os.path.join(models_dir, "features.npy")
    if os.path.exists(features_path):
        MODELS["features"] = np.load(features_path)
        logger.info("✅ features.npy loaded")
    else:
        logger.warning("⚠️ features.npy not found")

    # Load metadata.csv if present
    metadata_path = os.path.join(models_dir, "metadata.csv")
    if os.path.exists(metadata_path):
        MODELS["metadata"] = pd.read_csv(metadata_path)
        logger.info("✅ metadata.csv loaded")
    else:
        logger.warning("⚠️ metadata.csv not found")

    # Load KNN model if present
    knn_path = os.path.join(models_dir, "knn_model.pkl")
    if os.path.exists(knn_path):
        with open(knn_path, "rb") as f:
            MODELS["knn"] = pickle.load(f)
        logger.info("✅ knn_model.pkl loaded")
    else:
        # If knn not provided but features exist, fit one (optional)
        if MODELS["features"] is not None:
            logger.info("No knn_model.pkl found — fitting NearestNeighbors from features.npy")
            MODELS["knn"] = NearestNeighbors(n_neighbors=5, metric="euclidean")
            MODELS["knn"].fit(MODELS["features"])
            logger.info("✅ KNN fitted from features.npy")
        else:
            logger.warning("⚠️ knn_model.pkl not found and features not available — recommendations may not work")

def extract_features_from_bytes(image_bytes):
    if MODELS["feature_extractor"] is None:
        raise RuntimeError("Models not loaded")

    model, device = MODELS["feature_extractor"]
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = MODELS["transform"](img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(x).cpu().numpy()
    return feat
