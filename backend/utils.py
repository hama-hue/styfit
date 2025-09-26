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
from sklearn.cluster import KMeans
import pickle

logger = logging.getLogger(__name__)

# Force Torch cache to local folder (fixes Windows permission error)
os.environ["TORCH_HOME"] = os.path.join(os.path.dirname(__file__), "torch_cache")

MODELS = {
    "feature_extractor": None,
    "transform": None,
    "features": None,
    "metadata": None,
    "knn": None,
    "kmeans": None
}

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

    # Load features
    features_path = os.path.join(models_dir, "features.npy")
    if os.path.exists(features_path):
        MODELS["features"] = np.load(features_path)
        MODELS["knn"] = NearestNeighbors(n_neighbors=5, metric="euclidean")
        MODELS["knn"].fit(MODELS["features"])
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

    # Load or train KMeans
    kmeans_pkl = os.path.join(models_dir, "kmeans.pkl")
    if os.path.exists(kmeans_pkl):
        with open(kmeans_pkl, "rb") as f:
            MODELS["kmeans"] = pickle.load(f)
        logger.info("✅ kmeans.pkl loaded")
    elif MODELS["features"] is not None:
        MODELS["kmeans"] = KMeans(n_clusters=20, random_state=42).fit(MODELS["features"])
        logger.info("✅ KMeans fitted")

def extract_features_from_bytes(image_bytes):
    if MODELS["feature_extractor"] is None:
        raise RuntimeError("Models not loaded")

    model, device = MODELS["feature_extractor"]
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = MODELS["transform"](img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(x).cpu().numpy()
    return feat
