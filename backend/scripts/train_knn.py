# backend/scripts/train_knn.py
import numpy as np
from pathlib import Path
import pickle
from sklearn.neighbors import NearestNeighbors

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "models"
FEATURES_NPY = MODEL_DIR / "features.npy"
KNN_PKL = MODEL_DIR / "knn_model.pkl"

if not FEATURES_NPY.exists():
    raise SystemExit("Run extract_features.py first")

features = np.load(FEATURES_NPY)
n_neighbors = min(6, len(features))  # 1 query + 5 results typical
nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', n_jobs=-1)
nn.fit(features)

with open(KNN_PKL, "wb") as f:
    pickle.dump(nn, f)

print("Saved KNN model to", KNN_PKL)
