# backend/scripts/eval_precision_at_k.py
import numpy as np, pickle
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "models"
FEATURES_NPY = MODEL_DIR / "features.npy"
META_PKL = MODEL_DIR / "metadata.pkl"
KNN_PKL = MODEL_DIR / "knn_model.pkl"

features = np.load(FEATURES_NPY)
meta = pickle.load(open(META_PKL,"rb"))
knn = pickle.load(open(KNN_PKL,"rb"))

K = 5
dists, inds = knn.kneighbors(features, n_neighbors=K+1)  # +1 possibly self
precisions = []
for i, neighbors in enumerate(inds):
    # skip self if present:
    nb = [n for n in neighbors if n != i][:K]
    qcat = meta.iloc[i]['category']
    score = sum(1 for n in nb if meta.iloc[n]['category'] == qcat) / K
    precisions.append(score)
print("mean precision@%d = %.3f" % (K, np.mean(precisions)))
