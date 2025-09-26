# backend/scripts/train_classifier.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# ---------------- Load features and labels ----------------
FEATURES_PATH = "backend/models/features.npy"
META_PATH = "backend/models/metadata.pkl"
MODEL_OUT_PATH = "backend/models/classifier.pth"

print("Loading features and metadata...")
features = np.load(FEATURES_PATH)
with open(META_PATH, "rb") as f:
    df = pickle.load(f)

# Make sure your metadata has a 'category' column
if 'category' not in df.columns:
    raise ValueError("metadata.pkl must have a 'category' column for labels")

labels = df['category'].values

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)

# Convert to torch tensors
X = torch.tensor(features, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

print(f"Feature shape: {X.shape}, Labels shape: {y.shape}, Num classes: {len(le.classes_)}")

# ---------------- Build classifier ----------------
model = nn.Sequential(
    nn.Linear(X.shape[1], 128),
    nn.ReLU(),
    nn.Linear(128, len(le.classes_))
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------------- Training loop ----------------
epochs = 30
print("Training classifier...")
for epoch in range(epochs):
    optimizer.zero_grad()
    out = model(X)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# ---------------- Save trained model ----------------
torch.save({
    "model_state_dict": model.state_dict(),
    "label_encoder": le
}, MODEL_OUT_PATH)

print(f"Training complete! Model saved to {MODEL_OUT_PATH}")
