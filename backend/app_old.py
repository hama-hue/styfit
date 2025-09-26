# backend/app.py
from fastapi import FastAPI,Request, HTTPException, UploadFile, File, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import firebase_admin
from firebase_admin import credentials, auth, firestore
import cloudinary
import cloudinary.uploader
import os
import requests
from PIL import Image
from typing import Optional
import pickle
import numpy as np

# ML imports
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import os.path as osp

# --- FIX: Explicitly set the TORCH_HOME environment variable to a writable directory
os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".torch_cache")

# ---------------- CONFIG ----------------
# Use environment variables for sensitive data in production
SERVICE_ACCOUNT = os.getenv("SERVICE_ACCOUNT_JSON", "serviceAccountKey.json")
CLOUDINARY_NAME = os.getenv("CLOUDINARY_NAME", "dmnizdiey")
CLOUDINARY_KEY = os.getenv("CLOUDINARY_KEY", "518579141378864")
CLOUDINARY_SECRET = os.getenv("CLOUDINARY_SECRET", "7T2RTJAdURwe4U0Yqi9uobWMh1c")
FRONTEND_ORIGINS = ["*"] # For development purposes, allow all origins

# ML Model Paths
BASE_DIR = osp.dirname(osp.abspath(__file__))
MODELS_DIR = osp.join(BASE_DIR, "models")
DATASET_DIR = osp.join(BASE_DIR, "dataset")

CLASSIFIER_PTH = osp.join(MODELS_DIR, "classifier.pth")
KNN_MODEL_PKL = osp.join(MODELS_DIR, "knn_model.pkl")
METADATA_PKL = osp.join(MODELS_DIR, "metadata.pkl")
DATASET_CSV = osp.join(DATASET_DIR, "metadata.csv")

# ---------------- INITIALIZATION ----------------
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Firebase Admin SDK

if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT)
    firebase_admin.initialize_app(cred)
db = firestore.client()


async def verify_token(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        print("Missing Authorization header:", auth_header)
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    try:
        token = auth_header.split(" ")[1]
        print("Received token (first 20 chars):", token[:20], "...")
        decoded_token = auth.verify_id_token(token)
        print("Decoded token:", decoded_token)
        return decoded_token
    except Exception as e:
        print("Token verification error:", e)  # <-- real reason
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    


# Cloudinary
cloudinary.config(
    cloud_name=CLOUDINARY_NAME,
    api_key=CLOUDINARY_KEY,
    api_secret=CLOUDINARY_SECRET,
)

# ---------------- ML MODEL LOADING ----------------
# It's best practice to load models once on startup
try:
    # Load feature extractor (ResNet50 pre-trained on ImageNet)
    weights = ResNet50_Weights.IMAGENET1K_V1
    feature_model = resnet50(weights=weights)
    feature_model.fc = torch.nn.Identity()  # Remove the final classification layer
    feature_model.eval()
    
    # Load classifier head
    # The dimensions (2048, 8) must match the number of classes
    classifier_model = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(),
        nn.Linear(128, 8)
    )

    # Use torch.load with `weights_only=False` to bypass security check
    loaded_state = torch.load(CLASSIFIER_PTH, map_location='cpu', weights_only=False)
    
    # Load the state dict correctly from the nested dictionary
    if 'model_state_dict' in loaded_state:
        classifier_model.load_state_dict(loaded_state['model_state_dict'])
    else:
        classifier_model.load_state_dict(loaded_state)
    
    classifier_model.eval()

    # Load KNN model, metadata, and label encoder
    with open(KNN_MODEL_PKL, 'rb') as f:
        knn_model = pickle.load(f)

    with open(METADATA_PKL, 'rb') as f:
        metadata_df = pickle.load(f)

    # --- FIX: Dynamically create the LabelEncoder instead of loading a missing file
    # This resolves the FileNotFoundError for label_encoder.pkl
    label_encoder = LabelEncoder()
    # The categories are taken directly from the loaded metadata DataFrame
    label_encoder.fit(metadata_df['category'].unique())
        
    print("ML models and data loaded successfully.")
    
except FileNotFoundError as e:
    print(f"Error loading ML model files: {e}. Please ensure all .pkl and .pth files are in the correct directory.")
    feature_model = None
    classifier_model = None
    knn_model = None
    metadata_df = None
    label_encoder = None
except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")
    feature_model = None
    classifier_model = None
    knn_model = None
    metadata_df = None
    label_encoder = None

# --- Image Transformations (must match training pipeline) ---
image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------- HELPER FUNCTIONS ----------------
def predict_outfit(image_url: str) -> str:
    """
    Predicts the outfit type from an image URL using the pre-trained models.
    """
    if not all([feature_model, classifier_model, label_encoder]):
        return "Unknown"

    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        img = Image.open(response.raw).convert('RGB')
        
        img_tensor = image_transforms(img).unsqueeze(0)
        
        with torch.no_grad():
            features = feature_model(img_tensor)
            outputs = classifier_model(features)
            
            _, predicted_idx = torch.max(outputs, 1)
            predicted_class = predicted_idx.item()

            label = label_encoder.inverse_transform([predicted_class])[0]
            
            return label.capitalize()

    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Unknown"


# --- AUTHENTICATION DEPENDENCY ---
# This is a standard FastAPI pattern for dependency injection
oauth2_scheme = HTTPBearer()

async def get_current_user(token: HTTPAuthorizationCredentials = Depends(oauth2_scheme)):
    try:
        decoded_token = auth.verify_id_token(token.credentials)
        return decoded_token
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")


# ---------------- API ROUTES ----------------
class StyleReq(BaseModel):
    imageUrl: str
    occasion: Optional[str] = None
    bodyType: Optional[str] = None

class MakeupReq(BaseModel):
    skinTone: str

@app.post("/upload")
async def upload_image(file: UploadFile = File(...), user=Depends(verify_token)):
    """
    Handles image upload and saves metadata to Firestore with a content-based ID.
    """
    uid = user.get("uid")
    if not uid:
        raise HTTPException(status_code=401, detail="Authentication failed.")

    try:
        result = cloudinary.uploader.upload(file.file, folder="styfit_uploads")
        url = result.get("secure_url")

        predicted_category = predict_outfit(url)

        unique_id_suffix = os.urandom(8).hex()
        new_doc_id = f"{predicted_category.lower().replace(' ', '_')}-{unique_id_suffix}"

        doc_data = {
            "url": url,
            "filename": file.filename,
            "content_type": file.content_type,
            "owner": uid,
            "category": predicted_category,
            "createdAt": firestore.SERVER_TIMESTAMP
        }

        db.collection("uploads").document(new_doc_id).set(doc_data)

        return {"url": url, "docId": new_doc_id}

    except Exception as e:
        print(f"Error during file upload: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during file upload.")

@app.get("/data/filters")
def get_filters():
    """
    Returns dynamic dropdown options based on the dataset.
    """
    try:
        if metadata_df is not None:
            # FIX: Get unique filters directly from the loaded dataframe
            occasions = metadata_df['occasion'].unique().tolist()
            body_types = metadata_df['body_type'].unique().tolist()
            return {"occasions": occasions, "bodyTypes": body_types}
        else:
            # Fallback to hardcoded values if models failed to load
            return {"occasions": ["Casual", "Formal", "Party", "Sporty", "Everyday"],
                    "bodyTypes": ["Athletic", "Hourglass", "Pear", "Inverted Triangle", "Rectangle"]}
    except Exception as e:
        print(f"Error getting filters: {e}")
        raise HTTPException(status_code=500, detail="Could not load filter data.")


@app.post("/style/recommend")
def recommend_style(req: StyleReq, user=Depends(verify_token)):
    """
    Provides style recommendations based on uploaded image and user preferences.
    Uses ML model to analyze image content and filters to provide curated picks.
    """
    uid = user.get("uid")
    if feature_model is None or knn_model is None or metadata_df is None:
        raise HTTPException(status_code=500, detail="Models or metadata not loaded properly")

    try:
        response = requests.get(req.imageUrl, stream=True)
        img = Image.open(response.raw).convert('RGB')
        img_tensor = image_transforms(img).unsqueeze(0)
        
        with torch.no_grad():
            features = feature_model(img_tensor).numpy()
            
        dists, indices = knn_model.kneighbors(features)
        
        picks = []
        unique_indices = list(dict.fromkeys(indices[0]))
        
        for idx in unique_indices:
            # Skip self-match if present
            if metadata_df.iloc[idx]['image_path'] == req.imageUrl:
                continue
                
            item = metadata_df.iloc[idx]
            
            # Filtering logic
            if req.occasion and req.occasion.lower() != item['occasion'].lower():
                continue
            if req.bodyType and req.bodyType.lower() != item['body_type'].lower():
                continue
            
            picks.append({
                "brand": "StyFit",
                "imageUrl": item['image_path'],
                "palette_match": ["#F2D3A7", "#C89F6A", "#8FBF9F"]
            })
            
            if len(picks) >= 6:
                break
        
        db.collection("recommend_requests").add({
            "uid": uid,
            "imageUrl": req.imageUrl,
            "occasion": req.occasion,
            "bodyType": req.bodyType,
            "resultCount": len(picks),
            "createdAt": firestore.SERVER_TIMESTAMP
        })
        
        return {"picks": picks}

    except Exception as e:
        print(f"Error in recommendation logic: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during style recommendations.")


@app.post("/style/makeup")
def style_makeup(req: MakeupReq, user: dict = Depends(get_current_user)):
    uid = user.get("uid")
    mapping = {
        "fair": [{"name": "Ivory", "type": "foundation"}, {"name": "Rosewood", "type": "lipstick"}],
        "medium": [{"name": "Beige", "type": "foundation"}, {"name": "Terracotta", "type": "blush"}],
        "olive": [{"name": "Tan", "type": "foundation"}, {"name": "Bronze", "type": "highlighter"}],
        "deep": [{"name": "Mahogany", "type": "foundation"}, {"name": "Plum", "type": "lipstick"}],
    }
    shades = mapping.get(req.skinTone.lower(), [])
    db.collection("makeup_requests").add({"uid": uid, "skinTone": req.skinTone, "resultCount": len(shades), "createdAt": firestore.SERVER_TIMESTAMP})
    return {"makeup": shades}


@app.post("/style/palette")
def style_palette(req: MakeupReq, user: dict = Depends(get_current_user)):
    uid = user.get("uid")
    map_season = {
        "fair": ["#EFD9D1", "#F7C6BD", "#C7D6E8", "#9BB7D4", "#F2E2B3"],
        "medium": ["#F2D3A7", "#C89F6A", "#8FBF9F", "#D56B6B", "#FFE5C4"],
        "olive": ["#9B6B3F", "#708238", "#D9C08A", "#6B8E23", "#3A5A40"],
        "deep": ["#3B1F2B", "#7A3B47", "#B86B6B", "#4A2D2A", "#2F1A1A"],
    }
    palette = map_season.get(req.skinTone.lower(), [])
    db.collection("palette_requests").add({"uid": uid, "skinTone": req.skinTone, "resultCount": len(palette), "createdAt": firestore.SERVER_TIMESTAMP})
    return {"palette": palette}
