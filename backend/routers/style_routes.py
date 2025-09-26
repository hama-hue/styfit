from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import JSONResponse
from auth import verify_firebase_token
from utils import MODELS, extract_features_from_bytes
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/recommend")
async def recommend_style(file: UploadFile = File(...), user=Depends(verify_firebase_token)):
    if MODELS["features"] is None or MODELS["metadata"] is None or MODELS["knn"] is None:
        raise HTTPException(status_code=500, detail="Models or data not loaded")

    image_bytes = await file.read()
    query_feat = extract_features_from_bytes(image_bytes)

    dists, idxs = MODELS["knn"].kneighbors(query_feat)

    cols = ["id", "category", "occasion", "body_type", "image_path"]
    df = MODELS["metadata"]
    results = df.iloc[idxs[0]][cols].to_dict(orient="records")

    return JSONResponse(content={"recommendations": results})
