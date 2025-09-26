# backend/python/styling_api/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import requests
from io import BytesIO
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

app = FastAPI()

class StyReq(BaseModel):
    uid: str | None = None
    imageUrl: str
    occasion: str | None = None
    prefs: dict | None = None

def dominant_colors_from_url(url, n=4):
    r = requests.get(url, timeout=10)
    img = Image.open(BytesIO(r.content)).convert('RGB')
    img = img.resize((150,150))
    arr = np.array(img).reshape(-1,3).astype(float)
    km = KMeans(n_clusters=n, random_state=0).fit(arr)
    centers = km.cluster_centers_.astype(int).tolist()
    return centers

@app.post("/recommend")
def recommend(req: StyReq):
    try:
        palette = dominant_colors_from_url(req.imageUrl, n=4)
    except Exception as e:
        palette = []
    # simple mock picks
    picks = [
        {"title":"Casual Shirt", "imageUrl":"https://via.placeholder.com/200/0077CC/ffffff?text=Shirt","price":29.99,"brand":"MockBrand","buyUrl":"https://example.com/p/1"},
        {"title":"Slim Jeans", "imageUrl":"https://via.placeholder.com/200/333333/ffffff?text=Jeans","price":49.99,"brand":"JeansCo","buyUrl":"https://example.com/p/2"}
    ]
    return {"picks": picks, "palette": palette}
