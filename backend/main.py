# backend/main.py
import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Lifestyle Assistant")

# Enable CORS (frontend can connect) — add your deployed frontend URL here
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8001",
        "http://localhost:8001",
        "http://localhost:5500",
        os.getenv("FRONTEND_ORIGIN", ""),  # set FRONTEND_ORIGIN on Render (optional)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routers
from routers.style_routes import router as style_router
from routers.fitness_routes import router as fitness_router

app.include_router(style_router, prefix="/style", tags=["style"])
app.include_router(fitness_router, prefix="/fitness", tags=["fitness"])

# ✅ Lazy load: don't load heavy ML models at startup
logger.info("🚀 App started (ML models will load lazily on first request).")

@app.get("/")
def root():
    return {"status": "ok", "service": "AI Lifestyle Assistant"}

# Optional: health check endpoint to confirm models are loaded
from utils import MODELS
@app.get("/health")
def health_check():
    if MODELS.get("feature_extractor") is None:
        return {"status": "ok", "models": "not loaded yet"}
    return {"status": "ok", "models": "loaded"}
