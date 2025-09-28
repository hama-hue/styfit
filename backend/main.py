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

# Enable CORS (frontend can connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8001",
        "http://localhost:8001",
        "http://localhost:5500"
    ],  # ⚠️ Update with your frontend’s actual URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routers
from routers.style_routes import router as style_router
from routers.fitness_routes import router as fitness_router

app.include_router(style_router, prefix="/style", tags=["style"])
app.include_router(fitness_router, prefix="/fitness", tags=["fitness"])

# Load ML models at startup
from utils import load_models

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting backend... preparing models")
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    try:
        load_models(models_dir=models_dir)
        logger.info("✅ Models downloaded and loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "AI Lifestyle Assistant",
        "message": "Backend running with models ready"
    }
