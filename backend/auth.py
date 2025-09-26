import os
import firebase_admin
from firebase_admin import auth, credentials
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

logger = logging.getLogger(__name__)
security = HTTPBearer()

# Lazy Firebase initialization
def init_firebase():
    sa_path = os.getenv("FIREBASE_SERVICE_ACCOUNT")
    if not sa_path:
        raise RuntimeError("FIREBASE_SERVICE_ACCOUNT not set in .env")
    cred = credentials.Certificate(sa_path)
    try:
        firebase_admin.get_app()
    except ValueError:
        firebase_admin.initialize_app(cred)
        logger.info("✅ Firebase initialized.")

# Dependency
def verify_firebase_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        init_firebase()
        decoded = auth.verify_id_token(token)
        return decoded
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired Firebase token")
