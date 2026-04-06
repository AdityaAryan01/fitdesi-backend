# auth.py  — drop this in your fitdesi-backend/ folder
import os
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# ── Initialise Firebase Admin SDK once at startup ───────────────
# Looks for the service account JSON you downloaded from Firebase Console
_SERVICE_ACCOUNT_PATH = os.path.join(os.path.dirname(__file__), "firebase-service-account.json")

if not firebase_admin._apps:
    cred = credentials.Certificate(_SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred)

# ── Bearer token extractor ───────────────────────────────────────
_bearer = HTTPBearer(auto_error=False)

def verify_firebase_token(
    credentials: HTTPAuthorizationCredentials = Security(_bearer)
) -> dict:
    """
    FastAPI dependency. Use it on any route you want to protect:

        @app.get("/api/secret")
        def secret(token_data: dict = Depends(verify_firebase_token)):
            uid = token_data["uid"]

    Returns the decoded token dict (contains uid, email, etc.)
    Raises 401 if token is missing or invalid.
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        decoded = firebase_auth.verify_id_token(credentials.credentials)
        return decoded
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")