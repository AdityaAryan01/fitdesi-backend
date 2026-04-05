import os
from fastapi import Request, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import firebase_admin
from firebase_admin import credentials, auth

# Initialize Firebase Admin App
# It expects the credentials file to be available
cred_path = os.path.join(os.path.dirname(__file__), "firebase-credentials.json")
try:
    cred = credentials.Certificate(cred_path)
    # Check if app is already initialized to avoid ValueError if this module is reloaded
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
except Exception as e:
    print(f"Failed to initialize Firebase Admin: {e}")

security = HTTPBearer()

def get_current_user_uid(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    token = credentials.credentials
    try:
        # Verify the ID token using the Firebase Admin SDK
        decoded_token = auth.verify_id_token(token)
        uid = decoded_token.get("uid")
        if not uid:
            raise HTTPException(
                status_code=401,
                detail="Token does not contain a valid UID",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return uid
    except Exception as e:
        print(f"Firebase token verification failed: {e}")
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
