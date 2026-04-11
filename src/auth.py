from datetime import datetime, timedelta
from typing import Optional
import os

from fastapi import Depends, HTTPException, status, WebSocket
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────
# In production: load SECRET_KEY from env, never hardcode it
# Generate a strong key with: openssl rand -hex 32
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-only-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# ── Password hashing ─────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ── OAuth2 scheme (reads Bearer token from Authorization header) ──
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# ── Fake user DB (replace with real DB in production) ─────────
# To generate a hashed password: pwd_context.hash("yourpassword")
USERS_DB = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("admin123"),
        "role": "admin",
    },
    "operator": {
        "username": "operator",
        "hashed_password": pwd_context.hash("operator123"),
        "role": "operator",
    },
}

# ── Pydantic models ───────────────────────────────────────────
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None

class User(BaseModel):
    username: str
    role: str

# ── Helper functions ──────────────────────────────────────────
def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def authenticate_user(username: str, password: str) -> Optional[dict]:
    user = USERS_DB.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# ── Dependencies ──────────────────────────────────────────────
def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Dependency for REST endpoints — reads Bearer token from header."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None:
            raise credentials_exception
        return User(username=username, role=role)
    except JWTError:
        raise credentials_exception

def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Dependency for admin-only endpoints."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user

async def get_ws_user(websocket: WebSocket) -> Optional[User]:
    """Verify JWT from WebSocket query param: ws://host/ws?token=..."""
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008)  # Policy violation
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        role = payload.get("role")
        if not username:
            await websocket.close(code=1008)
            return None
        return User(username=username, role=role)
    except JWTError:
        await websocket.close(code=1008)
        return None