"""
Security utilities for authentication and authorization.
Implements JWT-based authentication with role-based access control.
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from jose import jwt, JWTError
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from pydantic import BaseModel, ValidationError

from .config import settings


# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_PREFIX}/auth/login",
    scopes={
        "read:events": "Read activity events",
        "read:alerts": "Read clinical alerts",
        "manage:alerts": "Acknowledge and resolve alerts",
        "read:patients": "Read patient information",
        "manage:patients": "Create, update, delete patients",
        "read:devices": "Read device information",
        "manage:devices": "Configure devices",
        "admin": "Full administrative access",
    },
)


class TokenPayload(BaseModel):
    """JWT token payload structure."""
    sub: str  # Subject (user ID)
    exp: datetime  # Expiration time
    iat: datetime  # Issued at
    scopes: List[str] = []  # Permission scopes
    type: str = "access"  # Token type: access or refresh


class TokenData(BaseModel):
    """Decoded token data."""
    user_id: str
    scopes: List[str] = []


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password: str) -> str:
    """Hash a password for storage."""
    return pwd_context.hash(password)


def create_access_token(
    subject: str,
    scopes: List[str] = None,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a JWT access token.

    Args:
        subject: User identifier (typically user ID or email)
        scopes: List of permission scopes
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT token string
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    payload = {
        "sub": subject,
        "exp": expire,
        "iat": datetime.utcnow(),
        "scopes": scopes or [],
        "type": "access",
    }

    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_refresh_token(subject: str) -> str:
    """
    Create a JWT refresh token.

    Args:
        subject: User identifier

    Returns:
        Encoded JWT refresh token string
    """
    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    payload = {
        "sub": subject,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh",
    }

    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_token(token: str) -> TokenPayload:
    """
    Decode and validate a JWT token.

    Args:
        token: JWT token string

    Returns:
        Decoded token payload

    Raises:
        JWTError: If token is invalid or expired
    """
    payload = jwt.decode(
        token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
    )
    return TokenPayload(**payload)


async def get_current_user(
    security_scopes: SecurityScopes,
    token: str = Depends(oauth2_scheme),
) -> TokenData:
    """
    Get current authenticated user from JWT token.

    This is a dependency that validates the token and checks required scopes.

    Args:
        security_scopes: Required scopes for the endpoint
        token: JWT token from Authorization header

    Returns:
        Token data with user info

    Raises:
        HTTPException: If authentication or authorization fails
    """
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
    else:
        authenticate_value = "Bearer"

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": authenticate_value},
    )

    try:
        payload = decode_token(token)

        if payload.type != "access":
            raise credentials_exception

        user_id = payload.sub
        if user_id is None:
            raise credentials_exception

        token_scopes = payload.scopes

    except (JWTError, ValidationError):
        raise credentials_exception

    # Check required scopes
    for scope in security_scopes.scopes:
        if scope not in token_scopes and "admin" not in token_scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Not enough permissions. Required scope: {scope}",
                headers={"WWW-Authenticate": authenticate_value},
            )

    return TokenData(user_id=user_id, scopes=token_scopes)


async def get_current_active_user(
    current_user: TokenData = Depends(get_current_user),
) -> TokenData:
    """Get current active user (additional checks can be added here)."""
    # Could add additional checks like user.is_active
    return current_user


def require_scopes(*scopes: str):
    """
    Decorator-style scope requirement for endpoints.

    Usage:
        @router.get("/admin-only")
        async def admin_endpoint(user = Depends(require_scopes("admin"))):
            ...
    """
    async def scope_checker(
        token: str = Depends(oauth2_scheme),
    ) -> TokenData:
        try:
            payload = decode_token(token)
            token_scopes = payload.scopes

            for scope in scopes:
                if scope not in token_scopes and "admin" not in token_scopes:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Required scope: {scope}",
                    )

            return TokenData(user_id=payload.sub, scopes=token_scopes)

        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )

    return scope_checker


class RateLimiter:
    """Simple in-memory rate limiter (use Redis in production)."""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self._requests: Dict[str, List[datetime]] = {}

    def is_allowed(self, client_id: str) -> bool:
        """Check if client is allowed to make request."""
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)

        if client_id not in self._requests:
            self._requests[client_id] = []

        # Clean old requests
        self._requests[client_id] = [
            req_time for req_time in self._requests[client_id]
            if req_time > minute_ago
        ]

        if len(self._requests[client_id]) >= self.requests_per_minute:
            return False

        self._requests[client_id].append(now)
        return True


# Global rate limiter instance
rate_limiter = RateLimiter(requests_per_minute=settings.RATE_LIMIT_PER_MINUTE)


async def check_rate_limit(request: Request):
    """Dependency to check rate limits."""
    client_ip = request.client.host if request.client else "unknown"

    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
        )
