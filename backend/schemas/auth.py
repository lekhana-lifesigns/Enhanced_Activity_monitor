"""Authentication schemas."""
from typing import List, Optional
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field, ConfigDict


class UserCreate(BaseModel):
    """Schema for creating a new user."""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None
    role: str = "viewer"
    facility_id: Optional[str] = None


class UserLogin(BaseModel):
    """Schema for user login."""
    username: str
    password: str


class UserResponse(BaseModel):
    """Schema for user response (excludes password)."""
    id: int
    email: str
    username: str
    full_name: Optional[str]
    role: str
    scopes: List[str]
    is_active: bool
    is_verified: bool
    facility_id: Optional[str]
    created_at: datetime
    last_login: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)


class UserUpdate(BaseModel):
    """Schema for updating user."""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None
    facility_id: Optional[str] = None


class Token(BaseModel):
    """OAuth2 token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenPayload(BaseModel):
    """JWT token payload."""
    sub: str
    exp: datetime
    iat: datetime
    scopes: List[str] = []
    type: str = "access"


class RefreshToken(BaseModel):
    """Refresh token request."""
    refresh_token: str
