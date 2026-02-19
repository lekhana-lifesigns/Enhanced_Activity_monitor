"""Authentication endpoints."""
from datetime import datetime, timedelta
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.core.database import get_db
from backend.core.security import (
    verify_password,
    hash_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_active_user,
    TokenData,
)
from backend.core.config import settings
from backend.models.user import User
from backend.schemas.auth import (
    Token,
    UserCreate,
    UserResponse,
    UserUpdate,
    RefreshToken,
)

router = APIRouter()


# Default scopes by role
ROLE_SCOPES = {
    "admin": ["admin"],
    "manager": ["read:events", "read:alerts", "manage:alerts", "read:patients", "manage:patients", "read:devices", "manage:devices"],
    "clinician": ["read:events", "read:alerts", "manage:alerts", "read:patients", "read:devices"],
    "viewer": ["read:events", "read:alerts", "read:devices"],
}


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    """
    OAuth2 compatible token login.
    Returns access and refresh tokens.
    """
    # Find user
    result = await db.execute(
        select(User).where(User.username == form_data.username)
    )
    user = result.scalar_one_or_none()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled",
        )

    # Get scopes for user's role
    scopes = user.scopes or ROLE_SCOPES.get(user.role, [])

    # Create tokens
    access_token = create_access_token(
        subject=str(user.id),
        scopes=scopes,
    )
    refresh_token = create_refresh_token(subject=str(user.id))

    # Update last login
    user.last_login = datetime.utcnow()
    await db.commit()

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    token_data: RefreshToken,
    db: AsyncSession = Depends(get_db),
):
    """Refresh access token using refresh token."""
    try:
        payload = decode_token(token_data.refresh_token)

        if payload.type != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )

        user_id = payload.sub

        # Verify user exists and is active
        result = await db.execute(
            select(User).where(User.id == int(user_id))
        )
        user = result.scalar_one_or_none()

        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive",
            )

        # Get scopes
        scopes = user.scopes or ROLE_SCOPES.get(user.role, [])

        # Create new tokens
        access_token = create_access_token(
            subject=str(user.id),
            scopes=scopes,
        )
        new_refresh_token = create_refresh_token(subject=str(user.id))

        return Token(
            access_token=access_token,
            refresh_token=new_refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
):
    """Register a new user (requires admin approval in production)."""
    # Check if user exists
    result = await db.execute(
        select(User).where(
            (User.email == user_data.email) | (User.username == user_data.username)
        )
    )
    existing = result.scalar_one_or_none()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email or username already registered",
        )

    # Create user
    user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hash_password(user_data.password),
        full_name=user_data.full_name,
        role=user_data.role,
        scopes=ROLE_SCOPES.get(user_data.role, []),
        facility_id=user_data.facility_id,
        is_active=True,  # Set to False for approval workflow
        is_verified=False,
    )

    db.add(user)
    await db.commit()
    await db.refresh(user)

    return UserResponse.model_validate(user)


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: TokenData = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Get current user information."""
    result = await db.execute(
        select(User).where(User.id == int(current_user.user_id))
    )
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return UserResponse.model_validate(user)


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    update_data: UserUpdate,
    current_user: TokenData = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Update current user information."""
    result = await db.execute(
        select(User).where(User.id == int(current_user.user_id))
    )
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Update allowed fields
    if update_data.email:
        user.email = update_data.email
    if update_data.full_name:
        user.full_name = update_data.full_name

    user.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(user)

    return UserResponse.model_validate(user)
