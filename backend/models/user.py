"""User model for authentication and authorization."""
from datetime import datetime
from typing import List

from sqlalchemy import String, Boolean, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base


class User(Base):
    """User model for API authentication."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=True)

    # Role and permissions
    role: Mapped[str] = mapped_column(String(50), default="viewer")  # admin, manager, clinician, viewer
    scopes: Mapped[List[str]] = mapped_column(JSON, default=list)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    # Additional info
    facility_id: Mapped[str] = mapped_column(String(100), nullable=True)  # Multi-tenant support

    def __repr__(self) -> str:
        return f"<User {self.username} ({self.role})>"
