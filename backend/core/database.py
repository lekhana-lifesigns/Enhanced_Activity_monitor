"""
Database configuration and session management.
Supports both SQLite (development) and PostgreSQL (production).
"""
import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from .config import settings

log = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all models."""
    pass


# Create async engine based on environment
if settings.ENVIRONMENT == "production" and settings.POSTGRES_PASSWORD:
    # Use PostgreSQL in production
    DATABASE_URL = settings.postgres_url
    engine = create_async_engine(
        DATABASE_URL,
        pool_size=settings.DATABASE_POOL_SIZE,
        max_overflow=settings.DATABASE_MAX_OVERFLOW,
        echo=settings.DEBUG,
    )
    log.info("Using PostgreSQL database")
else:
    # Use SQLite for development
    DATABASE_URL = settings.DATABASE_URL
    engine = create_async_engine(
        DATABASE_URL,
        poolclass=NullPool,  # SQLite doesn't support connection pooling
        echo=settings.DEBUG,
    )
    log.info("Using SQLite database: %s", DATABASE_URL)


# Session factory
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)

# Alias for compatibility
AsyncSessionLocal = async_session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides a database session.

    Usage:
        @router.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("Database tables initialized")


async def close_db():
    """Close database connections."""
    await engine.dispose()
    log.info("Database connections closed")
