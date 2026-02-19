"""Common schemas used across the API."""
from typing import Generic, TypeVar, List, Optional
from datetime import datetime

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response wrapper."""
    items: List[T]
    total: int
    page: int
    page_size: int
    total_pages: int

    model_config = ConfigDict(from_attributes=True)


class HealthCheck(BaseModel):
    """API health check response."""
    status: str = "healthy"
    version: str
    timestamp: datetime
    database: str = "connected"
    mqtt: str = "connected"
    redis: str = "connected"

    model_config = ConfigDict(from_attributes=True)


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
    code: Optional[str] = None
    timestamp: datetime = datetime.utcnow()


class SuccessResponse(BaseModel):
    """Generic success response."""
    message: str
    success: bool = True
