"""Audit log schemas for HIPAA compliance - aligned with edge cache."""
from typing import Optional, List
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class AuditLogCreate(BaseModel):
    """Schema for creating an audit log entry."""
    action: str
    user_id: Optional[str] = None
    patient_id: Optional[str] = None
    details: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    device_id: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None


class AuditLogFromEdge(BaseModel):
    """Schema for receiving audit log from edge (matches edge db.py format)."""
    ts: float
    action: str
    user_id: Optional[str] = None
    patient_id: Optional[str] = None
    details: Optional[str] = None
    ip_address: Optional[str] = None


class AuditLogResponse(BaseModel):
    """Schema for audit log response."""
    id: int
    ts: float
    timestamp: datetime
    action: str
    user_id: Optional[str]
    patient_id: Optional[str]
    details: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    session_id: Optional[str]
    device_id: Optional[str]
    resource_type: Optional[str]
    resource_id: Optional[str]
    success: bool
    error_message: Optional[str]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AuditLogList(BaseModel):
    """Paginated audit log list."""
    items: List[AuditLogResponse]
    total: int
    page: int
    page_size: int


class AuditLogFilter(BaseModel):
    """Audit log query filters."""
    user_id: Optional[str] = None
    patient_id: Optional[str] = None
    action: Optional[str] = None
    device_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    success: Optional[bool] = None
