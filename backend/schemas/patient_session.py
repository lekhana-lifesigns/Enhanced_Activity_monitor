"""Patient session schemas - aligned with edge cache."""
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict


class SessionStatus(str, Enum):
    """Patient session status."""
    ACTIVE = "active"
    ENDED = "ended"
    TRANSFERRED = "transferred"
    DISCHARGED = "discharged"


class PatientSessionCreate(BaseModel):
    """Schema for creating a patient session."""
    patient_id: str
    device_id: str
    bed_id: Optional[str] = None
    room_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PatientSessionFromEdge(BaseModel):
    """Schema for receiving session from edge (matches edge db.py format)."""
    patient_id: str
    device_id: str
    bed_id: Optional[str] = None
    start_ts: float
    end_ts: Optional[float] = None
    status: str = "active"
    metadata: Optional[Dict[str, Any]] = None


class PatientSessionResponse(BaseModel):
    """Schema for patient session response."""
    id: int
    patient_id: str
    device_id: str
    bed_id: Optional[str]
    room_id: Optional[str]
    start_ts: float
    end_ts: Optional[float]
    status: SessionStatus
    metadata: Optional[Dict[str, Any]]
    nurse_id: Optional[str]
    transfer_reason: Optional[str]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PatientSessionUpdate(BaseModel):
    """Schema for updating a patient session."""
    bed_id: Optional[str] = None
    room_id: Optional[str] = None
    nurse_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PatientSessionEnd(BaseModel):
    """Schema for ending a patient session."""
    transfer_reason: Optional[str] = None


class PatientSessionList(BaseModel):
    """Paginated patient session list."""
    items: List[PatientSessionResponse]
    total: int
    page: int
    page_size: int
