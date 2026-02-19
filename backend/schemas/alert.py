"""Alert schemas."""
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict


class AlertLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class AlertStatus(str, Enum):
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    DISMISSED = "DISMISSED"


class AlertCreate(BaseModel):
    """Schema for creating an alert."""
    device_id: str
    patient_id: Optional[str] = None
    alert_level: AlertLevel
    alert_type: str
    title: str
    description: Optional[str] = None
    confidence: Optional[float] = None
    agitation_score: Optional[float] = None
    pain_score: Optional[float] = None
    delirium_risk: Optional[float] = None
    respiratory_distress: Optional[float] = None  # Clinical field from edge
    hand_proximity_risk: Optional[float] = None   # Clinical field from edge
    anomaly_score: Optional[float] = None
    activity_label: Optional[str] = None
    posture_state: Optional[str] = None
    is_policy_violation: bool = False
    violation_type: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None


class AlertFromEdge(BaseModel):
    """Schema for receiving alert from edge device (matches edge db.py format)."""
    ts: float  # Unix timestamp
    device: str
    alert_level: str  # CRITICAL, HIGH, MEDIUM, LOW
    label: Optional[str] = None
    agitation_score: Optional[float] = None
    delirium_risk: Optional[float] = None
    respiratory_distress: Optional[float] = None
    hand_proximity_risk: Optional[float] = None
    payload: Optional[Dict[str, Any]] = None


class AlertResponse(BaseModel):
    """Schema for alert response."""
    id: int
    device_id: str
    patient_id: Optional[str]
    alert_level: AlertLevel
    alert_type: str
    title: str
    description: Optional[str]
    status: AlertStatus
    created_at: datetime
    acknowledged_at: Optional[datetime]
    resolved_at: Optional[datetime]
    acknowledged_by: Optional[str]
    resolved_by: Optional[str]
    confidence: Optional[float]
    agitation_score: Optional[float]
    pain_score: Optional[float]
    delirium_risk: Optional[float]
    respiratory_distress: Optional[float]  # Clinical field from edge
    hand_proximity_risk: Optional[float]   # Clinical field from edge
    anomaly_score: Optional[float]
    activity_label: Optional[str]
    posture_state: Optional[str]
    is_policy_violation: bool
    violation_type: Optional[str]
    escalation_count: int
    resolution_notes: Optional[str]

    model_config = ConfigDict(from_attributes=True)


class AlertUpdate(BaseModel):
    """Schema for updating an alert."""
    status: Optional[AlertStatus] = None
    resolution_notes: Optional[str] = None


class AlertAcknowledge(BaseModel):
    """Schema for acknowledging an alert."""
    notes: Optional[str] = None


class AlertResolve(BaseModel):
    """Schema for resolving an alert."""
    resolution_notes: str


class AlertList(BaseModel):
    """Paginated alert list response."""
    items: List[AlertResponse]
    total: int
    page: int
    page_size: int
    active_count: int
    critical_count: int


class AlertFilter(BaseModel):
    """Alert query filters."""
    device_id: Optional[str] = None
    patient_id: Optional[str] = None
    alert_level: Optional[AlertLevel] = None
    status: Optional[AlertStatus] = None
    alert_type: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    is_policy_violation: Optional[bool] = None
