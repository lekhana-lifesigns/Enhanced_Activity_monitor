"""Event schemas."""
from typing import Optional, Dict, Any, List
from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict


class EventBase(BaseModel):
    """Base event schema."""
    device_id: str
    patient_id: Optional[str] = None
    activity_label: Optional[str] = None
    activity_confidence: Optional[float] = None
    posture_state: Optional[str] = None
    posture_confidence: Optional[float] = None


class EventCreate(EventBase):
    """Schema for creating an event (from edge device)."""
    timestamp: Optional[datetime] = None
    agitation_score: Optional[float] = None
    pain_score: Optional[float] = None
    delirium_risk: Optional[float] = None
    anomaly_score: Optional[float] = None
    baseline_state: Optional[str] = None
    cluster_id: Optional[int] = None
    person_present: bool = True
    bbox: Optional[List[int]] = None  # [x1, y1, x2, y2]
    track_id: Optional[int] = None
    fall_detected: bool = False
    fall_confidence: Optional[float] = None
    person_on_bed: Optional[bool] = None
    person_near_bed: Optional[bool] = None
    inference_ms: Optional[float] = None
    fps: Optional[float] = None
    payload: Optional[Dict[str, Any]] = None


class EventResponse(EventBase):
    """Schema for event response."""
    id: int
    timestamp: datetime
    agitation_score: Optional[float]
    pain_score: Optional[float]
    delirium_risk: Optional[float]
    anomaly_score: Optional[float]
    baseline_state: Optional[str]
    person_present: bool
    fall_detected: bool
    fall_confidence: Optional[float]
    inference_ms: Optional[float]
    fps: Optional[float]

    model_config = ConfigDict(from_attributes=True)


class EventList(BaseModel):
    """Paginated event list response."""
    items: List[EventResponse]
    total: int
    page: int
    page_size: int


class EventFilter(BaseModel):
    """Event query filters."""
    device_id: Optional[str] = None
    patient_id: Optional[str] = None
    activity_label: Optional[str] = None
    posture_state: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    min_anomaly_score: Optional[float] = None
    fall_detected: Optional[bool] = None
    person_present: Optional[bool] = None


class EventFromEdge(BaseModel):
    """Schema for receiving event from edge device (matches edge db.py format).

    This schema matches the structure used by the edge storage/db.py insert_event.
    """
    ts: float  # Unix timestamp
    device: str  # Device ID
    label: Optional[str] = None  # Activity label
    confidence: Optional[float] = None
    payload: Optional[Dict[str, Any]] = None  # Full payload with all details


class EventStatistics(BaseModel):
    """Aggregated event statistics (matches edge db.py get_event_statistics)."""
    total_events: int
    unique_labels: int
    avg_confidence: float
    first_ts: Optional[float]
    last_ts: Optional[float]
