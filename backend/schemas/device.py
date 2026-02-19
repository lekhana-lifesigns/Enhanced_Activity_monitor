"""Device schemas."""
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict


class DeviceStatus(str, Enum):
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    MAINTENANCE = "MAINTENANCE"
    ERROR = "ERROR"


class DeviceCreate(BaseModel):
    """Schema for registering a device."""
    device_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    facility_id: Optional[str] = None
    location: Optional[str] = None
    bed_number: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class DeviceResponse(BaseModel):
    """Schema for device response."""
    id: int
    device_id: str
    name: Optional[str]
    description: Optional[str]
    facility_id: Optional[str]
    location: Optional[str]
    bed_number: Optional[str]
    patient_id: Optional[str]
    status: DeviceStatus
    is_active: bool
    last_heartbeat: Optional[datetime]
    last_event_at: Optional[datetime]
    cpu_percent: Optional[float]
    ram_percent: Optional[float]
    temperature: Optional[float]
    fps: Optional[float]
    inference_ms: Optional[float]
    baseline_state: Optional[str]
    baseline_samples: Optional[int]
    registered_at: datetime
    firmware_version: Optional[str]
    model_version: Optional[str]

    model_config = ConfigDict(from_attributes=True)


class DeviceUpdate(BaseModel):
    """Schema for updating device."""
    name: Optional[str] = None
    description: Optional[str] = None
    facility_id: Optional[str] = None
    location: Optional[str] = None
    bed_number: Optional[str] = None
    patient_id: Optional[str] = None
    is_active: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None


class DeviceHealth(BaseModel):
    """Schema for device health update (heartbeat)."""
    cpu_percent: float
    ram_percent: float
    temperature: Optional[float] = None
    fps: Optional[float] = None
    inference_ms: Optional[float] = None
    baseline_state: Optional[str] = None
    baseline_samples: Optional[int] = None
    firmware_version: Optional[str] = None
    model_version: Optional[str] = None


class DeviceList(BaseModel):
    """Paginated device list response."""
    items: List[DeviceResponse]
    total: int
    online_count: int
    offline_count: int
