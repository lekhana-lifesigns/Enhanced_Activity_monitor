"""System health schemas - aligned with edge cache."""
from typing import Optional, List
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class SystemHealthCreate(BaseModel):
    """Schema for creating system health record (from edge device)."""
    ts: float  # Unix timestamp
    device_id: str
    cpu_percent: Optional[float] = None
    ram_percent: Optional[float] = None
    temp_celsius: Optional[float] = None
    disk_percent: Optional[float] = None
    inference_ms: Optional[float] = None
    fps: Optional[float] = None
    gpu_percent: Optional[float] = None
    gpu_memory_percent: Optional[float] = None


class SystemHealthFromEdge(BaseModel):
    """Schema for receiving health from edge (matches edge db.py format)."""
    ts: float
    device_id: str
    cpu_percent: Optional[float] = None
    ram_percent: Optional[float] = None
    temp_celsius: Optional[float] = None
    disk_percent: Optional[float] = None
    inference_ms: Optional[float] = None
    fps: Optional[float] = None


class SystemHealthResponse(BaseModel):
    """Schema for system health response."""
    id: int
    ts: float
    timestamp: datetime
    device_id: str
    cpu_percent: Optional[float]
    ram_percent: Optional[float]
    temp_celsius: Optional[float]
    disk_percent: Optional[float]
    inference_ms: Optional[float]
    fps: Optional[float]
    gpu_percent: Optional[float]
    gpu_memory_percent: Optional[float]

    model_config = ConfigDict(from_attributes=True)


class SystemHealthList(BaseModel):
    """Paginated system health list."""
    items: List[SystemHealthResponse]
    total: int
    page: int
    page_size: int


class DeviceHealthSummary(BaseModel):
    """Summary of device health metrics."""
    device_id: str
    last_heartbeat: Optional[datetime]
    avg_cpu_percent: float
    avg_ram_percent: float
    avg_fps: float
    avg_inference_ms: float
    max_temp_celsius: Optional[float]
    sample_count: int
