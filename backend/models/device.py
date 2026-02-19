"""Device model for edge device management."""
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum

from sqlalchemy import String, Float, DateTime, JSON, Index, Boolean, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base


class DeviceStatus(str, Enum):
    """Device operational status."""
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    MAINTENANCE = "MAINTENANCE"
    ERROR = "ERROR"


class Device(Base):
    """
    Edge device (Jetson) registration and status tracking.
    Manages device configuration, health, and association with patients.
    """

    __tablename__ = "devices"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # Device identification
    device_id: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=True)  # Human-readable name
    description: Mapped[str] = mapped_column(String(500), nullable=True)

    # Location
    facility_id: Mapped[str] = mapped_column(String(100), index=True, nullable=True)
    location: Mapped[str] = mapped_column(String(255), nullable=True)  # Room/bed info
    bed_number: Mapped[str] = mapped_column(String(50), nullable=True)

    # Current patient assignment
    patient_id: Mapped[str] = mapped_column(String(100), index=True, nullable=True)

    # Status
    status: Mapped[DeviceStatus] = mapped_column(
        SQLEnum(DeviceStatus),
        index=True,
        default=DeviceStatus.OFFLINE
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Last communication
    last_heartbeat: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    last_event_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    last_alert_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    # Health metrics (from last heartbeat)
    cpu_percent: Mapped[float] = mapped_column(Float, nullable=True)
    ram_percent: Mapped[float] = mapped_column(Float, nullable=True)
    temperature: Mapped[float] = mapped_column(Float, nullable=True)  # Celsius
    fps: Mapped[float] = mapped_column(Float, nullable=True)
    inference_ms: Mapped[float] = mapped_column(Float, nullable=True)

    # Baseline status
    baseline_state: Mapped[str] = mapped_column(String(50), nullable=True)  # collecting, ready
    baseline_samples: Mapped[int] = mapped_column(nullable=True)

    # Configuration
    config: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=True)

    # Registration info
    registered_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Version info
    firmware_version: Mapped[str] = mapped_column(String(50), nullable=True)
    model_version: Mapped[str] = mapped_column(String(50), nullable=True)

    __table_args__ = (
        Index("idx_devices_facility", "facility_id"),
        Index("idx_devices_status", "status", "is_active"),
        Index("idx_devices_patient", "patient_id"),
    )

    def __repr__(self) -> str:
        return f"<Device {self.device_id} [{self.status.value}]>"
