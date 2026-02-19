"""Patient session model - matches edge cache schema for multi-device tracking."""
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum

from sqlalchemy import String, Float, DateTime, JSON, Index, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base


class SessionStatus(str, Enum):
    """Patient session status."""
    ACTIVE = "active"
    ENDED = "ended"
    TRANSFERRED = "transferred"
    DISCHARGED = "discharged"


class PatientSession(Base):
    """
    Patient session for multi-device tracking.
    Tracks when a patient is being monitored by a specific device.
    Schema aligned with edge storage/db.py patient_sessions table.
    """

    __tablename__ = "patient_sessions"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # Patient and device (matches edge schema)
    patient_id: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    device_id: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    bed_id: Mapped[str] = mapped_column(String(100), nullable=True)

    # Session timing (edge uses REAL timestamps)
    start_ts: Mapped[float] = mapped_column(Float, nullable=False)
    end_ts: Mapped[float] = mapped_column(Float, nullable=True)

    # Status (matches edge schema values)
    status: Mapped[SessionStatus] = mapped_column(
        SQLEnum(SessionStatus),
        index=True,
        nullable=False,
        default=SessionStatus.ACTIVE
    )

    # Metadata (matches edge schema - JSON blob)
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Additional backend fields for tracking
    room_id: Mapped[str] = mapped_column(String(100), nullable=True)
    nurse_id: Mapped[str] = mapped_column(String(100), nullable=True)
    transfer_reason: Mapped[str] = mapped_column(String(500), nullable=True)

    __table_args__ = (
        Index("idx_session_patient", "patient_id"),
        Index("idx_session_device", "device_id"),
        Index("idx_session_status", "status"),
        Index("idx_session_active", "patient_id", "device_id", "status"),
    )

    def __repr__(self) -> str:
        return f"<PatientSession patient={self.patient_id} device={self.device_id} status={self.status.value}>"
