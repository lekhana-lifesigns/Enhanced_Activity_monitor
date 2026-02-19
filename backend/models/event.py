"""Event model for activity monitoring data."""
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import String, Float, DateTime, JSON, Index, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base


class Event(Base):
    """
    Activity event from edge devices.
    Stores inference results, posture, activity, and clinical metrics.
    """

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # Device and patient
    device_id: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    patient_id: Mapped[str] = mapped_column(String(100), index=True, nullable=True)

    # Timestamp
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True, nullable=False, default=datetime.utcnow)

    # Activity classification
    activity_label: Mapped[str] = mapped_column(String(100), index=True, nullable=True)
    activity_confidence: Mapped[float] = mapped_column(Float, nullable=True)

    # Posture classification
    posture_state: Mapped[str] = mapped_column(String(50), index=True, nullable=True)
    posture_confidence: Mapped[float] = mapped_column(Float, nullable=True)

    # Clinical metrics
    agitation_score: Mapped[float] = mapped_column(Float, nullable=True)
    pain_score: Mapped[float] = mapped_column(Float, nullable=True)
    delirium_risk: Mapped[float] = mapped_column(Float, nullable=True)

    # Baseline anomaly detection
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=True)
    baseline_state: Mapped[str] = mapped_column(String(50), nullable=True)  # collecting, ready
    cluster_id: Mapped[int] = mapped_column(nullable=True)

    # Detection info
    person_present: Mapped[bool] = mapped_column(default=True)
    bbox: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=True)  # [x1, y1, x2, y2]
    track_id: Mapped[int] = mapped_column(nullable=True)

    # Fall detection
    fall_detected: Mapped[bool] = mapped_column(default=False)
    fall_confidence: Mapped[float] = mapped_column(Float, nullable=True)

    # Bed context
    person_on_bed: Mapped[bool] = mapped_column(nullable=True)
    person_near_bed: Mapped[bool] = mapped_column(nullable=True)

    # Performance metrics
    inference_ms: Mapped[float] = mapped_column(Float, nullable=True)
    fps: Mapped[float] = mapped_column(Float, nullable=True)

    # Full payload (for debugging/analysis)
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=True)

    # Indexes for common queries
    __table_args__ = (
        Index("idx_events_device_time", "device_id", "timestamp"),
        Index("idx_events_patient_time", "patient_id", "timestamp"),
        Index("idx_events_activity", "activity_label"),
        Index("idx_events_posture", "posture_state"),
        Index("idx_events_anomaly", "anomaly_score"),
    )

    def __repr__(self) -> str:
        return f"<Event {self.id} device={self.device_id} activity={self.activity_label}>"
