"""Alert model for clinical notifications."""
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum

from sqlalchemy import String, Float, DateTime, JSON, Index, Boolean, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base


class AlertLevel(str, Enum):
    """Alert severity levels."""
    CRITICAL = "CRITICAL"  # Immediate action required (fall, seizure)
    HIGH = "HIGH"  # Urgent attention (bed exit, high agitation)
    MEDIUM = "MEDIUM"  # Attention needed (policy violation, anomaly)
    LOW = "LOW"  # Informational (posture change)


class AlertStatus(str, Enum):
    """Alert lifecycle status."""
    ACTIVE = "ACTIVE"  # New, unacknowledged
    ACKNOWLEDGED = "ACKNOWLEDGED"  # Seen by staff
    IN_PROGRESS = "IN_PROGRESS"  # Being addressed
    RESOLVED = "RESOLVED"  # Issue resolved
    DISMISSED = "DISMISSED"  # False positive


class Alert(Base):
    """
    Clinical alert for staff notification.
    Tracks the full lifecycle from creation to resolution.
    """

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # Device and patient
    device_id: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    patient_id: Mapped[str] = mapped_column(String(100), index=True, nullable=True)

    # Alert classification
    alert_level: Mapped[AlertLevel] = mapped_column(
        SQLEnum(AlertLevel),
        index=True,
        nullable=False,
        default=AlertLevel.LOW
    )
    alert_type: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    # Types: FALL, BED_EXIT, HIGH_AGITATION, SEIZURE, POLICY_VIOLATION, ANOMALY, etc.

    # Description
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(String(1000), nullable=True)

    # Status tracking
    status: Mapped[AlertStatus] = mapped_column(
        SQLEnum(AlertStatus),
        index=True,
        nullable=False,
        default=AlertStatus.ACTIVE
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True, default=datetime.utcnow)
    acknowledged_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    resolved_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    # Who handled it
    acknowledged_by: Mapped[str] = mapped_column(String(255), nullable=True)
    resolved_by: Mapped[str] = mapped_column(String(255), nullable=True)

    # Clinical scores at time of alert (matches edge cache schema)
    confidence: Mapped[float] = mapped_column(Float, nullable=True)
    agitation_score: Mapped[float] = mapped_column(Float, nullable=True)
    pain_score: Mapped[float] = mapped_column(Float, nullable=True)
    delirium_risk: Mapped[float] = mapped_column(Float, nullable=True)
    respiratory_distress: Mapped[float] = mapped_column(Float, nullable=True)
    hand_proximity_risk: Mapped[float] = mapped_column(Float, nullable=True)
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=True)

    # Context
    activity_label: Mapped[str] = mapped_column(String(100), nullable=True)
    posture_state: Mapped[str] = mapped_column(String(50), nullable=True)

    # Policy violation details
    is_policy_violation: Mapped[bool] = mapped_column(Boolean, default=False)
    violation_type: Mapped[str] = mapped_column(String(100), nullable=True)

    # Escalation
    escalation_count: Mapped[int] = mapped_column(default=0)
    last_escalation_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    # Resolution notes
    resolution_notes: Mapped[str] = mapped_column(String(2000), nullable=True)

    # Full payload
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=True)

    # Indexes for common queries
    __table_args__ = (
        Index("idx_alerts_device_time", "device_id", "created_at"),
        Index("idx_alerts_patient_time", "patient_id", "created_at"),
        Index("idx_alerts_level_status", "alert_level", "status"),
        Index("idx_alerts_active", "status", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Alert {self.id} {self.alert_level.value} {self.alert_type} [{self.status.value}]>"
