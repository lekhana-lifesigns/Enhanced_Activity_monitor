"""Audit log model for HIPAA compliance - matches edge cache schema."""
from datetime import datetime
from typing import Optional

from sqlalchemy import String, Float, DateTime, Text, Index
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base


class AuditLog(Base):
    """
    Audit log for HIPAA compliance tracking.
    Records all access and modifications to patient data.
    Schema aligned with edge storage/db.py audit_log table.
    """

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # Timestamp (edge uses REAL, backend uses both for compatibility)
    ts: Mapped[float] = mapped_column(Float, index=True, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, index=True, nullable=False, default=datetime.utcnow
    )

    # Action performed (matches edge schema)
    action: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    # Actions: VIEW_PATIENT, UPDATE_PATIENT, VIEW_EVENTS, EXPORT_DATA,
    #          LOGIN, LOGOUT, ACKNOWLEDGE_ALERT, RESOLVE_ALERT, etc.

    # Who performed the action
    user_id: Mapped[str] = mapped_column(String(100), index=True, nullable=True)

    # Which patient was accessed (if applicable)
    patient_id: Mapped[str] = mapped_column(String(100), index=True, nullable=True)

    # Details of the action (matches edge schema)
    details: Mapped[str] = mapped_column(Text, nullable=True)

    # Source IP address (matches edge schema)
    ip_address: Mapped[str] = mapped_column(String(45), nullable=True)  # IPv6 compatible

    # Additional backend fields
    user_agent: Mapped[str] = mapped_column(String(500), nullable=True)
    session_id: Mapped[str] = mapped_column(String(100), nullable=True)
    device_id: Mapped[str] = mapped_column(String(100), nullable=True)
    resource_type: Mapped[str] = mapped_column(String(100), nullable=True)
    resource_id: Mapped[str] = mapped_column(String(100), nullable=True)
    success: Mapped[bool] = mapped_column(default=True)
    error_message: Mapped[str] = mapped_column(String(500), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_audit_ts", "ts"),
        Index("idx_audit_user", "user_id"),
        Index("idx_audit_patient", "patient_id"),
        Index("idx_audit_action", "action"),
        Index("idx_audit_user_time", "user_id", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<AuditLog {self.action} user={self.user_id} patient={self.patient_id}>"
