"""Patient consent model for GDPR/data governance - matches edge cache schema."""
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import String, Boolean, DateTime, JSON, Index
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base


class PatientConsent(Base):
    """
    Patient consent tracking for GDPR and data governance.
    Records consent grants and revocations for data processing.
    Schema aligned with edge storage/db.py patient_consent table.
    """

    __tablename__ = "patient_consent"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # Patient identification (matches edge schema)
    patient_id: Mapped[str] = mapped_column(String(100), index=True, nullable=False)

    # Consent type (matches edge schema)
    consent_type: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    # Types: VIDEO_MONITORING, DATA_STORAGE, DATA_SHARING, RESEARCH,
    #        FACE_RECOGNITION, ACTIVITY_TRACKING, ALERT_NOTIFICATIONS, etc.

    # Consent status (matches edge schema)
    granted: Mapped[bool] = mapped_column(Boolean, default=False)
    granted_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    revoked_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    # Metadata (matches edge schema - JSON blob)
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=True)

    # Additional backend fields
    granted_by: Mapped[str] = mapped_column(String(100), nullable=True)  # Who gave consent
    revoked_by: Mapped[str] = mapped_column(String(100), nullable=True)
    revocation_reason: Mapped[str] = mapped_column(String(500), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    version: Mapped[str] = mapped_column(String(50), nullable=True)  # Consent form version
    legal_basis: Mapped[str] = mapped_column(String(100), nullable=True)  # GDPR legal basis

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_consent_patient", "patient_id"),
        Index("idx_consent_type", "consent_type"),
        Index("idx_consent_patient_type", "patient_id", "consent_type"),
        Index("idx_consent_granted", "granted"),
    )

    def __repr__(self) -> str:
        status = "granted" if self.granted else "revoked"
        return f"<PatientConsent patient={self.patient_id} type={self.consent_type} {status}>"
