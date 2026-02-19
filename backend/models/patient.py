"""Patient model for clinical context."""
from datetime import datetime, date
from typing import Optional, Dict, Any, List

from sqlalchemy import String, Float, DateTime, Date, JSON, Index, Boolean, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base


class Patient(Base):
    """
    Patient information for clinical context.
    Stores policies, allowed postures, and clinical notes.
    Note: This is minimal PHI - full EHR integration would use external system.
    """

    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # Patient identification (de-identified or MRN depending on deployment)
    patient_id: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    external_id: Mapped[str] = mapped_column(String(100), nullable=True)  # EHR MRN

    # Demographics (optional, for clinical context)
    date_of_birth: Mapped[date] = mapped_column(Date, nullable=True)
    gender: Mapped[str] = mapped_column(String(20), nullable=True)

    # Current assignment
    device_id: Mapped[str] = mapped_column(String(100), index=True, nullable=True)
    facility_id: Mapped[str] = mapped_column(String(100), index=True, nullable=True)
    room: Mapped[str] = mapped_column(String(50), nullable=True)
    bed: Mapped[str] = mapped_column(String(50), nullable=True)

    # Admission info
    admitted_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    discharged_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)

    # Monitoring policies
    allowed_postures: Mapped[List[str]] = mapped_column(JSON, default=list)
    # e.g., ["supine", "left_lateral", "right_lateral"]
    forbidden_postures: Mapped[List[str]] = mapped_column(JSON, default=list)
    # e.g., ["prone"]
    must_be_in_bed: Mapped[bool] = mapped_column(Boolean, default=True)

    # Risk factors
    fall_risk_level: Mapped[str] = mapped_column(String(20), nullable=True)  # low, medium, high
    mobility_level: Mapped[str] = mapped_column(String(50), nullable=True)  # ambulatory, assisted, bed-bound
    cognitive_status: Mapped[str] = mapped_column(String(50), nullable=True)  # alert, confused, sedated

    # Clinical notes (free text for nursing context)
    notes: Mapped[str] = mapped_column(Text, nullable=True)
    special_instructions: Mapped[str] = mapped_column(Text, nullable=True)

    # Consent
    monitoring_consent: Mapped[bool] = mapped_column(Boolean, default=True)
    consent_given_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    consent_given_by: Mapped[str] = mapped_column(String(255), nullable=True)

    # Baseline info
    baseline_state: Mapped[str] = mapped_column(String(50), nullable=True)
    baseline_established_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by: Mapped[str] = mapped_column(String(255), nullable=True)

    # Additional clinical data
    diagnoses: Mapped[List[str]] = mapped_column(JSON, default=list)  # ICD-10 codes
    medications: Mapped[List[str]] = mapped_column(JSON, default=list)  # Relevant medications

    __table_args__ = (
        Index("idx_patients_facility", "facility_id", "is_active"),
        Index("idx_patients_device", "device_id"),
        Index("idx_patients_active", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<Patient {self.patient_id} device={self.device_id}>"
