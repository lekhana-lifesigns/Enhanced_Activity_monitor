"""Patient baseline model for ML models - matches edge cache schema."""
from datetime import datetime
from typing import Optional

from sqlalchemy import String, Integer, DateTime, LargeBinary, Index
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base


class PatientBaseline(Base):
    """
    Patient baseline models for anomaly detection.
    Stores DBSCAN + PCA per-patient baseline models.
    Schema aligned with edge storage/db.py patient_baselines table.
    """

    __tablename__ = "patient_baselines"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # Patient and device (matches edge schema with UNIQUE constraint)
    patient_id: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    device_id: Mapped[str] = mapped_column(String(100), index=True, nullable=False)

    # Baseline type (matches edge schema, default 'features')
    baseline_type: Mapped[str] = mapped_column(
        String(50), nullable=False, default="features"
    )
    # Types: features, posture, activity, temporal

    # Serialized model (matches edge schema - BLOB)
    model_blob: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    # Model statistics (matches edge schema)
    n_samples: Mapped[int] = mapped_column(Integer, default=0)
    n_clusters: Mapped[int] = mapped_column(Integer, default=0)
    fit_count: Mapped[int] = mapped_column(Integer, default=0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Additional backend fields for model management
    model_version: Mapped[str] = mapped_column(String(50), nullable=True)
    accuracy_score: Mapped[float] = mapped_column(nullable=True)
    silhouette_score: Mapped[float] = mapped_column(nullable=True)
    training_duration_ms: Mapped[int] = mapped_column(nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True)

    __table_args__ = (
        Index("idx_baseline_patient_device", "patient_id", "device_id"),
        Index("idx_baseline_patient_device_type", "patient_id", "device_id", "baseline_type", unique=True),
    )

    def __repr__(self) -> str:
        return f"<PatientBaseline patient={self.patient_id} device={self.device_id} type={self.baseline_type} samples={self.n_samples}>"
