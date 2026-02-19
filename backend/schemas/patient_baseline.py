"""Patient baseline schemas for ML models - aligned with edge cache."""
from typing import Optional, List
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class PatientBaselineCreate(BaseModel):
    """Schema for creating/updating a patient baseline."""
    patient_id: str
    device_id: str
    baseline_type: str = "features"
    model_blob: bytes
    n_samples: int = 0
    n_clusters: int = 0
    fit_count: int = 0
    model_version: Optional[str] = None
    accuracy_score: Optional[float] = None
    silhouette_score: Optional[float] = None
    training_duration_ms: Optional[int] = None


class PatientBaselineFromEdge(BaseModel):
    """Schema for receiving baseline from edge (matches edge db.py format)."""
    patient_id: str
    device_id: str
    baseline_type: str = "features"
    model_blob: bytes
    n_samples: int = 0
    n_clusters: int = 0
    fit_count: int = 0


class PatientBaselineResponse(BaseModel):
    """Schema for patient baseline response (without model blob)."""
    id: int
    patient_id: str
    device_id: str
    baseline_type: str
    n_samples: int
    n_clusters: int
    fit_count: int
    model_version: Optional[str]
    accuracy_score: Optional[float]
    silhouette_score: Optional[float]
    training_duration_ms: Optional[int]
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PatientBaselineDetail(PatientBaselineResponse):
    """Schema for patient baseline with model blob."""
    model_blob: bytes


class PatientBaselineList(BaseModel):
    """Paginated patient baseline list."""
    items: List[PatientBaselineResponse]
    total: int
    page: int
    page_size: int


class BaselineStatistics(BaseModel):
    """Baseline training statistics."""
    patient_id: str
    device_id: str
    baseline_type: str
    n_samples: int
    n_clusters: int
    fit_count: int
    last_updated: datetime
    accuracy_score: Optional[float]
    silhouette_score: Optional[float]
