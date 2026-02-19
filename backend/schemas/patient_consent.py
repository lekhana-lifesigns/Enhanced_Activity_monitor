"""Patient consent schemas for GDPR compliance - aligned with edge cache."""
from typing import Optional, Dict, Any, List
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class PatientConsentCreate(BaseModel):
    """Schema for creating/updating patient consent."""
    patient_id: str
    consent_type: str
    granted: bool = False
    granted_by: Optional[str] = None
    expires_at: Optional[datetime] = None
    version: Optional[str] = None
    legal_basis: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PatientConsentFromEdge(BaseModel):
    """Schema for receiving consent from edge (matches edge db.py format)."""
    patient_id: str
    consent_type: str
    granted: bool = False
    granted_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class PatientConsentResponse(BaseModel):
    """Schema for patient consent response."""
    id: int
    patient_id: str
    consent_type: str
    granted: bool
    granted_at: Optional[datetime]
    revoked_at: Optional[datetime]
    granted_by: Optional[str]
    revoked_by: Optional[str]
    revocation_reason: Optional[str]
    expires_at: Optional[datetime]
    version: Optional[str]
    legal_basis: Optional[str]
    metadata: Optional[Dict[str, Any]]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PatientConsentRevoke(BaseModel):
    """Schema for revoking consent."""
    revocation_reason: Optional[str] = None
    revoked_by: Optional[str] = None


class PatientConsentList(BaseModel):
    """Paginated patient consent list."""
    items: List[PatientConsentResponse]
    total: int
    page: int
    page_size: int


class ConsentStatus(BaseModel):
    """Summary of consent status for a patient."""
    patient_id: str
    consents: Dict[str, bool]  # consent_type -> granted
    all_required_granted: bool
    missing_consents: List[str]
