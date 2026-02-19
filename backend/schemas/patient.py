"""Patient schemas."""
from typing import Optional, List
from datetime import datetime, date

from pydantic import BaseModel, Field, ConfigDict


class PatientBase(BaseModel):
    """Base patient schema."""
    patient_id: str
    external_id: Optional[str] = None
    date_of_birth: Optional[date] = None
    gender: Optional[str] = None


class PatientCreate(PatientBase):
    """Schema for creating a patient."""
    device_id: Optional[str] = None
    facility_id: Optional[str] = None
    room: Optional[str] = None
    bed: Optional[str] = None
    allowed_postures: List[str] = ["supine", "left_lateral", "right_lateral"]
    forbidden_postures: List[str] = ["prone"]
    must_be_in_bed: bool = True
    fall_risk_level: Optional[str] = None
    mobility_level: Optional[str] = None
    cognitive_status: Optional[str] = None
    notes: Optional[str] = None
    special_instructions: Optional[str] = None
    monitoring_consent: bool = True
    diagnoses: List[str] = []
    medications: List[str] = []


class PatientResponse(PatientBase):
    """Schema for patient response."""
    id: int
    device_id: Optional[str]
    facility_id: Optional[str]
    room: Optional[str]
    bed: Optional[str]
    admitted_at: Optional[datetime]
    discharged_at: Optional[datetime]
    is_active: bool
    allowed_postures: List[str]
    forbidden_postures: List[str]
    must_be_in_bed: bool
    fall_risk_level: Optional[str]
    mobility_level: Optional[str]
    cognitive_status: Optional[str]
    notes: Optional[str]
    special_instructions: Optional[str]
    monitoring_consent: bool
    baseline_state: Optional[str]
    baseline_established_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PatientUpdate(BaseModel):
    """Schema for updating patient."""
    device_id: Optional[str] = None
    facility_id: Optional[str] = None
    room: Optional[str] = None
    bed: Optional[str] = None
    allowed_postures: Optional[List[str]] = None
    forbidden_postures: Optional[List[str]] = None
    must_be_in_bed: Optional[bool] = None
    fall_risk_level: Optional[str] = None
    mobility_level: Optional[str] = None
    cognitive_status: Optional[str] = None
    notes: Optional[str] = None
    special_instructions: Optional[str] = None
    monitoring_consent: Optional[bool] = None
    diagnoses: Optional[List[str]] = None
    medications: Optional[List[str]] = None


class PatientDischarge(BaseModel):
    """Schema for discharging a patient."""
    discharge_notes: Optional[str] = None


class PatientList(BaseModel):
    """Paginated patient list response."""
    items: List[PatientResponse]
    total: int
    active_count: int
