"""Patients endpoints."""
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status, Security
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from backend.core.database import get_db
from backend.core.security import get_current_active_user, TokenData
from backend.models.patient import Patient
from backend.schemas.patient import (
    PatientCreate,
    PatientResponse,
    PatientUpdate,
    PatientDischarge,
    PatientList,
)

router = APIRouter()


@router.get("", response_model=PatientList)
async def list_patients(
    facility_id: Optional[str] = Query(None),
    device_id: Optional[str] = Query(None),
    active_only: bool = Query(True),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    current_user: TokenData = Security(get_current_active_user, scopes=["read:patients"]),
    db: AsyncSession = Depends(get_db),
):
    """
    List patients with filtering.

    Required scope: read:patients
    """
    query = select(Patient)
    count_query = select(func.count(Patient.id))

    conditions = []
    if facility_id:
        conditions.append(Patient.facility_id == facility_id)
    if device_id:
        conditions.append(Patient.device_id == device_id)
    if active_only:
        conditions.append(Patient.is_active == True)

    if conditions:
        query = query.where(and_(*conditions))
        count_query = count_query.where(and_(*conditions))

    # Get counts
    total_result = await db.execute(count_query)
    total = total_result.scalar()

    active_result = await db.execute(
        select(func.count(Patient.id)).where(Patient.is_active == True)
    )
    active_count = active_result.scalar()

    # Paginate
    offset = (page - 1) * page_size
    query = query.order_by(Patient.patient_id).offset(offset).limit(page_size)

    result = await db.execute(query)
    patients = result.scalars().all()

    return PatientList(
        items=[PatientResponse.model_validate(p) for p in patients],
        total=total,
        active_count=active_count,
    )


@router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(
    patient_id: str,
    current_user: TokenData = Security(get_current_active_user, scopes=["read:patients"]),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific patient by ID."""
    result = await db.execute(
        select(Patient).where(Patient.patient_id == patient_id)
    )
    patient = result.scalar_one_or_none()

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    return PatientResponse.model_validate(patient)


@router.post("", response_model=PatientResponse, status_code=status.HTTP_201_CREATED)
async def create_patient(
    patient_data: PatientCreate,
    current_user: TokenData = Security(get_current_active_user, scopes=["manage:patients"]),
    db: AsyncSession = Depends(get_db),
):
    """
    Create/admit a new patient.

    Required scope: manage:patients
    """
    # Check if patient exists
    result = await db.execute(
        select(Patient).where(Patient.patient_id == patient_data.patient_id)
    )
    existing = result.scalar_one_or_none()

    if existing:
        raise HTTPException(
            status_code=400,
            detail="Patient ID already exists",
        )

    patient = Patient(
        patient_id=patient_data.patient_id,
        external_id=patient_data.external_id,
        date_of_birth=patient_data.date_of_birth,
        gender=patient_data.gender,
        device_id=patient_data.device_id,
        facility_id=patient_data.facility_id,
        room=patient_data.room,
        bed=patient_data.bed,
        admitted_at=datetime.utcnow(),
        is_active=True,
        allowed_postures=patient_data.allowed_postures,
        forbidden_postures=patient_data.forbidden_postures,
        must_be_in_bed=patient_data.must_be_in_bed,
        fall_risk_level=patient_data.fall_risk_level,
        mobility_level=patient_data.mobility_level,
        cognitive_status=patient_data.cognitive_status,
        notes=patient_data.notes,
        special_instructions=patient_data.special_instructions,
        monitoring_consent=patient_data.monitoring_consent,
        consent_given_at=datetime.utcnow() if patient_data.monitoring_consent else None,
        diagnoses=patient_data.diagnoses,
        medications=patient_data.medications,
        created_by=current_user.user_id,
    )

    db.add(patient)
    await db.commit()
    await db.refresh(patient)

    return PatientResponse.model_validate(patient)


@router.put("/{patient_id}", response_model=PatientResponse)
async def update_patient(
    patient_id: str,
    update_data: PatientUpdate,
    current_user: TokenData = Security(get_current_active_user, scopes=["manage:patients"]),
    db: AsyncSession = Depends(get_db),
):
    """
    Update patient information.

    Required scope: manage:patients
    """
    result = await db.execute(
        select(Patient).where(Patient.patient_id == patient_id)
    )
    patient = result.scalar_one_or_none()

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Update fields
    update_fields = update_data.model_dump(exclude_unset=True)
    for field, value in update_fields.items():
        setattr(patient, field, value)

    patient.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(patient)

    return PatientResponse.model_validate(patient)


@router.post("/{patient_id}/discharge", response_model=PatientResponse)
async def discharge_patient(
    patient_id: str,
    discharge_data: PatientDischarge,
    current_user: TokenData = Security(get_current_active_user, scopes=["manage:patients"]),
    db: AsyncSession = Depends(get_db),
):
    """
    Discharge a patient.

    Required scope: manage:patients
    """
    result = await db.execute(
        select(Patient).where(Patient.patient_id == patient_id)
    )
    patient = result.scalar_one_or_none()

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    if not patient.is_active:
        raise HTTPException(status_code=400, detail="Patient is already discharged")

    patient.is_active = False
    patient.discharged_at = datetime.utcnow()
    patient.device_id = None  # Unassign from device

    if discharge_data.discharge_notes:
        patient.notes = (patient.notes or "") + f"\n\nDischarge Notes: {discharge_data.discharge_notes}"

    await db.commit()
    await db.refresh(patient)

    return PatientResponse.model_validate(patient)


@router.post("/{patient_id}/assign-device", response_model=PatientResponse)
async def assign_device(
    patient_id: str,
    device_id: str,
    current_user: TokenData = Security(get_current_active_user, scopes=["manage:patients"]),
    db: AsyncSession = Depends(get_db),
):
    """
    Assign a patient to a device.

    Required scope: manage:patients
    """
    result = await db.execute(
        select(Patient).where(Patient.patient_id == patient_id)
    )
    patient = result.scalar_one_or_none()

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    patient.device_id = device_id
    patient.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(patient)

    return PatientResponse.model_validate(patient)


@router.post("/{patient_id}/reset-baseline", response_model=PatientResponse)
async def reset_baseline(
    patient_id: str,
    current_user: TokenData = Security(get_current_active_user, scopes=["manage:patients"]),
    db: AsyncSession = Depends(get_db),
):
    """
    Reset patient baseline model.

    Required scope: manage:patients
    """
    result = await db.execute(
        select(Patient).where(Patient.patient_id == patient_id)
    )
    patient = result.scalar_one_or_none()

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    patient.baseline_state = None
    patient.baseline_established_at = None
    patient.updated_at = datetime.utcnow()

    # TODO: Send command to edge device to reset baseline

    await db.commit()
    await db.refresh(patient)

    return PatientResponse.model_validate(patient)
