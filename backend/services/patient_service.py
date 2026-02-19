"""Patient service - business logic for patient management."""

from datetime import datetime
from typing import Optional
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.patient import Patient, PatientStatus
from backend.schemas.patient import PatientCreate, PatientUpdate


class PatientService:
    """Service for managing patients."""

    @staticmethod
    async def create_patient(
        db: AsyncSession,
        patient_data: PatientCreate,
    ) -> Patient:
        """Create a new patient record."""
        patient = Patient(
            patient_id=patient_data.patient_id,
            mrn=patient_data.mrn,
            room_id=patient_data.room_id,
            bed_id=patient_data.bed_id,
            risk_level=patient_data.risk_level,
            mobility_status=patient_data.mobility_status,
            fall_risk_score=patient_data.fall_risk_score,
            notes=patient_data.notes,
            metadata=patient_data.metadata,
        )
        db.add(patient)
        await db.commit()
        await db.refresh(patient)
        return patient

    @staticmethod
    async def get_patient(db: AsyncSession, patient_id: str) -> Optional[Patient]:
        """Get patient by patient_id."""
        result = await db.execute(
            select(Patient).where(Patient.patient_id == patient_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_patient_by_mrn(db: AsyncSession, mrn: str) -> Optional[Patient]:
        """Get patient by MRN."""
        result = await db.execute(
            select(Patient).where(Patient.mrn == mrn)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_patients(
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        status: Optional[PatientStatus] = None,
        room_id: Optional[str] = None,
        risk_level: Optional[str] = None,
    ) -> list[Patient]:
        """Get patients with filtering."""
        query = select(Patient)

        conditions = []
        if status:
            conditions.append(Patient.status == status)
        if room_id:
            conditions.append(Patient.room_id == room_id)
        if risk_level:
            conditions.append(Patient.risk_level == risk_level)

        if conditions:
            query = query.where(and_(*conditions))

        query = query.order_by(Patient.admitted_at.desc()).offset(skip).limit(limit)
        result = await db.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def update_patient(
        db: AsyncSession,
        patient_id: str,
        update_data: PatientUpdate,
    ) -> Optional[Patient]:
        """Update patient record."""
        patient = await PatientService.get_patient(db, patient_id)
        if not patient:
            return None

        update_dict = update_data.model_dump(exclude_unset=True)
        for field, value in update_dict.items():
            setattr(patient, field, value)

        await db.commit()
        await db.refresh(patient)
        return patient

    @staticmethod
    async def admit_patient(
        db: AsyncSession,
        patient_id: str,
        room_id: str,
        bed_id: Optional[str] = None,
    ) -> Optional[Patient]:
        """Admit or transfer a patient to a room."""
        patient = await PatientService.get_patient(db, patient_id)
        if not patient:
            return None

        patient.status = PatientStatus.ADMITTED
        patient.room_id = room_id
        patient.bed_id = bed_id
        patient.admitted_at = datetime.utcnow()
        patient.discharged_at = None

        await db.commit()
        await db.refresh(patient)
        return patient

    @staticmethod
    async def discharge_patient(
        db: AsyncSession,
        patient_id: str,
    ) -> Optional[Patient]:
        """Discharge a patient."""
        patient = await PatientService.get_patient(db, patient_id)
        if not patient:
            return None

        patient.status = PatientStatus.DISCHARGED
        patient.discharged_at = datetime.utcnow()
        patient.room_id = None
        patient.bed_id = None

        await db.commit()
        await db.refresh(patient)
        return patient

    @staticmethod
    async def update_risk_assessment(
        db: AsyncSession,
        patient_id: str,
        risk_level: str,
        fall_risk_score: Optional[float] = None,
        mobility_status: Optional[str] = None,
    ) -> Optional[Patient]:
        """Update patient risk assessment."""
        patient = await PatientService.get_patient(db, patient_id)
        if not patient:
            return None

        patient.risk_level = risk_level
        if fall_risk_score is not None:
            patient.fall_risk_score = fall_risk_score
        if mobility_status:
            patient.mobility_status = mobility_status

        await db.commit()
        await db.refresh(patient)
        return patient

    @staticmethod
    async def get_high_risk_patients(db: AsyncSession) -> list[Patient]:
        """Get all high-risk admitted patients."""
        query = select(Patient).where(
            and_(
                Patient.status == PatientStatus.ADMITTED,
                Patient.risk_level.in_(["high", "critical"])
            )
        ).order_by(Patient.fall_risk_score.desc().nullsfirst())

        result = await db.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def get_patient_statistics(db: AsyncSession) -> dict:
        """Get patient statistics."""
        patients = await PatientService.get_patients(db, limit=10000)

        status_counts = {status.value: 0 for status in PatientStatus}
        risk_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}

        for patient in patients:
            status_counts[patient.status.value] += 1
            if patient.risk_level in risk_counts:
                risk_counts[patient.risk_level] += 1

        return {
            "total": len(patients),
            "by_status": status_counts,
            "by_risk_level": risk_counts,
        }

    @staticmethod
    async def assign_to_room(
        db: AsyncSession,
        patient_id: str,
        room_id: str,
        bed_id: Optional[str] = None,
    ) -> Optional[Patient]:
        """Assign/transfer patient to a new room."""
        patient = await PatientService.get_patient(db, patient_id)
        if not patient:
            return None

        patient.room_id = room_id
        patient.bed_id = bed_id

        await db.commit()
        await db.refresh(patient)
        return patient
