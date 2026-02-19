"""Event service - business logic for activity events."""

from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.event import Event
from backend.models.device import Device
from backend.schemas.event import EventCreate


class EventService:
    """Service for managing activity events."""

    @staticmethod
    async def create_event(
        db: AsyncSession,
        event_data: EventCreate,
        device_id: Optional[str] = None
    ) -> Event:
        """Create a new activity event."""
        event = Event(
            device_id=device_id,
            patient_id=event_data.patient_id,
            event_type=event_data.event_type,
            activity_class=event_data.activity_class,
            confidence=event_data.confidence,
            keypoints=event_data.keypoints,
            metadata=event_data.metadata,
        )
        db.add(event)
        await db.commit()
        await db.refresh(event)
        return event

    @staticmethod
    async def get_event(db: AsyncSession, event_id: int) -> Optional[Event]:
        """Get event by ID."""
        result = await db.execute(select(Event).where(Event.id == event_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_events(
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        device_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[Event]:
        """Get events with filtering."""
        query = select(Event)

        conditions = []
        if device_id:
            conditions.append(Event.device_id == device_id)
        if patient_id:
            conditions.append(Event.patient_id == patient_id)
        if event_type:
            conditions.append(Event.event_type == event_type)
        if start_time:
            conditions.append(Event.timestamp >= start_time)
        if end_time:
            conditions.append(Event.timestamp <= end_time)

        if conditions:
            query = query.where(and_(*conditions))

        query = query.order_by(Event.timestamp.desc()).offset(skip).limit(limit)
        result = await db.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def get_event_count(
        db: AsyncSession,
        device_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        """Get count of events matching filters."""
        query = select(func.count(Event.id))

        conditions = []
        if device_id:
            conditions.append(Event.device_id == device_id)
        if patient_id:
            conditions.append(Event.patient_id == patient_id)
        if event_type:
            conditions.append(Event.event_type == event_type)
        if start_time:
            conditions.append(Event.timestamp >= start_time)
        if end_time:
            conditions.append(Event.timestamp <= end_time)

        if conditions:
            query = query.where(and_(*conditions))

        result = await db.execute(query)
        return result.scalar() or 0

    @staticmethod
    async def get_activity_summary(
        db: AsyncSession,
        patient_id: str,
        hours: int = 24
    ) -> dict:
        """Get activity summary for a patient over specified hours."""
        start_time = datetime.utcnow() - timedelta(hours=hours)

        query = select(
            Event.activity_class,
            func.count(Event.id).label('count'),
            func.avg(Event.confidence).label('avg_confidence')
        ).where(
            and_(
                Event.patient_id == patient_id,
                Event.timestamp >= start_time
            )
        ).group_by(Event.activity_class)

        result = await db.execute(query)
        rows = result.all()

        return {
            "patient_id": patient_id,
            "period_hours": hours,
            "activities": [
                {
                    "activity": row.activity_class,
                    "count": row.count,
                    "avg_confidence": round(float(row.avg_confidence), 3)
                }
                for row in rows
            ]
        }

    @staticmethod
    async def delete_old_events(
        db: AsyncSession,
        days: int = 30
    ) -> int:
        """Delete events older than specified days."""
        cutoff = datetime.utcnow() - timedelta(days=days)

        result = await db.execute(
            select(Event).where(Event.timestamp < cutoff)
        )
        events = result.scalars().all()
        count = len(events)

        for event in events:
            await db.delete(event)

        await db.commit()
        return count
