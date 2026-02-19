"""Events endpoints."""
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query, status, Security
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc

from backend.core.database import get_db
from backend.core.security import get_current_active_user, TokenData
from backend.models.event import Event
from backend.schemas.event import EventCreate, EventResponse, EventList, EventFilter

router = APIRouter()


@router.get("", response_model=EventList)
async def list_events(
    device_id: Optional[str] = Query(None, description="Filter by device ID"),
    patient_id: Optional[str] = Query(None, description="Filter by patient ID"),
    activity_label: Optional[str] = Query(None, description="Filter by activity"),
    posture_state: Optional[str] = Query(None, description="Filter by posture"),
    start_time: Optional[datetime] = Query(None, description="Start time filter"),
    end_time: Optional[datetime] = Query(None, description="End time filter"),
    min_anomaly_score: Optional[float] = Query(None, description="Minimum anomaly score"),
    fall_detected: Optional[bool] = Query(None, description="Filter by fall detection"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=1000, description="Items per page"),
    current_user: TokenData = Security(get_current_active_user, scopes=["read:events"]),
    db: AsyncSession = Depends(get_db),
):
    """
    List activity events with filtering and pagination.

    Required scope: read:events
    """
    # Build query
    query = select(Event)
    count_query = select(func.count(Event.id))

    conditions = []

    if device_id:
        conditions.append(Event.device_id == device_id)
    if patient_id:
        conditions.append(Event.patient_id == patient_id)
    if activity_label:
        conditions.append(Event.activity_label == activity_label)
    if posture_state:
        conditions.append(Event.posture_state == posture_state)
    if start_time:
        conditions.append(Event.timestamp >= start_time)
    if end_time:
        conditions.append(Event.timestamp <= end_time)
    if min_anomaly_score is not None:
        conditions.append(Event.anomaly_score >= min_anomaly_score)
    if fall_detected is not None:
        conditions.append(Event.fall_detected == fall_detected)

    if conditions:
        query = query.where(and_(*conditions))
        count_query = count_query.where(and_(*conditions))

    # Get total count
    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # Get paginated results
    offset = (page - 1) * page_size
    query = query.order_by(desc(Event.timestamp)).offset(offset).limit(page_size)

    result = await db.execute(query)
    events = result.scalars().all()

    return EventList(
        items=[EventResponse.model_validate(e) for e in events],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{event_id}", response_model=EventResponse)
async def get_event(
    event_id: int,
    current_user: TokenData = Security(get_current_active_user, scopes=["read:events"]),
    db: AsyncSession = Depends(get_db),
):
    """
    Get a specific event by ID.

    Required scope: read:events
    """
    result = await db.execute(select(Event).where(Event.id == event_id))
    event = result.scalar_one_or_none()

    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Event not found",
        )

    return EventResponse.model_validate(event)


@router.post("", response_model=EventResponse, status_code=status.HTTP_201_CREATED)
async def create_event(
    event_data: EventCreate,
    db: AsyncSession = Depends(get_db),
    # Note: Events from edge devices use API key auth, not user auth
):
    """
    Create a new event (from edge device).

    This endpoint is typically called by edge devices via MQTT consumer
    or direct API call with API key authentication.
    """
    event = Event(
        device_id=event_data.device_id,
        patient_id=event_data.patient_id,
        timestamp=event_data.timestamp or datetime.utcnow(),
        activity_label=event_data.activity_label,
        activity_confidence=event_data.activity_confidence,
        posture_state=event_data.posture_state,
        posture_confidence=event_data.posture_confidence,
        agitation_score=event_data.agitation_score,
        pain_score=event_data.pain_score,
        delirium_risk=event_data.delirium_risk,
        anomaly_score=event_data.anomaly_score,
        baseline_state=event_data.baseline_state,
        cluster_id=event_data.cluster_id,
        person_present=event_data.person_present,
        bbox=event_data.bbox,
        track_id=event_data.track_id,
        fall_detected=event_data.fall_detected,
        fall_confidence=event_data.fall_confidence,
        person_on_bed=event_data.person_on_bed,
        person_near_bed=event_data.person_near_bed,
        inference_ms=event_data.inference_ms,
        fps=event_data.fps,
        payload=event_data.payload,
    )

    db.add(event)
    await db.commit()
    await db.refresh(event)

    return EventResponse.model_validate(event)


@router.get("/stats/summary")
async def get_event_stats(
    device_id: Optional[str] = Query(None),
    hours: int = Query(24, ge=1, le=168, description="Hours to analyze"),
    current_user: TokenData = Security(get_current_active_user, scopes=["read:events"]),
    db: AsyncSession = Depends(get_db),
):
    """
    Get event statistics for a time period.

    Required scope: read:events
    """
    since = datetime.utcnow() - timedelta(hours=hours)

    conditions = [Event.timestamp >= since]
    if device_id:
        conditions.append(Event.device_id == device_id)

    # Total events
    total_result = await db.execute(
        select(func.count(Event.id)).where(and_(*conditions))
    )
    total_events = total_result.scalar()

    # Events by activity
    activity_result = await db.execute(
        select(Event.activity_label, func.count(Event.id))
        .where(and_(*conditions))
        .group_by(Event.activity_label)
    )
    activity_counts = {row[0]: row[1] for row in activity_result.all() if row[0]}

    # Events by posture
    posture_result = await db.execute(
        select(Event.posture_state, func.count(Event.id))
        .where(and_(*conditions))
        .group_by(Event.posture_state)
    )
    posture_counts = {row[0]: row[1] for row in posture_result.all() if row[0]}

    # Falls detected
    falls_result = await db.execute(
        select(func.count(Event.id))
        .where(and_(*conditions, Event.fall_detected == True))
    )
    falls_count = falls_result.scalar()

    # Average metrics
    metrics_result = await db.execute(
        select(
            func.avg(Event.anomaly_score),
            func.avg(Event.agitation_score),
            func.avg(Event.fps),
            func.avg(Event.inference_ms),
        )
        .where(and_(*conditions))
    )
    metrics = metrics_result.one()

    return {
        "period_hours": hours,
        "total_events": total_events,
        "activity_distribution": activity_counts,
        "posture_distribution": posture_counts,
        "falls_detected": falls_count,
        "avg_anomaly_score": round(metrics[0] or 0, 3),
        "avg_agitation_score": round(metrics[1] or 0, 3),
        "avg_fps": round(metrics[2] or 0, 1),
        "avg_inference_ms": round(metrics[3] or 0, 1),
    }
