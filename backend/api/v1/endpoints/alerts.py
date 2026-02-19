"""Alerts endpoints."""
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status, Security
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc

from backend.core.database import get_db
from backend.core.security import get_current_active_user, TokenData
from backend.models.alert import Alert, AlertLevel, AlertStatus
from backend.schemas.alert import (
    AlertCreate,
    AlertResponse,
    AlertUpdate,
    AlertList,
    AlertAcknowledge,
    AlertResolve,
)

router = APIRouter()


@router.get("", response_model=AlertList)
async def list_alerts(
    device_id: Optional[str] = Query(None),
    patient_id: Optional[str] = Query(None),
    alert_level: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None, alias="status"),
    alert_type: Optional[str] = Query(None),
    active_only: bool = Query(False, description="Show only active alerts"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    current_user: TokenData = Security(get_current_active_user, scopes=["read:alerts"]),
    db: AsyncSession = Depends(get_db),
):
    """
    List alerts with filtering and pagination.

    Required scope: read:alerts
    """
    query = select(Alert)
    count_query = select(func.count(Alert.id))

    conditions = []

    if device_id:
        conditions.append(Alert.device_id == device_id)
    if patient_id:
        conditions.append(Alert.patient_id == patient_id)
    if alert_level:
        conditions.append(Alert.alert_level == AlertLevel(alert_level))
    if status_filter:
        conditions.append(Alert.status == AlertStatus(status_filter))
    if alert_type:
        conditions.append(Alert.alert_type == alert_type)
    if active_only:
        conditions.append(Alert.status == AlertStatus.ACTIVE)

    if conditions:
        query = query.where(and_(*conditions))
        count_query = count_query.where(and_(*conditions))

    # Get counts
    total_result = await db.execute(count_query)
    total = total_result.scalar()

    active_result = await db.execute(
        select(func.count(Alert.id)).where(Alert.status == AlertStatus.ACTIVE)
    )
    active_count = active_result.scalar()

    critical_result = await db.execute(
        select(func.count(Alert.id)).where(
            and_(
                Alert.status == AlertStatus.ACTIVE,
                Alert.alert_level == AlertLevel.CRITICAL,
            )
        )
    )
    critical_count = critical_result.scalar()

    # Get paginated results
    offset = (page - 1) * page_size
    query = query.order_by(desc(Alert.created_at)).offset(offset).limit(page_size)

    result = await db.execute(query)
    alerts = result.scalars().all()

    return AlertList(
        items=[AlertResponse.model_validate(a) for a in alerts],
        total=total,
        page=page,
        page_size=page_size,
        active_count=active_count,
        critical_count=critical_count,
    )


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(
    alert_id: int,
    current_user: TokenData = Security(get_current_active_user, scopes=["read:alerts"]),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific alert by ID."""
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert = result.scalar_one_or_none()

    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    return AlertResponse.model_validate(alert)


@router.post("", response_model=AlertResponse, status_code=status.HTTP_201_CREATED)
async def create_alert(
    alert_data: AlertCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new alert (typically from event processor)."""
    alert = Alert(
        device_id=alert_data.device_id,
        patient_id=alert_data.patient_id,
        alert_level=AlertLevel(alert_data.alert_level),
        alert_type=alert_data.alert_type,
        title=alert_data.title,
        description=alert_data.description,
        status=AlertStatus.ACTIVE,
        confidence=alert_data.confidence,
        agitation_score=alert_data.agitation_score,
        pain_score=alert_data.pain_score,
        delirium_risk=alert_data.delirium_risk,
        anomaly_score=alert_data.anomaly_score,
        activity_label=alert_data.activity_label,
        posture_state=alert_data.posture_state,
        is_policy_violation=alert_data.is_policy_violation,
        violation_type=alert_data.violation_type,
        payload=alert_data.payload,
    )

    db.add(alert)
    await db.commit()
    await db.refresh(alert)

    return AlertResponse.model_validate(alert)


@router.post("/{alert_id}/acknowledge", response_model=AlertResponse)
async def acknowledge_alert(
    alert_id: int,
    ack_data: AlertAcknowledge,
    current_user: TokenData = Security(get_current_active_user, scopes=["manage:alerts"]),
    db: AsyncSession = Depends(get_db),
):
    """
    Acknowledge an alert.

    Required scope: manage:alerts
    """
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert = result.scalar_one_or_none()

    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    if alert.status != AlertStatus.ACTIVE:
        raise HTTPException(
            status_code=400,
            detail=f"Alert is not active (current status: {alert.status.value})",
        )

    alert.status = AlertStatus.ACKNOWLEDGED
    alert.acknowledged_at = datetime.utcnow()
    alert.acknowledged_by = current_user.user_id

    if ack_data.notes:
        alert.resolution_notes = ack_data.notes

    await db.commit()
    await db.refresh(alert)

    return AlertResponse.model_validate(alert)


@router.post("/{alert_id}/resolve", response_model=AlertResponse)
async def resolve_alert(
    alert_id: int,
    resolve_data: AlertResolve,
    current_user: TokenData = Security(get_current_active_user, scopes=["manage:alerts"]),
    db: AsyncSession = Depends(get_db),
):
    """
    Resolve an alert.

    Required scope: manage:alerts
    """
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert = result.scalar_one_or_none()

    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    if alert.status == AlertStatus.RESOLVED:
        raise HTTPException(status_code=400, detail="Alert is already resolved")

    alert.status = AlertStatus.RESOLVED
    alert.resolved_at = datetime.utcnow()
    alert.resolved_by = current_user.user_id
    alert.resolution_notes = resolve_data.resolution_notes

    await db.commit()
    await db.refresh(alert)

    return AlertResponse.model_validate(alert)


@router.post("/{alert_id}/dismiss", response_model=AlertResponse)
async def dismiss_alert(
    alert_id: int,
    current_user: TokenData = Security(get_current_active_user, scopes=["manage:alerts"]),
    db: AsyncSession = Depends(get_db),
):
    """
    Dismiss an alert (mark as false positive).

    Required scope: manage:alerts
    """
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert = result.scalar_one_or_none()

    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert.status = AlertStatus.DISMISSED
    alert.resolved_at = datetime.utcnow()
    alert.resolved_by = current_user.user_id

    await db.commit()
    await db.refresh(alert)

    return AlertResponse.model_validate(alert)
