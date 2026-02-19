"""Alert service - business logic for clinical alerts."""

from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.alert import Alert, AlertSeverity, AlertStatus
from backend.schemas.alert import AlertCreate


class AlertService:
    """Service for managing clinical alerts."""

    @staticmethod
    async def create_alert(
        db: AsyncSession,
        alert_data: AlertCreate,
        device_id: Optional[str] = None
    ) -> Alert:
        """Create a new alert."""
        alert = Alert(
            device_id=device_id,
            patient_id=alert_data.patient_id,
            alert_type=alert_data.alert_type,
            severity=alert_data.severity,
            title=alert_data.title,
            description=alert_data.description,
            metadata=alert_data.metadata,
        )
        db.add(alert)
        await db.commit()
        await db.refresh(alert)
        return alert

    @staticmethod
    async def get_alert(db: AsyncSession, alert_id: int) -> Optional[Alert]:
        """Get alert by ID."""
        result = await db.execute(select(Alert).where(Alert.id == alert_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_alerts(
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        device_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        status: Optional[AlertStatus] = None,
        active_only: bool = False,
    ) -> list[Alert]:
        """Get alerts with filtering."""
        query = select(Alert)

        conditions = []
        if device_id:
            conditions.append(Alert.device_id == device_id)
        if patient_id:
            conditions.append(Alert.patient_id == patient_id)
        if severity:
            conditions.append(Alert.severity == severity)
        if status:
            conditions.append(Alert.status == status)
        if active_only:
            conditions.append(
                Alert.status.in_([AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED])
            )

        if conditions:
            query = query.where(and_(*conditions))

        query = query.order_by(Alert.created_at.desc()).offset(skip).limit(limit)
        result = await db.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def acknowledge_alert(
        db: AsyncSession,
        alert_id: int,
        user_id: int,
        notes: Optional[str] = None
    ) -> Optional[Alert]:
        """Acknowledge an alert."""
        alert = await AlertService.get_alert(db, alert_id)
        if not alert:
            return None

        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = user_id
        alert.acknowledged_at = datetime.utcnow()
        if notes:
            alert.resolution_notes = notes

        await db.commit()
        await db.refresh(alert)
        return alert

    @staticmethod
    async def resolve_alert(
        db: AsyncSession,
        alert_id: int,
        user_id: int,
        resolution_notes: Optional[str] = None
    ) -> Optional[Alert]:
        """Resolve an alert."""
        alert = await AlertService.get_alert(db, alert_id)
        if not alert:
            return None

        alert.status = AlertStatus.RESOLVED
        alert.resolved_by = user_id
        alert.resolved_at = datetime.utcnow()
        if resolution_notes:
            alert.resolution_notes = resolution_notes

        await db.commit()
        await db.refresh(alert)
        return alert

    @staticmethod
    async def dismiss_alert(
        db: AsyncSession,
        alert_id: int,
        user_id: int,
        reason: Optional[str] = None
    ) -> Optional[Alert]:
        """Dismiss an alert (false positive)."""
        alert = await AlertService.get_alert(db, alert_id)
        if not alert:
            return None

        alert.status = AlertStatus.DISMISSED
        alert.resolved_by = user_id
        alert.resolved_at = datetime.utcnow()
        alert.resolution_notes = reason or "Dismissed as false positive"

        await db.commit()
        await db.refresh(alert)
        return alert

    @staticmethod
    async def get_alert_statistics(
        db: AsyncSession,
        hours: int = 24,
        device_id: Optional[str] = None
    ) -> dict:
        """Get alert statistics for a time period."""
        start_time = datetime.utcnow() - timedelta(hours=hours)

        conditions = [Alert.created_at >= start_time]
        if device_id:
            conditions.append(Alert.device_id == device_id)

        # Count by severity
        severity_query = select(
            Alert.severity,
            func.count(Alert.id).label('count')
        ).where(and_(*conditions)).group_by(Alert.severity)

        severity_result = await db.execute(severity_query)
        severity_counts = {row.severity.value: row.count for row in severity_result.all()}

        # Count by status
        status_query = select(
            Alert.status,
            func.count(Alert.id).label('count')
        ).where(and_(*conditions)).group_by(Alert.status)

        status_result = await db.execute(status_query)
        status_counts = {row.status.value: row.count for row in status_result.all()}

        # Calculate average resolution time
        resolved_query = select(
            func.avg(
                func.julianday(Alert.resolved_at) - func.julianday(Alert.created_at)
            ).label('avg_resolution_days')
        ).where(
            and_(
                *conditions,
                Alert.resolved_at.isnot(None)
            )
        )

        resolved_result = await db.execute(resolved_query)
        avg_resolution = resolved_result.scalar()

        return {
            "period_hours": hours,
            "by_severity": severity_counts,
            "by_status": status_counts,
            "avg_resolution_hours": round(float(avg_resolution or 0) * 24, 2),
            "total": sum(severity_counts.values())
        }

    @staticmethod
    async def escalate_stale_alerts(
        db: AsyncSession,
        stale_minutes: int = 30
    ) -> list[Alert]:
        """Find and escalate alerts that haven't been acknowledged."""
        cutoff = datetime.utcnow() - timedelta(minutes=stale_minutes)

        query = select(Alert).where(
            and_(
                Alert.status == AlertStatus.ACTIVE,
                Alert.created_at < cutoff,
                Alert.severity.in_([AlertSeverity.HIGH, AlertSeverity.CRITICAL])
            )
        )

        result = await db.execute(query)
        stale_alerts = list(result.scalars().all())

        for alert in stale_alerts:
            if alert.severity == AlertSeverity.HIGH:
                alert.severity = AlertSeverity.CRITICAL
            alert.metadata = alert.metadata or {}
            alert.metadata["escalated"] = True
            alert.metadata["escalated_at"] = datetime.utcnow().isoformat()

        await db.commit()
        return stale_alerts
