"""Device service - business logic for edge devices."""

from datetime import datetime, timedelta
from typing import Optional
import secrets
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.device import Device, DeviceStatus


class DeviceService:
    """Service for managing edge devices."""

    @staticmethod
    def generate_api_key() -> str:
        """Generate a secure API key for device authentication."""
        return f"eam_{secrets.token_urlsafe(32)}"

    @staticmethod
    async def register_device(
        db: AsyncSession,
        device_id: str,
        name: str,
        location: Optional[str] = None,
        room_id: Optional[str] = None,
        capabilities: Optional[dict] = None,
        config: Optional[dict] = None,
    ) -> Device:
        """Register a new edge device."""
        api_key = DeviceService.generate_api_key()

        device = Device(
            device_id=device_id,
            name=name,
            location=location,
            room_id=room_id,
            api_key=api_key,
            capabilities=capabilities or {},
            config=config or {},
            status=DeviceStatus.OFFLINE,
        )
        db.add(device)
        await db.commit()
        await db.refresh(device)
        return device

    @staticmethod
    async def get_device(db: AsyncSession, device_id: str) -> Optional[Device]:
        """Get device by device_id."""
        result = await db.execute(
            select(Device).where(Device.device_id == device_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_device_by_api_key(db: AsyncSession, api_key: str) -> Optional[Device]:
        """Get device by API key."""
        result = await db.execute(
            select(Device).where(Device.api_key == api_key)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_devices(
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        status: Optional[DeviceStatus] = None,
        room_id: Optional[str] = None,
    ) -> list[Device]:
        """Get devices with filtering."""
        query = select(Device)

        conditions = []
        if status:
            conditions.append(Device.status == status)
        if room_id:
            conditions.append(Device.room_id == room_id)

        if conditions:
            query = query.where(and_(*conditions))

        query = query.order_by(Device.created_at.desc()).offset(skip).limit(limit)
        result = await db.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def update_heartbeat(
        db: AsyncSession,
        device_id: str,
        health_metrics: Optional[dict] = None,
        version: Optional[str] = None,
    ) -> Optional[Device]:
        """Update device heartbeat and health metrics."""
        device = await DeviceService.get_device(db, device_id)
        if not device:
            return None

        device.last_heartbeat = datetime.utcnow()
        device.status = DeviceStatus.ONLINE

        if health_metrics:
            device.health_metrics = health_metrics
        if version:
            device.firmware_version = version

        await db.commit()
        await db.refresh(device)
        return device

    @staticmethod
    async def update_device_status(
        db: AsyncSession,
        device_id: str,
        status: DeviceStatus,
    ) -> Optional[Device]:
        """Update device status."""
        device = await DeviceService.get_device(db, device_id)
        if not device:
            return None

        device.status = status
        await db.commit()
        await db.refresh(device)
        return device

    @staticmethod
    async def update_device_config(
        db: AsyncSession,
        device_id: str,
        config: dict,
    ) -> Optional[Device]:
        """Update device configuration."""
        device = await DeviceService.get_device(db, device_id)
        if not device:
            return None

        device.config = config
        await db.commit()
        await db.refresh(device)
        return device

    @staticmethod
    async def check_stale_devices(
        db: AsyncSession,
        stale_minutes: int = 5,
    ) -> list[Device]:
        """Find and mark stale devices as offline."""
        cutoff = datetime.utcnow() - timedelta(minutes=stale_minutes)

        query = select(Device).where(
            and_(
                Device.status == DeviceStatus.ONLINE,
                Device.last_heartbeat < cutoff
            )
        )

        result = await db.execute(query)
        stale_devices = list(result.scalars().all())

        for device in stale_devices:
            device.status = DeviceStatus.OFFLINE

        await db.commit()
        return stale_devices

    @staticmethod
    async def get_device_health_summary(db: AsyncSession) -> dict:
        """Get overall health summary of all devices."""
        devices = await DeviceService.get_devices(db, limit=1000)

        status_counts = {status.value: 0 for status in DeviceStatus}
        total_cpu = 0
        total_memory = 0
        count_with_metrics = 0

        for device in devices:
            status_counts[device.status.value] += 1
            if device.health_metrics:
                if "cpu_percent" in device.health_metrics:
                    total_cpu += device.health_metrics["cpu_percent"]
                    count_with_metrics += 1
                if "memory_percent" in device.health_metrics:
                    total_memory += device.health_metrics["memory_percent"]

        return {
            "total_devices": len(devices),
            "by_status": status_counts,
            "avg_cpu_percent": round(total_cpu / max(count_with_metrics, 1), 2),
            "avg_memory_percent": round(total_memory / max(count_with_metrics, 1), 2),
        }

    @staticmethod
    async def rotate_api_key(db: AsyncSession, device_id: str) -> Optional[str]:
        """Rotate device API key."""
        device = await DeviceService.get_device(db, device_id)
        if not device:
            return None

        new_key = DeviceService.generate_api_key()
        device.api_key = new_key
        await db.commit()
        return new_key
