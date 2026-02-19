"""Devices endpoints."""
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status, Security
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from backend.core.database import get_db
from backend.core.security import get_current_active_user, TokenData
from backend.models.device import Device, DeviceStatus
from backend.schemas.device import (
    DeviceCreate,
    DeviceResponse,
    DeviceUpdate,
    DeviceHealth,
    DeviceList,
)

router = APIRouter()


@router.get("", response_model=DeviceList)
async def list_devices(
    facility_id: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None, alias="status"),
    active_only: bool = Query(True),
    current_user: TokenData = Security(get_current_active_user, scopes=["read:devices"]),
    db: AsyncSession = Depends(get_db),
):
    """
    List all devices with filtering.

    Required scope: read:devices
    """
    query = select(Device)
    conditions = []

    if facility_id:
        conditions.append(Device.facility_id == facility_id)
    if status_filter:
        conditions.append(Device.status == DeviceStatus(status_filter))
    if active_only:
        conditions.append(Device.is_active == True)

    if conditions:
        from sqlalchemy import and_
        query = query.where(and_(*conditions))

    result = await db.execute(query.order_by(Device.device_id))
    devices = result.scalars().all()

    # Count by status
    online = sum(1 for d in devices if d.status == DeviceStatus.ONLINE)
    offline = sum(1 for d in devices if d.status != DeviceStatus.ONLINE)

    return DeviceList(
        items=[DeviceResponse.model_validate(d) for d in devices],
        total=len(devices),
        online_count=online,
        offline_count=offline,
    )


@router.get("/{device_id}", response_model=DeviceResponse)
async def get_device(
    device_id: str,
    current_user: TokenData = Security(get_current_active_user, scopes=["read:devices"]),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific device by ID."""
    result = await db.execute(
        select(Device).where(Device.device_id == device_id)
    )
    device = result.scalar_one_or_none()

    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    return DeviceResponse.model_validate(device)


@router.post("", response_model=DeviceResponse, status_code=status.HTTP_201_CREATED)
async def register_device(
    device_data: DeviceCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Register a new edge device.

    This endpoint is called when a new Jetson comes online.
    """
    # Check if device already exists
    result = await db.execute(
        select(Device).where(Device.device_id == device_data.device_id)
    )
    existing = result.scalar_one_or_none()

    if existing:
        # Update existing device
        existing.status = DeviceStatus.ONLINE
        existing.last_heartbeat = datetime.utcnow()
        if device_data.name:
            existing.name = device_data.name
        if device_data.facility_id:
            existing.facility_id = device_data.facility_id
        if device_data.location:
            existing.location = device_data.location
        if device_data.config:
            existing.config = device_data.config

        await db.commit()
        await db.refresh(existing)
        return DeviceResponse.model_validate(existing)

    # Create new device
    device = Device(
        device_id=device_data.device_id,
        name=device_data.name or f"Device {device_data.device_id}",
        description=device_data.description,
        facility_id=device_data.facility_id,
        location=device_data.location,
        bed_number=device_data.bed_number,
        status=DeviceStatus.ONLINE,
        is_active=True,
        last_heartbeat=datetime.utcnow(),
        config=device_data.config,
    )

    db.add(device)
    await db.commit()
    await db.refresh(device)

    return DeviceResponse.model_validate(device)


@router.put("/{device_id}", response_model=DeviceResponse)
async def update_device(
    device_id: str,
    update_data: DeviceUpdate,
    current_user: TokenData = Security(get_current_active_user, scopes=["manage:devices"]),
    db: AsyncSession = Depends(get_db),
):
    """
    Update device configuration.

    Required scope: manage:devices
    """
    result = await db.execute(
        select(Device).where(Device.device_id == device_id)
    )
    device = result.scalar_one_or_none()

    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    # Update fields
    update_fields = update_data.model_dump(exclude_unset=True)
    for field, value in update_fields.items():
        setattr(device, field, value)

    device.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(device)

    return DeviceResponse.model_validate(device)


@router.post("/{device_id}/heartbeat", response_model=DeviceResponse)
async def device_heartbeat(
    device_id: str,
    health_data: DeviceHealth,
    db: AsyncSession = Depends(get_db),
):
    """
    Device heartbeat - update health metrics.

    Called periodically by edge devices to report health status.
    """
    result = await db.execute(
        select(Device).where(Device.device_id == device_id)
    )
    device = result.scalar_one_or_none()

    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    # Update health metrics
    device.status = DeviceStatus.ONLINE
    device.last_heartbeat = datetime.utcnow()
    device.cpu_percent = health_data.cpu_percent
    device.ram_percent = health_data.ram_percent
    device.temperature = health_data.temperature
    device.fps = health_data.fps
    device.inference_ms = health_data.inference_ms
    device.baseline_state = health_data.baseline_state
    device.baseline_samples = health_data.baseline_samples
    device.firmware_version = health_data.firmware_version
    device.model_version = health_data.model_version

    await db.commit()
    await db.refresh(device)

    return DeviceResponse.model_validate(device)


@router.delete("/{device_id}", status_code=status.HTTP_204_NO_CONTENT)
async def deactivate_device(
    device_id: str,
    current_user: TokenData = Security(get_current_active_user, scopes=["manage:devices"]),
    db: AsyncSession = Depends(get_db),
):
    """
    Deactivate a device (soft delete).

    Required scope: manage:devices
    """
    result = await db.execute(
        select(Device).where(Device.device_id == device_id)
    )
    device = result.scalar_one_or_none()

    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    device.is_active = False
    device.status = DeviceStatus.OFFLINE
    await db.commit()
