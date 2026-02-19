"""API v1 router - aggregates all endpoint routers."""
from fastapi import APIRouter

from .endpoints import auth, events, alerts, devices, patients, websocket

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(events.router, prefix="/events", tags=["Events"])
api_router.include_router(alerts.router, prefix="/alerts", tags=["Alerts"])
api_router.include_router(devices.router, prefix="/devices", tags=["Devices"])
api_router.include_router(patients.router, prefix="/patients", tags=["Patients"])
api_router.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])
