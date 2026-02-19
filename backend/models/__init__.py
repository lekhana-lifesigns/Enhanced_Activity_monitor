"""Database models for the Enhanced Activity Monitor API.

Models are aligned with the edge cache schema (storage/db.py) to ensure
consistent data between edge devices and the central backend.
"""

from .user import User
from .event import Event
from .alert import Alert, AlertLevel, AlertStatus
from .device import Device, DeviceStatus
from .patient import Patient, PatientStatus

# Models aligned with edge cache schema
from .system_health import SystemHealth
from .patient_session import PatientSession, SessionStatus
from .audit_log import AuditLog
from .patient_consent import PatientConsent
from .patient_baseline import PatientBaseline

__all__ = [
    # Core models
    "User",
    "Event",
    "Alert",
    "AlertLevel",
    "AlertStatus",
    "Device",
    "DeviceStatus",
    "Patient",
    "PatientStatus",
    # Edge-aligned models
    "SystemHealth",
    "PatientSession",
    "SessionStatus",
    "AuditLog",
    "PatientConsent",
    "PatientBaseline",
]
