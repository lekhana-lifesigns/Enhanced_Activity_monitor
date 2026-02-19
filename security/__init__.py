# security/__init__.py
"""
Security module for HIPAA-compliant clinical monitoring.

Provides:
- Secure credential management (no hardcoded secrets)
- HIPAA-compliant logging with PHI de-identification
- Face image encryption for biometric data protection
"""

from .secure_config import (
 SecureCredentialManager,
 SecureMQTTConfig,
 get_credential_manager,
 require_secure_mode,
)

from .hipaa_logging import (
 HIPAALogger,
 AuditLogger,
 PHIFilter,
 get_audit_logger,
 hipaa_logged,
)

from .face_encryption import (
 FaceImageEncryptor,
 SecurePatientFaceStorage,
 migrate_unencrypted_faces,
)

__all__ = [
 # Credential management
 'SecureCredentialManager',
 'SecureMQTTConfig',
 'get_credential_manager',
 'require_secure_mode',
 # HIPAA logging
 'HIPAALogger',
 'AuditLogger',
 'PHIFilter',
 'get_audit_logger',
 'hipaa_logged',
 # Face encryption
 'FaceImageEncryptor',
 'SecurePatientFaceStorage',
 'migrate_unencrypted_faces',
]
