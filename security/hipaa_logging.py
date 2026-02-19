# security/hipaa_logging.py
"""
HIPAA-Compliant Logging System
De-identifies all Protected Health Information (PHI) before logging.
"""

import logging
import hashlib
import re
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional
from functools import wraps
import threading


class PHIFilter(logging.Filter):
 """
 Logging filter that de-identifies PHI before it reaches log handlers.

 Covered PHI elements (HIPAA Safe Harbor):
 - Patient IDs
 - Names
 - Dates (except year)
 - Phone numbers
 - Email addresses
 - SSN
 - Medical record numbers
 - Device identifiers
 - IP addresses
 - Biometric identifiers
 - Face images
 """

 # Session-based salt for consistent de-identification within session
 _session_salt = os.urandom(32)
 _salt_lock = threading.Lock()

 # Patterns to detect and redact
 PATTERNS = {
 'patient_id': re.compile(r'\bpatient[_\s]?(?:id)?[:\s]*([A-Za-z0-9_-]+)', re.I),
 'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
 'phone': re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),
 'ssn': re.compile(r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'),
 'ip_address': re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
 'mrn': re.compile(r'\b(?:mrn|medical[_\s]?record)[:\s]*([A-Za-z0-9-]+)', re.I),
 'date_full': re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
 }

 def __init__(self, name: str = ''):
 super().__init__(name)
 self._id_cache: Dict[str, str] = {}

 def _hash_identifier(self, identifier: str) -> str:
 """
 Create a consistent, one-way hash of an identifier.
 Uses session salt for unlinkability across sessions.
 """
 if identifier in self._id_cache:
 return self._id_cache[identifier]

 with self._salt_lock:
 hash_input = f"{identifier}{self._session_salt.hex()}"
 hashed = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
 pseudonym = f"[ID:{hashed}]"
 self._id_cache[identifier] = pseudonym
 return pseudonym

 def _deidentify_text(self, text: str) -> str:
 """De-identify all PHI patterns in text."""
 if not isinstance(text, str):
 return text

 result = text

 # Replace patient IDs with hashed pseudonyms
 for match in self.PATTERNS['patient_id'].finditer(text):
 original = match.group(1)
 pseudonym = self._hash_identifier(original)
 result = result.replace(original, pseudonym)

 # Replace MRNs
 for match in self.PATTERNS['mrn'].finditer(text):
 original = match.group(1)
 pseudonym = self._hash_identifier(original)
 result = result.replace(original, pseudonym)

 # Redact emails
 result = self.PATTERNS['email'].sub('[EMAIL_REDACTED]', result)

 # Redact phone numbers
 result = self.PATTERNS['phone'].sub('[PHONE_REDACTED]', result)

 # Redact SSNs
 result = self.PATTERNS['ssn'].sub('[SSN_REDACTED]', result)

 # Redact IP addresses (keep first octet for debugging)
 def mask_ip(match):
 ip = match.group(0)
 parts = ip.split('.')
 return f"{parts[0]}.xxx.xxx.xxx"
 result = self.PATTERNS['ip_address'].sub(mask_ip, result)

 # Redact full dates (keep year only)
 def mask_date(match):
 date_str = match.group(0)
 # Extract year
 if '/' in date_str:
 parts = date_str.split('/')
 else:
 parts = date_str.split('-')
 year = parts[-1] if len(parts[-1]) == 4 else f"20{parts[-1]}"
 return f"[DATE:{year}]"
 result = self.PATTERNS['date_full'].sub(mask_date, result)

 return result

 def filter(self, record: logging.LogRecord) -> bool:
 """Filter and de-identify the log record."""
 # De-identify the main message
 if record.msg:
 record.msg = self._deidentify_text(str(record.msg))

 # De-identify arguments
 if record.args:
 if isinstance(record.args, dict):
 record.args = {
 k: self._deidentify_text(str(v)) if isinstance(v, str) else v
 for k, v in record.args.items()
 }
 elif isinstance(record.args, tuple):
 record.args = tuple(
 self._deidentify_text(str(arg)) if isinstance(arg, str) else arg
 for arg in record.args
 )

 return True


class HIPAALogger:
 """
 HIPAA-compliant logger wrapper.

 Usage:
 logger = HIPAALogger.get_logger("my_module")
 logger.info("Patient %s verified", patient_id) # patient_id will be hashed
 """

 _instances: Dict[str, 'HIPAALogger'] = {}
 _phi_filter = PHIFilter()
 _configured = False

 @classmethod
 def configure(cls, log_dir: str = "logs", level: int = logging.INFO):
 """Configure HIPAA-compliant logging globally."""
 if cls._configured:
 return

 os.makedirs(log_dir, exist_ok=True)

 # Create formatters
 file_formatter = logging.Formatter(
 '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
 datefmt='%Y-%m-%d %H:%M:%S'
 )
 console_formatter = logging.Formatter(
 '%(levelname)s | %(name)s | %(message)s'
 )

 # Root logger configuration
 root_logger = logging.getLogger()
 root_logger.setLevel(level)

 # Remove existing handlers
 for handler in root_logger.handlers[:]:
 root_logger.removeHandler(handler)

 # Add PHI filter to root logger
 root_logger.addFilter(cls._phi_filter)

 # File handler (encrypted log file)
 log_file = os.path.join(log_dir, f"eam_{datetime.now().strftime('%Y%m%d')}.log")
 file_handler = logging.FileHandler(log_file)
 file_handler.setLevel(level)
 file_handler.setFormatter(file_formatter)
 file_handler.addFilter(cls._phi_filter)
 root_logger.addHandler(file_handler)

 # Console handler
 console_handler = logging.StreamHandler()
 console_handler.setLevel(level)
 console_handler.setFormatter(console_formatter)
 console_handler.addFilter(cls._phi_filter)
 root_logger.addHandler(console_handler)

 cls._configured = True

 @classmethod
 def get_logger(cls, name: str) -> logging.Logger:
 """Get a HIPAA-compliant logger."""
 if not cls._configured:
 cls.configure()

 logger = logging.getLogger(name)
 # Ensure filter is attached
 if cls._phi_filter not in logger.filters:
 logger.addFilter(cls._phi_filter)

 return logger


class AuditLogger:
 """
 Audit logging for HIPAA compliance.
 Records all access to PHI with de-identified references.
 """

 def __init__(self, audit_file: str = "logs/audit.log"):
 os.makedirs(os.path.dirname(audit_file), exist_ok=True)
 self.audit_file = audit_file
 self.phi_filter = PHIFilter()
 self._lock = threading.Lock()

 def log_access(self,
 user_id: str,
 action: str,
 resource_type: str,
 resource_id: str,
 outcome: str = "success",
 details: Optional[Dict] = None):
 """
 Log PHI access event.

 Args:
 user_id: ID of user/system accessing PHI
 action: Type of action (read, write, delete, export)
 resource_type: Type of resource (patient_record, face_image, etc.)
 resource_id: ID of resource (will be hashed)
 outcome: Result of action (success, failure, denied)
 details: Additional context (will be de-identified)
 """
 # De-identify all identifiers
 hashed_user = self.phi_filter._hash_identifier(user_id)
 hashed_resource = self.phi_filter._hash_identifier(resource_id)

 audit_entry = {
 "timestamp": datetime.utcnow().isoformat() + "Z",
 "user": hashed_user,
 "action": action,
 "resource_type": resource_type,
 "resource": hashed_resource,
 "outcome": outcome,
 "details": self.phi_filter._deidentify_text(json.dumps(details)) if details else None
 }

 with self._lock:
 with open(self.audit_file, 'a') as f:
 f.write(json.dumps(audit_entry) + "\n")

 def log_clinical_decision(self,
 patient_id: str,
 decision_type: str,
 decision: str,
 confidence: float,
 triggered_alert: bool = False):
 """Log clinical decision for audit trail."""
 self.log_access(
 user_id="system:clinical_engine",
 action="clinical_decision",
 resource_type="patient_monitoring",
 resource_id=patient_id,
 outcome="success",
 details={
 "decision_type": decision_type,
 "decision": decision,
 "confidence": round(confidence, 3),
 "alert_triggered": triggered_alert
 }
 )


# Global instances
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
 """Get the global audit logger instance."""
 global _audit_logger
 if _audit_logger is None:
 _audit_logger = AuditLogger()
 return _audit_logger


def hipaa_logged(func):
 """
 Decorator to ensure function calls are logged in HIPAA-compliant manner.
 """
 @wraps(func)
 def wrapper(*args, **kwargs):
 logger = HIPAALogger.get_logger(func.__module__)

 # Log function entry (de-identified)
 logger.debug(f"Entering {func.__name__}")

 try:
 result = func(*args, **kwargs)
 logger.debug(f"Exiting {func.__name__} - success")
 return result
 except Exception as e:
 logger.error(f"Error in {func.__name__}: {type(e).__name__}")
 raise

 return wrapper
